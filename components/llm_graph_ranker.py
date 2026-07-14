from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope import AutoModelForCausalLM

from .config import logger


class LLMGraphRanker(nn.Module):
    def __init__(
        self,
        model_name: str,
        gcn_dim: int,
        num_history: int,
        num_candidates: int,
        match_feature_dim: int,
        interaction_feature_dim: int = 0,
        scoring_mode: str = "gcn",
        ce_loss_weight: float = 1.0,
        ranking_loss_weight: float = 0.2,
        aux_loss_weight: float = 0.5,
        ranking_margin: float = 0.1,
        hardest_negative_k: int = 1,
        gen_loss_weight: float = 0.3,
        gen_inference_weight: float = 0.0,
        gen_loss_warmup_epochs: int = 0,
        label_token_ids: Optional[List[int]] = None,
        answer_prompt_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.model_name = model_name
        model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=model_dtype,
            device_map="auto",
        )
        self.llm_dtype = next(self.llm.parameters()).dtype
        self.device = next(self.llm.parameters()).device
        for param in self.llm.parameters():
            param.requires_grad = False
        self.hidden_size = self.llm.config.hidden_size
        from gcn_detection import Projector
        self.projector = Projector(gcn_dim, self.llm.config.hidden_size)
        self.candidate_scorer = nn.Linear(self.llm.config.hidden_size, 1)
        self.history_aware_scorer = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1),
        )
        self.retrieval_aware_scorer = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1),
        )
        self.candidate_match_scorer = nn.Sequential(
            nn.Linear(match_feature_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1),
        )
        self.interaction_feature_dim = interaction_feature_dim
        if interaction_feature_dim > 0:
            self.interaction_feature_norm = nn.LayerNorm(interaction_feature_dim)
            self.interaction_scorer = nn.Sequential(
                nn.Linear(interaction_feature_dim, self.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size, 1),
            )
            self.interaction_score_weight = nn.Parameter(torch.tensor(0.05))
        else:
            self.interaction_feature_norm = None
            self.interaction_scorer = None
            self.interaction_score_weight = None
        self.mean_history_scorer = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1),
        )
        self.mean_history_weight = nn.Parameter(torch.tensor(0.50))
        self.retrieval_weight = nn.Parameter(torch.tensor(0.50))
        self.match_weight = nn.Parameter(torch.tensor(0.30))
        self.match_feature_norm = nn.LayerNorm(match_feature_dim)
        self.history_position_embeddings = nn.Parameter(
            torch.randn(1, num_history, self.llm.config.hidden_size) * 0.02
        )
        self.inter_candidate_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.inter_candidate_norm = nn.LayerNorm(self.hidden_size)
        self.history_marker = nn.Parameter(
            torch.randn(1, 1, self.llm.config.hidden_size) * 0.02
        )
        self.retrieval_marker = nn.Parameter(
            torch.randn(1, 1, self.llm.config.hidden_size) * 0.02
        )
        self.candidate_marker = nn.Parameter(
            torch.randn(1, 1, self.llm.config.hidden_size) * 0.02
        )
        self.candidate_slot_embeddings = nn.Parameter(
            torch.randn(1, num_candidates, self.llm.config.hidden_size) * 0.02
        )
        self.answer_query = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.num_history = num_history
        self.num_candidates = num_candidates
        self.scoring_mode = scoring_mode
        self.ce_loss_weight = ce_loss_weight
        self.ranking_loss_weight = ranking_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.ranking_margin = ranking_margin
        self.hardest_negative_k = hardest_negative_k
        self.gen_loss_weight = gen_loss_weight
        self.current_gen_loss_weight = gen_loss_weight
        self.gen_inference_weight = gen_inference_weight
        self.gen_loss_warmup_epochs = gen_loss_warmup_epochs
        if label_token_ids is not None:
            self.register_buffer(
                "label_token_ids_buf",
                torch.tensor(label_token_ids, dtype=torch.long),
            )
        else:
            self.label_token_ids_buf = None
        if answer_prompt_ids is not None:
            self.register_buffer(
                "answer_prompt_ids_buf",
                torch.tensor(answer_prompt_ids, dtype=torch.long),
            )
        else:
            self.answer_prompt_ids_buf = None

    def _get_lm_head(self):
        if hasattr(self.llm, "lm_head"):
            return self.llm.lm_head
        raise AttributeError(f"Cannot find lm_head in LLM model {self.model_name}")

    def _get_base_model(self):
        for attr_name in ("base_model", "model", "transformer"):
            base_model = getattr(self.llm, attr_name, None)
            if base_model is not None:
                return base_model
        raise AttributeError(
            f"Unable to resolve base model for {self.model_name}. Expected one of: base_model, model, transformer."
        )

    def _forward_base_model(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        base_model = self._get_base_model()
        return base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    def _project_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if torch.isnan(embeddings).any():
            embeddings = torch.nan_to_num(embeddings, nan=0.0)
        normalized = F.normalize(embeddings.to(self.device), p=2, dim=-1).to(self.llm_dtype)
        return self.projector(normalized)

    def _masked_mean_pool(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(hidden_states.device).float().unsqueeze(-1)
        masked_hidden = hidden_states * mask
        token_counts = mask.sum(dim=1).clamp_min(1.0)
        return masked_hidden.sum(dim=1) / token_counts

    def _score_with_history(
        self,
        candidate_hidden: torch.Tensor,
        history_hidden: torch.Tensor,
        history_mask: torch.Tensor,
        retrieval_hidden: Optional[torch.Tensor] = None,
        candidate_match_features: Optional[torch.Tensor] = None,
        interaction_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        expanded_candidates = candidate_hidden.unsqueeze(2).expand(
            -1,
            -1,
            history_hidden.size(1),
            -1,
        )
        expanded_history = history_hidden.unsqueeze(1).expand(
            -1,
            candidate_hidden.size(1),
            -1,
            -1,
        )
        pair_scorer_input = torch.cat(
            [
                expanded_candidates,
                expanded_history,
                expanded_candidates * expanded_history,
                expanded_candidates - expanded_history,
            ],
            dim=-1,
        )
        pair_scores = self.history_aware_scorer(pair_scorer_input).squeeze(-1)
        history_valid_mask = history_mask.to(pair_scores.device).bool()
        expanded_valid_mask = history_valid_mask.unsqueeze(1).expand_as(pair_scores)
        attention_logits = pair_scores.masked_fill(~expanded_valid_mask, -1e4)
        attention_weights = F.softmax(attention_logits, dim=2)
        attention_weights = attention_weights * expanded_valid_mask.float()
        attn_norm = attention_weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        attention_weights = attention_weights / attn_norm
        attn_scores = (attention_weights * pair_scores).sum(dim=2)
        no_history_mask = ~history_valid_mask.any(dim=1)
        attn_scores = attn_scores.masked_fill(no_history_mask.unsqueeze(1), 0.0)
        candidate_scores = attn_scores
        history_float_mask = history_valid_mask.float()
        seq_len = history_hidden.size(1)
        position_idx = torch.arange(seq_len, device=history_hidden.device, dtype=torch.float32)
        recency_weights = torch.exp(position_idx - (seq_len - 1))
        recency_weights = recency_weights.unsqueeze(0) * history_float_mask
        recency_norm = recency_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        recency_weights = recency_weights / recency_norm
        mean_history = (history_hidden * recency_weights.unsqueeze(-1)).sum(dim=1)
        expanded_mean = mean_history.unsqueeze(1).expand(-1, candidate_hidden.size(1), -1)
        mean_path_input = torch.cat(
            [
                candidate_hidden,
                expanded_mean,
                candidate_hidden * expanded_mean,
                candidate_hidden - expanded_mean,
            ],
            dim=-1,
        )
        mean_path_scores = self.mean_history_scorer(mean_path_input).squeeze(-1)
        mean_path_scores = mean_path_scores.masked_fill(no_history_mask.unsqueeze(1), 0.0)
        mean_weight = F.softplus(self.mean_history_weight)
        candidate_scores = candidate_scores + mean_weight * mean_path_scores

        if retrieval_hidden is not None:
            expanded_retrieval = retrieval_hidden.unsqueeze(1).expand(-1, candidate_hidden.size(1), -1)
            retrieval_input = torch.cat(
                [
                    candidate_hidden,
                    expanded_retrieval,
                    candidate_hidden * expanded_retrieval,
                    candidate_hidden - expanded_retrieval,
                ],
                dim=-1,
            )
            retrieval_scores = self.retrieval_aware_scorer(retrieval_input).squeeze(-1)
            ret_weight = F.softplus(self.retrieval_weight)
            candidate_scores = candidate_scores + ret_weight * retrieval_scores
        if candidate_match_features is not None:
            match_scores = self.candidate_match_scorer(
                self.match_feature_norm(candidate_match_features.float())
            ).squeeze(-1)
            m_weight = F.softplus(self.match_weight)
            candidate_scores = candidate_scores + m_weight * match_scores

        base_candidate_scores = candidate_scores.clone()
        if (
            interaction_features is not None
            and self.interaction_scorer is not None
            and self.interaction_feature_norm is not None
            and self.interaction_score_weight is not None
        ):
            inter_scores = self.interaction_scorer(
                self.interaction_feature_norm(interaction_features.float())
            ).squeeze(-1)
            inter_weight = F.softplus(self.interaction_score_weight)
            candidate_scores = candidate_scores + inter_weight * inter_scores
            if self.training:
                return candidate_scores, base_candidate_scores
            else:
                return candidate_scores
        return candidate_scores

    def _compute_ranking_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        positive_scores = logits.gather(1, labels.unsqueeze(1))
        negative_mask = torch.ones_like(logits, dtype=torch.bool)
        negative_mask.scatter_(1, labels.unsqueeze(1), False)
        negative_scores = logits.masked_select(negative_mask).view(logits.size(0), -1)
        if self.hardest_negative_k > 0:
            k = min(self.hardest_negative_k, negative_scores.size(1))
            negative_scores, _ = negative_scores.topk(k, dim=1)
        bpr_loss = -F.logsigmoid(positive_scores - negative_scores).mean()
        return bpr_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_retrieval_embedding: Optional[torch.Tensor],
        candidate_embeddings: torch.Tensor,
        candidate_match_features: Optional[torch.Tensor] = None,
        interaction_features: Optional[torch.Tensor] = None,
        candidate_text_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        current_device = input_ids.device
        self.projector = self.projector.to(current_device).to(self.llm_dtype)
        input_embeddings = self.llm.get_input_embeddings()(input_ids)
        gen_logits = None
        if self.scoring_mode == "text_only":
            outputs = self._forward_base_model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
            )
            last_hidden_state = outputs.last_hidden_state.float()
            if candidate_text_mask is None:
                raise ValueError("candidate_text_mask is required for text_only scoring mode")
            candidate_token_mask = candidate_text_mask.to(last_hidden_state.device)
            token_counts = candidate_token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            pooled_candidate_hidden = (
                last_hidden_state.unsqueeze(1) * candidate_token_mask.unsqueeze(-1)
            ).sum(dim=2) / token_counts
            logits = self.candidate_scorer(pooled_candidate_hidden).squeeze(-1)
        else:
            history_features = self._project_embeddings(history_embeddings)
            hist_valid = history_mask.float().to(history_features.device).unsqueeze(-1)
            history_features = history_features + (
                self.history_position_embeddings.expand(input_ids.size(0), -1, -1).to(self.llm_dtype)
                * hist_valid.to(self.llm_dtype)
            )
            if history_retrieval_embedding is None:
                history_retrieval_embedding = torch.zeros(
                    (history_embeddings.size(0), history_embeddings.size(-1)),
                    device=history_embeddings.device,
                    dtype=history_embeddings.dtype,
                )
            retrieval_feature = self._project_embeddings(history_retrieval_embedding.unsqueeze(1)).squeeze(1)
            candidate_features = self._project_embeddings(candidate_embeddings)
            candidate_features = candidate_features + self.candidate_slot_embeddings.to(self.llm_dtype)
            if self.answer_prompt_ids_buf is None:
                raise ValueError("answer_prompt_ids is required for generative scoring in gcn mode")
            if self.label_token_ids_buf is None:
                raise ValueError("label_token_ids is required for generative scoring in gcn mode")
            candidate_label_ids = self.label_token_ids_buf.unsqueeze(0).expand(input_ids.size(0), -1)
            candidate_label_embeddings = self.llm.get_input_embeddings()(candidate_label_ids)
            candidate_segment_embeddings = torch.stack(
                [candidate_label_embeddings, candidate_features],
                dim=2,
            ).reshape(input_ids.size(0), self.num_candidates * 2, self.hidden_size)
            answer_prompt_ids = self.answer_prompt_ids_buf.unsqueeze(0).expand(input_ids.size(0), -1)
            answer_prompt_embeddings = self.llm.get_input_embeddings()(answer_prompt_ids)
            answer_prompt_len = answer_prompt_embeddings.size(1)
            history_marker = self.history_marker.expand(input_ids.size(0), -1, -1).to(self.llm_dtype)
            retrieval_marker = self.retrieval_marker.expand(input_ids.size(0), -1, -1).to(self.llm_dtype)
            candidate_marker = self.candidate_marker.expand(input_ids.size(0), -1, -1).to(self.llm_dtype)
            answer_query = self.answer_query.expand(input_ids.size(0), -1, -1).to(self.llm_dtype)
            combined_embeddings = torch.cat(
                [
                    input_embeddings,
                    history_marker,
                    history_features,
                    retrieval_marker,
                    retrieval_feature.unsqueeze(1),
                    candidate_marker,
                    candidate_segment_embeddings,
                    answer_prompt_embeddings,
                    answer_query,
                ],
                dim=1,
            )
            marker_mask = torch.ones((input_ids.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
            retrieval_mask = torch.ones((input_ids.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
            candidate_mask = torch.ones((input_ids.size(0), self.num_candidates * 2), dtype=attention_mask.dtype, device=attention_mask.device)
            answer_prompt_mask = torch.ones((input_ids.size(0), answer_prompt_len), dtype=attention_mask.dtype, device=attention_mask.device)
            answer_query_mask = torch.ones((input_ids.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
            history_llm_mask = torch.ones((input_ids.size(0), self.num_history), dtype=attention_mask.dtype, device=attention_mask.device)
            combined_attention_mask = torch.cat(
                [
                    attention_mask,
                    marker_mask,
                    history_llm_mask,
                    marker_mask,
                    retrieval_mask,
                    marker_mask,
                    candidate_mask,
                    answer_prompt_mask,
                    answer_query_mask,
                ],
                dim=1,
            )
            outputs = self._forward_base_model(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
            )
            last_hidden_state = outputs.last_hidden_state
            prompt_length = input_ids.size(1)
            history_start = prompt_length + 1
            history_end = history_start + self.num_history
            retrieval_index = history_end + 1
            candidate_start = retrieval_index + 2
            candidate_end = candidate_start + self.num_candidates * 2
            answer_query_index = candidate_end + answer_prompt_len
            history_hidden = last_hidden_state[:, history_start:history_end, :].float()
            retrieval_hidden = last_hidden_state[:, retrieval_index, :].float()
            candidate_segment_hidden = last_hidden_state[:, candidate_start:candidate_end, :].float()
            candidate_hidden = candidate_segment_hidden[:, 1::2, :]
            attn_out, _ = self.inter_candidate_attn(candidate_hidden, candidate_hidden, candidate_hidden)
            candidate_hidden = self.inter_candidate_norm(candidate_hidden + attn_out)
            if self.label_token_ids_buf is not None:
                answer_hidden = last_hidden_state[:, answer_query_index, :].to(self.llm_dtype)
                temperature = 1.0
                gen_vocab_logits = self._get_lm_head()(answer_hidden).float() / temperature
                gen_logits = gen_vocab_logits[:, self.label_token_ids_buf]
            if self.training:
                logits, base_logits = self._score_with_history(
                    candidate_hidden,
                    history_hidden,
                    history_mask,
                    retrieval_hidden=retrieval_hidden,
                    candidate_match_features=candidate_match_features,
                    interaction_features=interaction_features,
                )
            else:
                logits = self._score_with_history(
                    candidate_hidden,
                    history_hidden,
                    history_mask,
                    retrieval_hidden=retrieval_hidden,
                    candidate_match_features=candidate_match_features,
                    interaction_features=interaction_features,
                )
                base_logits = None
        loss = None
        gen_loss = None
        aux_loss = None
        if labels is not None:
            labels = labels.long()
            ce_loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            ranking_loss = self._compute_ranking_loss(logits, labels)
            loss = (self.ce_loss_weight * ce_loss) + (self.ranking_loss_weight * ranking_loss)
            if base_logits is not None:
                base_ce_loss = F.cross_entropy(base_logits, labels, label_smoothing=0.05)
                base_ranking_loss = self._compute_ranking_loss(base_logits, labels)
                aux_loss = self.aux_loss_weight * ((self.ce_loss_weight * base_ce_loss) + (self.ranking_loss_weight * base_ranking_loss))
                loss = loss + aux_loss
            if gen_logits is not None:
                gen_loss = F.cross_entropy(gen_logits, labels)
                loss = loss + self.current_gen_loss_weight * gen_loss
        else:
            ce_loss = None
            ranking_loss = None
        return type(
            "Output",
            (),
            {
                "logits": logits,
                "loss": loss,
                "ce_loss": ce_loss,
                "ranking_loss": ranking_loss,
                "aux_loss": aux_loss,
                "gen_loss": gen_loss,
                "gen_logits": gen_logits,
            },
        )()

    @torch.no_grad()
    def inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_retrieval_embedding: Optional[torch.Tensor],
        candidate_embeddings: torch.Tensor,
        candidate_match_features: Optional[torch.Tensor] = None,
        interaction_features: Optional[torch.Tensor] = None,
        candidate_text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            history_embeddings=history_embeddings,
            history_mask=history_mask,
            history_retrieval_embedding=history_retrieval_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_match_features=candidate_match_features,
            interaction_features=interaction_features,
            candidate_text_mask=candidate_text_mask,
            labels=None,
        )
        logits = outputs.logits
        gen_logits = getattr(outputs, "gen_logits", None)
        if gen_logits is not None and self.gen_inference_weight > 0:
            gen_log_probs = F.log_softmax(gen_logits, dim=-1)
            logits = logits + self.gen_inference_weight * gen_log_probs
        return logits
