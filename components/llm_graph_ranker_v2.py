from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope import AutoModelForCausalLM

from .config import logger


class ListwiseInteractionBlock(nn.Module):
    """Candidate self-attention + candidate-history retrieval cross-attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.history_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.retrieval_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_history = nn.LayerNorm(d_model)
        self.norm_retrieval = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        candidates: torch.Tensor,
        history: torch.Tensor,
        retrieval: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.norm_self(candidates)
        attn, _ = self.self_attn(hidden, hidden, hidden, need_weights=False)
        candidates = candidates + self.dropout(attn)

        hidden = self.norm_history(candidates)
        attn, _ = self.history_attn(
            hidden,
            history,
            history,
            key_padding_mask=~history_mask,
            need_weights=False,
        )
        candidates = candidates + self.dropout(attn)

        hidden = self.norm_retrieval(candidates)
        attn, _ = self.retrieval_attn(
            hidden,
            retrieval,
            retrieval,
            need_weights=False,
        )
        candidates = candidates + self.dropout(attn)

        candidates = candidates + self.ffn(self.norm_ffn(candidates))
        return candidates


class LLMGraphRankerV2(nn.Module):
    """Text-aware listwise reranker.

    The original v1 only feeds a minimal instruction prompt to the frozen LLM and
    ranks from the appended GCN vectors. V2 intentionally uses the movie titles in
    the prompt, pools candidate-title hidden states, and fuses them with GCN and
    safe interaction features before a stack of listwise cross-attention blocks.
    """

    def __init__(
        self,
        model_name: str,
        gcn_dim: int,
        num_history: int,
        num_candidates: int,
        match_feature_dim: int,
        interaction_feature_dim: int = 0,
        d_model: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2,
        ce_loss_weight: float = 1.0,
        ranking_loss_weight: float = 0.5,
        ranking_margin: float = 0.1,
        hardest_negative_k: int = 4,
        text_aux_loss_weight: float = 0.10,
        gcn_aux_loss_weight: float = 0.10,
        feature_aux_loss_weight: float = 0.05,
        label_smoothing: float = 0.05,
        **_: object,
    ):
        super().__init__()
        self.model_name = model_name
        self.config = {
            "model_name": model_name,
            "gcn_dim": gcn_dim,
            "num_history": num_history,
            "num_candidates": num_candidates,
            "match_feature_dim": match_feature_dim,
            "interaction_feature_dim": interaction_feature_dim,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "hardest_negative_k": hardest_negative_k,
        }

        model_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=model_dtype,
            device_map="auto",
        )
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            try:
                self.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.llm.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for %s", model_name)
        self.llm_dtype = next(self.llm.parameters()).dtype
        self.device = next(self.llm.parameters()).device
        for parameter in self.llm.parameters():
            parameter.requires_grad = False

        self.hidden_size = int(self.llm.config.hidden_size)
        self.d_model = d_model
        self.num_history = num_history
        self.num_candidates = num_candidates
        self.interaction_feature_dim = interaction_feature_dim
        self.feature_dim = int(match_feature_dim) + int(interaction_feature_dim)
        self.ce_loss_weight = ce_loss_weight
        self.ranking_loss_weight = ranking_loss_weight
        self.ranking_margin = ranking_margin
        self.hardest_negative_k = hardest_negative_k
        self.text_aux_loss_weight = text_aux_loss_weight
        self.gcn_aux_loss_weight = gcn_aux_loss_weight
        self.feature_aux_loss_weight = feature_aux_loss_weight
        self.label_smoothing = label_smoothing

        from gcn_detection import Projector

        self.gcn_projector = Projector(gcn_dim, self.hidden_size)
        self.candidate_text_proj = nn.Linear(self.hidden_size, d_model)
        self.candidate_gcn_proj = nn.Linear(self.hidden_size, d_model)
        self.history_proj = nn.Linear(self.hidden_size, d_model)
        self.retrieval_proj = nn.Linear(self.hidden_size, d_model)
        self.prompt_proj = nn.Linear(self.hidden_size, d_model)

        self.history_position = nn.Parameter(
            torch.randn(1, num_history, d_model) * 0.02
        )
        self.candidate_slot = nn.Parameter(
            torch.randn(1, num_candidates, d_model) * 0.02
        )
        self.history_marker = nn.Parameter(
            torch.randn(1, 1, self.hidden_size) * 0.02
        )
        self.retrieval_marker = nn.Parameter(
            torch.randn(1, 1, self.hidden_size) * 0.02
        )
        self.candidate_marker = nn.Parameter(
            torch.randn(1, 1, self.hidden_size) * 0.02
        )

        self.candidate_input_norm = nn.LayerNorm(d_model)
        self.history_input_norm = nn.LayerNorm(d_model)
        self.retrieval_input_norm = nn.LayerNorm(d_model)
        self.candidate_gcn_gate = nn.Parameter(torch.tensor(1.0))
        self.candidate_text_gate = nn.Parameter(torch.tensor(0.5))
        self.feature_gate = nn.Parameter(torch.tensor(0.5))
        self.prompt_gate = nn.Parameter(torch.tensor(0.5))

        if self.feature_dim > 0:
            self.feature_norm = nn.LayerNorm(self.feature_dim)
            self.feature_mlp = nn.Sequential(
                nn.Linear(self.feature_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            self.feature_score_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
            )
        else:
            self.feature_norm = None
            self.feature_mlp = None
            self.feature_score_head = None

        self.blocks = nn.ModuleList(
            [
                ListwiseInteractionBlock(d_model, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.score_norm = nn.LayerNorm(d_model)
        self.fused_score_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.text_score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.gcn_score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.score_weights = nn.Parameter(torch.tensor([1.0, 0.10, 0.10, 0.20]))

    def _get_base_model(self):
        for attr_name in ("base_model", "model", "transformer"):
            base_model = getattr(self.llm, attr_name, None)
            if base_model is not None:
                return base_model
        raise AttributeError(
            f"Unable to resolve base model for {self.model_name}. "
            "Expected one of: base_model, model, transformer."
        )

    def _forward_base_model(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        return self._get_base_model()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    def _project_gcn(self, embeddings: torch.Tensor) -> torch.Tensor:
        if torch.isnan(embeddings).any():
            embeddings = torch.nan_to_num(embeddings, nan=0.0)
        normalized = F.normalize(embeddings.float(), p=2, dim=-1).to(self.llm_dtype)
        return self.gcn_projector(normalized)

    def _pool_candidate_text(
        self,
        prompt_hidden: torch.Tensor,
        candidate_text_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if candidate_text_mask is None:
            return torch.zeros(
                (prompt_hidden.size(0), self.num_candidates, prompt_hidden.size(-1)),
                device=prompt_hidden.device,
                dtype=torch.float32,
            )
        mask = candidate_text_mask.to(prompt_hidden.device).float()
        if mask.size(-1) != prompt_hidden.size(1):
            if mask.size(-1) > prompt_hidden.size(1):
                mask = mask[:, :, : prompt_hidden.size(1)]
            else:
                pad = torch.zeros(
                    (mask.size(0), mask.size(1), prompt_hidden.size(1) - mask.size(-1)),
                    device=mask.device,
                    dtype=mask.dtype,
                )
                mask = torch.cat([mask, pad], dim=-1)
        counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = torch.bmm(mask, prompt_hidden.float()) / counts
        return pooled

    def _build_feature_embedding(
        self,
        candidate_match_features: Optional[torch.Tensor],
        interaction_features: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.feature_dim <= 0:
            return torch.zeros(
                (batch_size, self.num_candidates, self.d_model),
                device=device,
                dtype=torch.float32,
            )
        match_dim = self.feature_dim - self.interaction_feature_dim
        if candidate_match_features is None:
            match = torch.zeros(
                (batch_size, self.num_candidates, match_dim),
                device=device,
                dtype=torch.float32,
            )
        else:
            match = candidate_match_features.float().to(device)
        if self.interaction_feature_dim <= 0:
            interaction = match.new_zeros((batch_size, self.num_candidates, 0))
        elif interaction_features is None:
            interaction = torch.zeros(
                (batch_size, self.num_candidates, self.interaction_feature_dim),
                device=device,
                dtype=torch.float32,
            )
        else:
            interaction = interaction_features.float().to(device)
        features = torch.cat([match, interaction], dim=-1)
        features = self.feature_norm(features)
        return self.feature_mlp(features)

    def _compute_ranking_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        positive_scores = logits.gather(1, labels.unsqueeze(1))
        negative_mask = torch.ones_like(logits, dtype=torch.bool)
        negative_mask.scatter_(1, labels.unsqueeze(1), False)
        negative_scores = logits.masked_select(negative_mask).view(logits.size(0), -1)
        if self.hardest_negative_k > 0:
            k = min(self.hardest_negative_k, negative_scores.size(1))
            negative_scores, _ = negative_scores.topk(k, dim=1)
            return -F.logsigmoid(positive_scores - negative_scores).mean()
        return -F.logsigmoid(positive_scores - negative_scores).mean()

    def _aux_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        rank = self._compute_ranking_loss(logits, labels)
        return ce + self.ranking_loss_weight * rank

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
        self.gcn_projector = self.gcn_projector.to(current_device).to(self.llm_dtype)
        input_embeddings = self.llm.get_input_embeddings()(input_ids)
        batch_size = input_ids.size(0)
        prompt_length = input_ids.size(1)

        history_embeddings = history_embeddings.float().to(current_device)
        candidate_embeddings = candidate_embeddings.float().to(current_device)
        history_mask = history_mask.to(current_device).bool()
        safe_history_mask = history_mask.clone()
        empty_history = ~safe_history_mask.any(dim=1)
        if empty_history.any():
            safe_history_mask[empty_history, 0] = True

        if history_retrieval_embedding is None:
            history_retrieval_embedding = torch.zeros(
                (batch_size, history_embeddings.size(-1)),
                device=current_device,
                dtype=torch.float32,
            )
        history_retrieval_embedding = history_retrieval_embedding.float().to(current_device)

        history_features = self._project_gcn(history_embeddings)
        retrieval_feature = self._project_gcn(
            history_retrieval_embedding.unsqueeze(1)
        ).squeeze(1)
        candidate_features = self._project_gcn(candidate_embeddings)

        history_marker = self.history_marker.expand(batch_size, -1, -1).to(self.llm_dtype)
        retrieval_marker = self.retrieval_marker.expand(batch_size, -1, -1).to(self.llm_dtype)
        candidate_marker = self.candidate_marker.expand(batch_size, -1, -1).to(self.llm_dtype)
        combined_embeddings = torch.cat(
            [
                input_embeddings,
                history_marker,
                history_features,
                retrieval_marker,
                retrieval_feature.unsqueeze(1),
                candidate_marker,
                candidate_features,
            ],
            dim=1,
        )

        one = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=current_device)
        candidate_attention = torch.ones(
            (batch_size, self.num_candidates),
            dtype=attention_mask.dtype,
            device=current_device,
        )
        combined_attention_mask = torch.cat(
            [
                attention_mask,
                one,
                history_mask.to(attention_mask.dtype),
                one,
                one,
                one,
                candidate_attention,
            ],
            dim=1,
        )

        outputs = self._forward_base_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state
        history_start = prompt_length + 1
        history_end = history_start + self.num_history
        retrieval_index = history_end + 1
        candidate_start = retrieval_index + 2
        candidate_end = candidate_start + self.num_candidates

        prompt_hidden = last_hidden_state[:, :prompt_length, :].detach()
        history_hidden = last_hidden_state[:, history_start:history_end, :].float()
        retrieval_hidden = last_hidden_state[:, retrieval_index, :].float()
        candidate_hidden = last_hidden_state[:, candidate_start:candidate_end, :].float()

        prompt_mask = attention_mask.float().unsqueeze(-1)
        prompt_context = (prompt_hidden.float() * prompt_mask).sum(dim=1)
        prompt_context = prompt_context / prompt_mask.sum(dim=1).clamp_min(1.0)
        candidate_text_hidden = self._pool_candidate_text(prompt_hidden, candidate_text_mask)
        text_valid = (
            candidate_text_mask.to(current_device).float().sum(dim=-1, keepdim=True) > 0
        ).float() if candidate_text_mask is not None else torch.zeros(
            (batch_size, self.num_candidates, 1),
            device=current_device,
            dtype=torch.float32,
        )

        candidate_gcn_emb = self.candidate_gcn_proj(candidate_hidden)
        candidate_text_emb = self.candidate_text_proj(candidate_text_hidden) * text_valid
        feature_emb = self._build_feature_embedding(
            candidate_match_features,
            interaction_features,
            batch_size,
            current_device,
        )
        history_emb = self.history_proj(history_hidden) + self.history_position
        retrieval_emb = self.retrieval_proj(retrieval_hidden)
        retrieval_emb = retrieval_emb + F.softplus(self.prompt_gate) * self.prompt_proj(
            prompt_context
        )

        candidates = (
            F.softplus(self.candidate_gcn_gate) * candidate_gcn_emb
            + F.softplus(self.candidate_text_gate) * candidate_text_emb
            + F.softplus(self.feature_gate) * feature_emb
            + self.candidate_slot
        )
        candidates = self.candidate_input_norm(candidates)
        history = self.history_input_norm(history_emb)
        retrieval = self.retrieval_input_norm(retrieval_emb).unsqueeze(1)

        for block in self.blocks:
            candidates = block(candidates, history, retrieval, safe_history_mask)

        fused_logits = self.fused_score_head(self.score_norm(candidates)).squeeze(-1)
        text_logits = self.text_score_head(candidate_text_emb).squeeze(-1)
        gcn_logits = self.gcn_score_head(candidate_gcn_emb).squeeze(-1)
        feature_logits = (
            self.feature_score_head(feature_emb).squeeze(-1)
            if self.feature_score_head is not None
            else torch.zeros_like(fused_logits)
        )
        weights = self.score_weights
        logits = (
            weights[0] * fused_logits
            + weights[1] * text_logits
            + weights[2] * gcn_logits
            + weights[3] * feature_logits
        )

        loss = None
        ce_loss = None
        ranking_loss = None
        text_aux_loss = None
        gcn_aux_loss = None
        feature_aux_loss = None
        if labels is not None:
            labels = labels.long()
            ce_loss = F.cross_entropy(
                logits, labels, label_smoothing=self.label_smoothing
            )
            ranking_loss = self._compute_ranking_loss(logits, labels)
            loss = self.ce_loss_weight * ce_loss + self.ranking_loss_weight * ranking_loss
            text_aux_loss = self._aux_loss(text_logits, labels)
            gcn_aux_loss = self._aux_loss(gcn_logits, labels)
            loss = loss + self.text_aux_loss_weight * text_aux_loss
            loss = loss + self.gcn_aux_loss_weight * gcn_aux_loss
            if self.feature_score_head is not None:
                feature_aux_loss = self._aux_loss(feature_logits, labels)
                loss = loss + self.feature_aux_loss_weight * feature_aux_loss

        return SimpleNamespace(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            ranking_loss=ranking_loss,
            text_aux_loss=text_aux_loss,
            gcn_aux_loss=gcn_aux_loss,
            feature_aux_loss=feature_aux_loss,
            fused_logits=fused_logits,
            text_logits=text_logits,
            gcn_logits=gcn_logits,
            feature_logits=feature_logits,
        )

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
        return self.forward(
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
        ).logits
