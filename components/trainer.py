import json
import os
import time
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import logger


def compute_grad_norm(parameter: Optional[torch.nn.Parameter]) -> float:
    if parameter is None or parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().norm().item())


class GCNRecommendationTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        device=None,
        model_save_path="./best_gcn_ranker_head.pth",
        metrics_log_path: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.best_loss = float("inf")
        self.best_top1 = 0.0
        self.model_save_path = model_save_path
        self.metrics_log_path = metrics_log_path

    def _append_metrics_event(self, event: str, payload: Dict):
        if not self.metrics_log_path:
            return
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "event": event,
        }
        record.update(payload)
        try:
            os.makedirs(os.path.dirname(self.metrics_log_path), exist_ok=True)
            with open(self.metrics_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to append metrics event to %s: %s", self.metrics_log_path, exc)

    def save_probehead(self, path=None):
        path = path or self.model_save_path
        state_dict = {
            "model_name": self.model.model_name,
            "projector": self.model.projector.state_dict(),
            "candidate_scorer": self.model.candidate_scorer.state_dict(),
            "history_aware_scorer": self.model.history_aware_scorer.state_dict(),
            "retrieval_aware_scorer": self.model.retrieval_aware_scorer.state_dict(),
            "candidate_match_scorer": self.model.candidate_match_scorer.state_dict(),
            "interaction_feature_dim": int(getattr(self.model, "interaction_feature_dim", 0) or 0),
            "mean_history_scorer": self.model.mean_history_scorer.state_dict(),
            "match_feature_norm": self.model.match_feature_norm.state_dict(),
            "history_position_embeddings": self.model.history_position_embeddings.detach().cpu(),
            "inter_candidate_attn": self.model.inter_candidate_attn.state_dict(),
            "inter_candidate_norm": self.model.inter_candidate_norm.state_dict(),
            "history_marker": self.model.history_marker.detach().cpu(),
            "retrieval_marker": self.model.retrieval_marker.detach().cpu(),
            "candidate_marker": self.model.candidate_marker.detach().cpu(),
            "candidate_slot_embeddings": self.model.candidate_slot_embeddings.detach().cpu(),
            "answer_query": self.model.answer_query.detach().cpu(),
            "mean_history_weight": self.model.mean_history_weight.detach().cpu(),
            "retrieval_weight": self.model.retrieval_weight.detach().cpu(),
            "match_weight": self.model.match_weight.detach().cpu(),
        }
        if getattr(self.model, "interaction_feature_norm", None) is not None:
            state_dict["interaction_feature_norm"] = self.model.interaction_feature_norm.state_dict()
        if getattr(self.model, "interaction_scorer", None) is not None:
            state_dict["interaction_scorer"] = self.model.interaction_scorer.state_dict()
        if getattr(self.model, "interaction_score_weight", None) is not None:
            state_dict["interaction_score_weight"] = self.model.interaction_score_weight.detach().cpu()
        torch.save(state_dict, path)
        logger.info("Ranker head saved to %s", path)

    def load_probehead(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint_model_name = checkpoint.get("model_name")
        if checkpoint_model_name is not None and checkpoint_model_name != self.model.model_name:
            logger.warning(
                "Skip loading probe head from %s because checkpoint model_name=%s but current model_name=%s",
                path,
                checkpoint_model_name,
                self.model.model_name,
            )
            return False
        def load_if(name, target):
            if name in checkpoint:
                target.load_state_dict(checkpoint[name])
        load_if("projector", self.model.projector)
        load_if("candidate_scorer", self.model.candidate_scorer)
        load_if("history_aware_scorer", self.model.history_aware_scorer)
        load_if("retrieval_aware_scorer", self.model.retrieval_aware_scorer)
        load_if("candidate_match_scorer", self.model.candidate_match_scorer)
        load_if("mean_history_scorer", self.model.mean_history_scorer)
        load_if("match_feature_norm", self.model.match_feature_norm)
        if "history_marker" in checkpoint:
            self.model.history_marker.data.copy_(checkpoint["history_marker"].to(self.device))
        if "retrieval_marker" in checkpoint:
            self.model.retrieval_marker.data.copy_(checkpoint["retrieval_marker"].to(self.device))
        if "candidate_marker" in checkpoint:
            self.model.candidate_marker.data.copy_(checkpoint["candidate_marker"].to(self.device))
        if "candidate_slot_embeddings" in checkpoint:
            self.model.candidate_slot_embeddings.data.copy_(checkpoint["candidate_slot_embeddings"].to(self.device))
        if "answer_query" in checkpoint:
            self.model.answer_query.data.copy_(checkpoint["answer_query"].to(self.device))
        if "mean_history_weight" in checkpoint:
            self.model.mean_history_weight.data.copy_(checkpoint["mean_history_weight"].to(self.device))
        if "retrieval_weight" in checkpoint:
            self.model.retrieval_weight.data.copy_(checkpoint["retrieval_weight"].to(self.device))
        if "match_weight" in checkpoint:
            self.model.match_weight.data.copy_(checkpoint["match_weight"].to(self.device))
        if "interaction_feature_norm" in checkpoint and getattr(self.model, "interaction_feature_norm", None) is not None:
            self.model.interaction_feature_norm.load_state_dict(checkpoint["interaction_feature_norm"])
        if "interaction_scorer" in checkpoint and getattr(self.model, "interaction_scorer", None) is not None:
            self.model.interaction_scorer.load_state_dict(checkpoint["interaction_scorer"])
        if "interaction_score_weight" in checkpoint and getattr(self.model, "interaction_score_weight", None) is not None:
            self.model.interaction_score_weight.data.copy_(checkpoint["interaction_score_weight"].to(self.device))
        if "history_position_embeddings" in checkpoint:
            self.model.history_position_embeddings.data.copy_(checkpoint["history_position_embeddings"].to(self.device))
        if "inter_candidate_attn" in checkpoint:
            self.model.inter_candidate_attn.load_state_dict(checkpoint["inter_candidate_attn"])
        if "inter_candidate_norm" in checkpoint:
            self.model.inter_candidate_norm.load_state_dict(checkpoint["inter_candidate_norm"])
        logger.info("Ranker head loaded from %s", path)
        return True

    def _set_trainable(self):
        self.model.projector.requires_grad_(True)
        self.model.candidate_scorer.requires_grad_(True)
        self.model.history_aware_scorer.requires_grad_(True)
        self.model.retrieval_aware_scorer.requires_grad_(True)
        self.model.candidate_match_scorer.requires_grad_(True)
        self.model.mean_history_scorer.requires_grad_(True)
        self.model.match_feature_norm.requires_grad_(True)
        self.model.history_marker.requires_grad_(True)
        self.model.retrieval_marker.requires_grad_(True)
        self.model.candidate_marker.requires_grad_(True)
        self.model.candidate_slot_embeddings.requires_grad_(True)
        self.model.answer_query.requires_grad_(True)
        self.model.mean_history_weight.requires_grad_(True)
        self.model.retrieval_weight.requires_grad_(True)
        self.model.match_weight.requires_grad_(True)
        if getattr(self.model, "interaction_feature_norm", None) is not None:
            self.model.interaction_feature_norm.requires_grad_(True)
        if getattr(self.model, "interaction_scorer", None) is not None:
            self.model.interaction_scorer.requires_grad_(True)
        if getattr(self.model, "interaction_score_weight", None) is not None:
            self.model.interaction_score_weight.requires_grad_(True)

    def train(self, dataloader, epochs=3, lr=5e-5, eval_dataloader=None):
        self._set_trainable()
        optimizer = torch.optim.AdamW(
            filter(lambda param: param.requires_grad, self.model.parameters()),
            lr=lr,
        )
        total_steps = len(dataloader) * epochs
        warmup_steps = max(1, total_steps // 10)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
        )

        for epoch in range(epochs):
            if self.model.gen_loss_warmup_epochs > 0:
                warmup_ratio = min(1.0, float(epoch + 1) / float(self.model.gen_loss_warmup_epochs))
                self.model.current_gen_loss_weight = self.model.gen_loss_weight * warmup_ratio
            else:
                self.model.current_gen_loss_weight = self.model.gen_loss_weight
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                history_embeddings = batch["history_embeddings"].to(self.device)
                history_mask = batch["history_mask"].to(self.device)
                history_retrieval_embedding = batch["history_retrieval_embedding"].to(self.device)
                candidate_embeddings = batch["candidate_embeddings"].to(self.device)
                candidate_match_features = batch["candidate_match_features"].to(self.device)
                interaction_features = batch["interaction_features"].to(self.device)
                candidate_text_mask = batch["candidate_text_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    history_embeddings=history_embeddings,
                    history_mask=history_mask,
                    history_retrieval_embedding=history_retrieval_embedding,
                    candidate_embeddings=candidate_embeddings,
                    candidate_match_features=candidate_match_features,
                    interaction_features=interaction_features,
                    candidate_text_mask=candidate_text_mask,
                    labels=labels,
                )
                loss = outputs.loss
                if loss is None or torch.isnan(loss):
                    logger.error("Invalid loss detected at batch %s", batch_idx)
                    continue
                loss.backward()
                if batch_idx == 0 or ((batch_idx + 1) % 200 == 0):
                    self._log_gen_debug(epoch, batch_idx, outputs, labels)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                pbar.set_postfix({
                    "batch_loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

            avg_loss = total_loss / len(dataloader)
            logger.info("Epoch %s completed. Average loss: %.4f", epoch + 1, avg_loss)
            self._append_metrics_event(
                "train_epoch_end",
                {
                    "epoch": int(epoch + 1),
                    "avg_loss": float(avg_loss),
                    "gen_loss_weight": float(self.model.current_gen_loss_weight),
                },
            )
            if eval_dataloader is not None:
                eval_metrics = self.evaluate(eval_dataloader)
                self._append_metrics_event(
                    "eval_after_train_epoch",
                    {
                        "epoch": int(epoch + 1),
                        "top1_acc": float(eval_metrics["top1_acc"]),
                        "top3_acc": float(eval_metrics["top3_acc"]),
                        "top5_acc": float(eval_metrics["top5_acc"]),
                    },
                )
                if eval_metrics["top1_acc"] > self.best_top1:
                    self.best_top1 = eval_metrics["top1_acc"]
                    self.save_probehead()
                    logger.info("New best top-1: %.4f, checkpoint saved", self.best_top1)
            else:
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save_probehead()

    def _build_gen_debug_info(self, outputs, labels: torch.Tensor):
        if outputs.gen_logits is None:
            return None
        with torch.no_grad():
            gen_logits = outputs.gen_logits.float()
            gen_probs = F.softmax(gen_logits, dim=-1)
            batch_indices = torch.arange(labels.size(0), device=labels.device)
            target_probs = gen_probs[batch_indices, labels]
            target_logits = gen_logits[batch_indices, labels]
            masked_logits = gen_logits.clone()
            masked_logits[batch_indices, labels] = float("-inf")
            best_neg_logits = masked_logits.max(dim=-1).values
            entropy = -(gen_probs * gen_probs.clamp_min(1e-8).log()).sum(dim=-1)
            prediction = gen_logits.argmax(dim=-1)
            return {
                "target_prob": float(target_probs.mean().item()),
                "top1": float((prediction == labels).float().mean().item()),
                "entropy": float(entropy.mean().item()),
                "margin": float((target_logits - best_neg_logits).mean().item()),
                "logit_std": float(gen_logits.std(dim=-1).mean().item()),
            }

    def _log_gen_debug(self, epoch: int, batch_idx: int, outputs, labels: torch.Tensor):
        debug_info = self._build_gen_debug_info(outputs, labels)
        if debug_info is None:
            return
        answer_query_grad = compute_grad_norm(self.model.answer_query)
        projector_grad = 0.0
        for parameter in self.model.projector.parameters():
            projector_grad += compute_grad_norm(parameter)
        logger.info(
            "Gen debug e%s b%s: target_prob=%.4f top1=%.4f entropy=%.4f margin=%.4f logit_std=%.4f answer_grad=%.4e projector_grad=%.4e",
            epoch + 1,
            batch_idx,
            debug_info["target_prob"],
            debug_info["top1"],
            debug_info["entropy"],
            debug_info["margin"],
            debug_info["logit_std"],
            answer_query_grad,
            projector_grad,
        )

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        top1_total = 0
        top3_total = 0
        top5_total = 0
        sample_count = 0
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            history_embeddings = batch["history_embeddings"].to(self.device)
            history_mask = batch["history_mask"].to(self.device)
            history_retrieval_embedding = batch["history_retrieval_embedding"].to(self.device)
            candidate_embeddings = batch["candidate_embeddings"].to(self.device)
            candidate_match_features = batch["candidate_match_features"].to(self.device)
            interaction_features = batch["interaction_features"].to(self.device)
            candidate_text_mask = batch["candidate_text_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            logits = self.model.inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                history_embeddings=history_embeddings,
                history_mask=history_mask,
                history_retrieval_embedding=history_retrieval_embedding,
                candidate_embeddings=candidate_embeddings,
                candidate_match_features=candidate_match_features,
                interaction_features=interaction_features,
                candidate_text_mask=candidate_text_mask,
            )
            top1_pred = torch.argmax(logits, dim=1)
            top3_pred = torch.topk(logits, k=min(3, logits.size(1)), dim=1).indices
            top5_pred = torch.topk(logits, k=min(5, logits.size(1)), dim=1).indices
            top1_total += (top1_pred == labels).sum().item()
            top3_total += (top3_pred == labels.unsqueeze(1)).any(dim=1).sum().item()
            top5_total += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum().item()
            sample_count += labels.size(0)

        metrics = {
            "model_name": self.model.model_name,
            "top1_acc": top1_total / sample_count if sample_count else 0.0,
            "top3_acc": top3_total / sample_count if sample_count else 0.0,
            "top5_acc": top5_total / sample_count if sample_count else 0.0,
        }
        logger.info(
            "Evaluation completed. Top-1: %.4f, Top-3: %.4f, Top-5: %.4f",
            metrics["top1_acc"],
            metrics["top3_acc"],
            metrics["top5_acc"],
        )
        self._append_metrics_event(
            "evaluate",
            {
                "top1_acc": float(metrics["top1_acc"]),
                "top3_acc": float(metrics["top3_acc"]),
                "top5_acc": float(metrics["top5_acc"]),
            },
        )
        return metrics
