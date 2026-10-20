import json
import math
import os
import time
from typing import Dict, Optional

import torch
from tqdm import tqdm

from .config import logger


class GCNRecommendationTrainerV2:
    def __init__(
        self,
        model,
        tokenizer,
        device=None,
        model_save_path="./best_gcn_ranker_v2.pth",
        metrics_log_path: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.best_loss = float("inf")
        self.best_top1 = 0.0
        self.best_epoch = 0
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
            with open(self.metrics_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to append metrics event: %s", exc)

    def save_checkpoint(self, path=None):
        path = path or self.model_save_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.model.state_dict().items()
            if not key.startswith("llm.")
        }
        torch.save(
            {
                "model_name": self.model.model_name,
                "config": getattr(self.model, "config", {}),
                "state_dict": state_dict,
            },
            path,
        )
        logger.info("V2 checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> bool:
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint_model_name = checkpoint.get("model_name")
        if checkpoint_model_name is not None and checkpoint_model_name != self.model.model_name:
            logger.warning(
                "Skip v2 checkpoint %s because model_name=%s, expected=%s",
                path,
                checkpoint_model_name,
                self.model.model_name,
            )
            return False
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        missing = [key for key in missing if not key.startswith("llm.")]
        if missing:
            logger.warning("V2 checkpoint missing keys: %s", missing[:12])
        if unexpected:
            logger.warning("V2 checkpoint unexpected keys: %s", unexpected[:12])
        logger.info("V2 checkpoint loaded from %s", path)
        return True

    def _set_trainable(self):
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(not name.startswith("llm."))

    def train(
        self,
        dataloader,
        epochs=3,
        lr=2e-5,
        eval_dataloader=None,
        weight_decay=0.01,
        early_stopping_patience=2,
        early_stopping_min_delta=1e-4,
        grad_accum_steps=1,
        max_grad_norm=1.0,
    ):
        self._set_trainable()
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters found for v2 model")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
        grad_accum_steps = max(1, int(grad_accum_steps))
        steps_per_epoch = max(1, math.ceil(len(dataloader) / grad_accum_steps))
        total_steps = max(1, steps_per_epoch * epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=max(0.03, min(0.1, 1.0 / max(1, epochs * 4))),
            anneal_strategy="cos",
        )
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            valid_batches = 0
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(dataloader, desc=f"V2 Epoch {epoch + 1}/{epochs}")
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
                if loss is None or not torch.isfinite(loss):
                    logger.error("Invalid v2 loss at epoch=%s batch=%s", epoch + 1, batch_idx)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                (loss / grad_accum_steps).backward()
                should_step = (
                    (batch_idx + 1) % grad_accum_steps == 0
                    or (batch_idx + 1) == len(dataloader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += float(loss.detach().item())
                valid_batches += 1
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{outputs.ce_loss.item():.4f}",
                    "rank": f"{outputs.ranking_loss.item():.4f}",
                    "text_aux": f"{outputs.text_aux_loss.item():.4f}",
                    "gcn_aux": f"{outputs.gcn_aux_loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

            average_loss = total_loss / max(1, valid_batches)
            logger.info("V2 Epoch %s completed. Average loss: %.4f", epoch + 1, average_loss)
            self._append_metrics_event(
                "train_epoch_end",
                {"epoch": int(epoch + 1), "avg_loss": float(average_loss)},
            )

            if eval_dataloader is not None:
                metrics = self.evaluate(eval_dataloader)
                self._append_metrics_event(
                    "eval_after_train_epoch",
                    {
                        "epoch": int(epoch + 1),
                        "top1_acc": float(metrics["top1_acc"]),
                        "top3_acc": float(metrics["top3_acc"]),
                        "top5_acc": float(metrics["top5_acc"]),
                    },
                )
                if metrics["top1_acc"] > self.best_top1 + early_stopping_min_delta:
                    self.best_top1 = metrics["top1_acc"]
                    self.best_epoch = epoch + 1
                    self.save_checkpoint()
                    patience_counter = 0
                    logger.info(
                        "V2 new best top-1: %.4f at epoch %s, checkpoint saved",
                        self.best_top1,
                        epoch + 1,
                    )
                else:
                    patience_counter += 1
                    logger.info(
                        "V2 no top-1 improvement at epoch %s (%s/%s, best=%.4f)",
                        epoch + 1,
                        patience_counter,
                        early_stopping_patience,
                        self.best_top1,
                    )
                    if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                        logger.info(
                            "V2 early stopping at epoch %s. Best epoch=%s best top-1=%.4f",
                            epoch + 1,
                            self.best_epoch,
                            self.best_top1,
                        )
                        break
            else:
                if average_loss < self.best_loss - early_stopping_min_delta:
                    self.best_loss = average_loss
                    self.best_epoch = epoch + 1
                    self.save_checkpoint()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                        logger.info(
                            "V2 early stopping at epoch %s. Best epoch=%s best loss=%.4f",
                            epoch + 1,
                            self.best_epoch,
                            self.best_loss,
                        )
                        break

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total = 0
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        for batch in tqdm(dataloader, desc="V2 Evaluating"):
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
            top1 = logits.argmax(dim=-1)
            top3 = logits.topk(k=min(3, logits.size(-1)), dim=-1).indices
            top5 = logits.topk(k=min(5, logits.size(-1)), dim=-1).indices
            correct_top1 += int((top1 == labels).sum().item())
            correct_top3 += int((top3 == labels.unsqueeze(1)).any(dim=-1).sum().item())
            correct_top5 += int((top5 == labels.unsqueeze(1)).any(dim=-1).sum().item())
            total += int(labels.size(0))
        metrics = {
            "top1_acc": correct_top1 / max(1, total),
            "top3_acc": correct_top3 / max(1, total),
            "top5_acc": correct_top5 / max(1, total),
        }
        logger.info(
            "V2 evaluation completed. Top-1: %.4f, Top-3: %.4f, Top-5: %.4f",
            metrics["top1_acc"],
            metrics["top3_acc"],
            metrics["top5_acc"],
        )
        return metrics
