import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
from torch.utils.data import DataLoader

from components.config import build_model_tag, logger, setup_file_logging
from components.data import (
    attach_retrieval_details,
    infer_interaction_feature_dim,
    load_input_entries,
    load_retrieval_details,
    resolve_input_file,
    resolve_retrieval_details_file,
)
from components.kg_retrieval_enhancement import get_domain_config
from components.llm_graph_ranker_v2 import LLMGraphRankerV2
from components.model import prepare_tokenizer
from components.model_v2 import GCNRecommendationDatasetV2
from components.neo4j_utils import get_item_gcn_embedding
from components.trainer_v2 import GCNRecommendationTrainerV2


V2_INTERACTION_FEATURE_WHITELIST = [
    "gsim",
    "lmax",
    "user_avg_feat",
    "user_cnt_feat",
    "item_avg_feat",
    "item_cnt_feat",
    "co_feat",
    "rank_feat",
]


def build_samples(entries, gcn_embedding_dict, num_history, num_candidates):
    samples = []
    skipped_empty_history = 0
    skipped_short_candidates = 0
    skipped_missing_target = 0

    for entry_index, entry in enumerate(entries):
        raw_history = entry.get("history_movies", entry.get("history_books", []))
        raw_candidates = entry.get("candidate_movies", entry.get("candidate_books", []))
        raw_target = entry.get("target_movie", entry.get("target_book", {}))
        target_id = (
            str(raw_target.get("original_id"))
            if raw_target.get("original_id") is not None
            else None
        )

        def valid_movie(item_dict):
            item_id = str(item_dict.get("original_id"))
            if item_id not in gcn_embedding_dict:
                valid_ids = list(gcn_embedding_dict.keys())
                if target_id is not None and target_id in valid_ids:
                    valid_ids.remove(target_id)
                if not valid_ids:
                    return {
                        "original_id": item_id,
                        "title": item_dict.get("title", "Unknown"),
                    }
                item_id = str(np.random.choice(valid_ids))
            return {
                "original_id": item_id,
                "title": item_dict.get("title", "Unknown"),
            }

        filtered_history = [valid_movie(item) for item in raw_history]
        filtered_candidates = [valid_movie(item) for item in raw_candidates]
        if not filtered_history:
            skipped_empty_history += 1
            continue
        if len(filtered_history) > num_history:
            filtered_history = filtered_history[-num_history:]
        if len(filtered_candidates) < num_candidates:
            skipped_short_candidates += 1
            continue

        filtered_candidates = filtered_candidates[:num_candidates]
        candidate_ids = [str(item.get("original_id")) for item in filtered_candidates]
        if target_id is None or candidate_ids.count(target_id) != 1:
            skipped_missing_target += 1
            continue
        samples.append(
            {
                "sample_idx": entry.get("sample_idx", entry_index),
                "history_movies": filtered_history,
                "candidate_movies": filtered_candidates,
                "target_idx": candidate_ids.index(target_id),
            }
        )

    logger.info(
        "Built %s ranking samples. Skipped: empty_history=%s short_candidates=%s "
        "missing_target=%s",
        len(samples),
        skipped_empty_history,
        skipped_short_candidates,
        skipped_missing_target,
    )
    return samples


def main():
    parser = argparse.ArgumentParser(description="Text-aware GCN Recommendation V2")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--domain", type=str, default="ml1m", choices=["amazonbook", "ml20m", "ml1m"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--retrieval-details", type=str, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--prompt_mode", type=str, default="with_titles",
                        choices=["with_titles", "text_only", "minimal", "structure_only"])
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ranking_loss_weight", type=float, default=0.5)
    parser.add_argument("--hardest_negative_k", type=int, default=4)
    parser.add_argument("--text_aux_loss_weight", type=float, default=0.10)
    parser.add_argument("--gcn_aux_loss_weight", type=float, default=0.10)
    parser.add_argument("--feature_aux_loss_weight", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--limit_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--resume_best_top1", type=float, default=0.0)
    parser.add_argument("--model_save_path", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    domain = args.domain
    domain_config = get_domain_config(domain)
    match_feature_dim = len(domain_config["MATCH_FEATURE_NAMES"])
    num_history = 10
    num_candidates = 20
    model_name = args.model
    model_tag = build_model_tag(model_name)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_paths = setup_file_logging(
        log_dir=f"./output/{domain}/rank_save/logs",
        run_tag=run_tag,
        model_tag=model_tag,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_file = args.input_file or resolve_input_file(domain)
    retrieval_details_file = args.retrieval_details or resolve_retrieval_details_file(domain)
    model_save_path = args.model_save_path or (
        f"./output/{domain}/rank_save/{model_tag}_best_gcn_ranker_v2.pth"
    )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    logger.info("V2 model save path: %s", model_save_path)
    logger.info("V2 using retrieval details file: %s", retrieval_details_file)
    logger.info("V2 prompt mode: %s, max_prompt_length=%s", args.prompt_mode, args.max_prompt_length)
    if not os.path.exists(retrieval_details_file):
        raise FileNotFoundError(f"Cannot find retrieval detail file: {retrieval_details_file}")

    gcn_embedding_dict, embedding_dim = get_item_gcn_embedding(domain=domain, limit=None)
    entries = load_input_entries(input_file)
    samples = build_samples(entries, gcn_embedding_dict, num_history, num_candidates)
    retrieval_detail_map, interaction_feature_names = load_retrieval_details(
        retrieval_details_file
    )
    samples = attach_retrieval_details(
        samples,
        retrieval_detail_map,
        interaction_feature_names=interaction_feature_names,
        interaction_feature_whitelist=V2_INTERACTION_FEATURE_WHITELIST,
    )
    if not samples:
        raise RuntimeError("No ranking samples were built for v2")

    first_match_features = np.asarray(samples[0].get("candidate_match_features"))
    if first_match_features.ndim >= 2 and first_match_features.shape[-1] > 0:
        loaded_match_feature_dim = int(first_match_features.shape[-1])
        if loaded_match_feature_dim != match_feature_dim:
            logger.warning(
                "Match feature dimension mismatch: config=%s data=%s; using data dimension",
                match_feature_dim,
                loaded_match_feature_dim,
            )
            match_feature_dim = loaded_match_feature_dim
    retrieval_dims = {
        len(sample["history_retrieval_embedding"])
        for sample in samples
        if sample.get("history_retrieval_embedding") is not None
    }
    if retrieval_dims and retrieval_dims != {embedding_dim}:
        raise ValueError(
            f"Retrieval dimension mismatch: details={sorted(retrieval_dims)} "
            f"gcn_embeddings={embedding_dim}"
        )

    interaction_feature_dim = infer_interaction_feature_dim(samples, num_candidates)
    logger.info("V2 using %s ranking samples", len(samples))
    logger.info("V2 GCN embedding dim: %s", embedding_dim)
    logger.info("V2 interaction feature dim: %s", interaction_feature_dim)
    logger.info("V2 selected interaction features: %s", V2_INTERACTION_FEATURE_WHITELIST)
    if args.limit_samples and args.limit_samples > 0:
        samples = samples[: args.limit_samples]
        logger.info("V2 debug limit_samples=%s", len(samples))

    tokenizer = prepare_tokenizer(model_name)
    train_size = int(0.9 * len(samples))
    train_size = max(1, min(train_size, len(samples) - 1)) if len(samples) > 1 else 1
    train_dataset = GCNRecommendationDatasetV2(
        samples[:train_size],
        tokenizer,
        gcn_embedding_dict,
        embedding_dim,
        match_feature_dim=match_feature_dim,
        max_prompt_length=args.max_prompt_length,
        num_history=num_history,
        num_candidates=num_candidates,
        interaction_feature_dim=interaction_feature_dim,
        prompt_mode=args.prompt_mode,
    )
    eval_dataset = GCNRecommendationDatasetV2(
        samples[train_size:],
        tokenizer,
        gcn_embedding_dict,
        embedding_dim,
        match_feature_dim=match_feature_dim,
        max_prompt_length=args.max_prompt_length,
        num_history=num_history,
        num_candidates=num_candidates,
        interaction_feature_dim=interaction_feature_dim,
        prompt_mode=args.prompt_mode,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = LLMGraphRankerV2(
        model_name=model_name,
        gcn_dim=embedding_dim,
        num_history=num_history,
        num_candidates=num_candidates,
        match_feature_dim=match_feature_dim,
        interaction_feature_dim=interaction_feature_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        ranking_loss_weight=args.ranking_loss_weight,
        hardest_negative_k=args.hardest_negative_k,
        text_aux_loss_weight=args.text_aux_loss_weight,
        gcn_aux_loss_weight=args.gcn_aux_loss_weight,
        feature_aux_loss_weight=args.feature_aux_loss_weight,
        label_smoothing=args.label_smoothing,
    )
    trainer = GCNRecommendationTrainerV2(
        model,
        tokenizer,
        device=device,
        model_save_path=model_save_path,
        metrics_log_path=logging_paths["metrics_log_path"],
    )

    if args.resume_checkpoint:
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        if not trainer.load_checkpoint(args.resume_checkpoint):
            raise RuntimeError(f"Failed to load resume checkpoint: {args.resume_checkpoint}")
        trainer.best_top1 = args.resume_best_top1
        logger.info(
            "V2 resumed from %s (baseline Top-1=%.4f)",
            args.resume_checkpoint,
            trainer.best_top1,
        )

    if args.eval_only:
        if not args.resume_checkpoint and not os.path.exists(model_save_path):
            raise ValueError("eval_only requires --resume_checkpoint or an existing model_save_path")
        if not args.resume_checkpoint:
            trainer.load_checkpoint(model_save_path)
        metrics = trainer.evaluate(eval_dataloader)
        metrics["model_name"] = model_name
        results_save_path = f"./output/{domain}/rank_save/eval_metrics_v2_{run_tag}.json"
        with open(results_save_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=4)
        logger.info("V2 eval-only metrics saved to %s", results_save_path)
        return

    trainer.train(
        train_dataloader,
        epochs=args.epochs,
        lr=args.lr,
        eval_dataloader=eval_dataloader,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        grad_accum_steps=args.grad_accum_steps,
    )

    if os.path.exists(model_save_path):
        trainer.load_checkpoint(model_save_path)
    elif args.resume_checkpoint:
        trainer.load_checkpoint(args.resume_checkpoint)
    metrics = trainer.evaluate(eval_dataloader)
    metrics["model_name"] = model_name
    results_save_path = f"./output/{domain}/rank_save/eval_metrics_v2_{run_tag}.json"
    with open(results_save_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=4)
    logger.info("V2 run metrics: %s", metrics)
    logger.info("V2 evaluation metrics saved to %s", results_save_path)


if __name__ == "__main__":
    main()
