# Exported subset of original gcn_recommend.py for repository export.
# Note: This file assumes sibling modules are present in same folder.

import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
from torch.utils.data import DataLoader

from components.config import setup_file_logging, build_model_tag, logger
from components.data import (
	resolve_input_file,
	resolve_retrieval_details_file,
	load_input_entries,
	load_retrieval_details,
	attach_retrieval_details,
	infer_interaction_feature_dim,
)
from components.model import prepare_tokenizer, GCNRecommendationDataset
from components.neo4j_utils import get_item_gcn_embedding
from components.llm_graph_ranker import LLMGraphRanker
from components.trainer import GCNRecommendationTrainer
from components.utils import (
	compute_answer_prompt_ids,
	compute_label_token_ids,
)
from kg_retrieval_enhancement import get_domain_config


def main():
	parser = argparse.ArgumentParser(description="GCN Recommendation Training")
	parser.add_argument("--model", type=str, default="shakechen/Llama-2-7b-chat-hf")
	parser.add_argument("--domain", type=str, default="amazonbook", choices=["amazonbook", "ml20m", "ml1m"])
	args = parser.parse_args()

	domain = args.domain
	domain_config = get_domain_config(domain)
	match_feature_dim = len(domain_config["MATCH_FEATURE_NAMES"])
	num_history = 10
	num_candidates = 20
	prompt_mode = "minimal"
	model_name = args.model
	model_tag = build_model_tag(model_name)
	run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
	logging_paths = setup_file_logging(
		log_dir=f"./output/{domain}/rank_save/logs",
		run_tag=run_tag,
		model_tag=model_tag,
	)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	input_file = resolve_input_file(domain)
	retrieval_details_file = resolve_retrieval_details_file(domain)
	model_save_path = f"./output/{domain}/rank_save/{model_tag}_best_gcn_ranker_head.pth"
	os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
	logger.info("Using retrieval details file: %s", retrieval_details_file)
	if not os.path.exists(retrieval_details_file):
		raise FileNotFoundError(
			"Cannot find offline retrieval detail file. Please run preprocess_retrieval_augmented_samples.py first."
		)
	gcn_embedding_dict, embedding_dim = get_item_gcn_embedding(domain=domain, limit=None)
	entries = load_input_entries(input_file)
	samples = []
	skipped_empty_history = skipped_short_candidates = skipped_missing_target = 0
	for entry_index, entry in enumerate(entries):
		raw_history = entry.get("history_movies", entry.get("history_books", []))
		raw_candidates = entry.get("candidate_movies", entry.get("candidate_books", []))
		raw_target = entry.get("target_movie", entry.get("target_book", {}))
		target_id = str(raw_target.get("original_id")) if raw_target.get("original_id") is not None else None
		def _get_valid_movie(item_dict, gcn_embedding_dict, target_id=None):
			item_id = str(item_dict.get("original_id"))
			if item_id not in gcn_embedding_dict:
				valid_ids = list(gcn_embedding_dict.keys())
				if target_id is not None and target_id in valid_ids:
					valid_ids.remove(target_id)
				if not valid_ids:
					return {"original_id": item_id, "title": item_dict.get("title", "Unknown")}
				item_id = str(np.random.choice(valid_ids))
			return {"original_id": item_id, "title": item_dict.get("title", "Unknown")}
		filtered_history = [_get_valid_movie(m, gcn_embedding_dict, target_id=target_id) for m in raw_history]
		filtered_candidates = [_get_valid_movie(m, gcn_embedding_dict, target_id=target_id) for m in raw_candidates]
		if not filtered_history:
			skipped_empty_history += 1
			continue
		if len(filtered_history) > num_history:
			filtered_history = filtered_history[-num_history:]
		if len(filtered_candidates) < num_candidates:
			skipped_short_candidates += 1
			continue
		filtered_candidates = filtered_candidates[:num_candidates]
		candidate_ids = [str(m.get("original_id")) for m in filtered_candidates]
		if target_id is None or candidate_ids.count(target_id) != 1:
			skipped_missing_target += 1
			continue
		target_idx = candidate_ids.index(target_id)
		samples.append({
			"sample_idx": entry.get("sample_idx", entry_index),
			"history_movies": filtered_history,
			"candidate_movies": filtered_candidates,
			"target_idx": target_idx,
		})
	logger.info(
		"Built %s ranking samples. Skipped: empty_history=%s, short_candidates=%s, missing_target=%s",
		len(samples),
		skipped_empty_history,
		skipped_short_candidates,
		skipped_missing_target,
	)
	retrieval_detail_map, interaction_feature_names = load_retrieval_details(retrieval_details_file)
	samples = attach_retrieval_details(
		samples,
		retrieval_detail_map,
		interaction_feature_names=interaction_feature_names,
		interaction_feature_whitelist=[
			"gsim", "lmax", "first_norm", "user_avg_feat", "user_cnt_feat", "item_avg_feat", "item_cnt_feat", "co_feat", "genre_feat", "rank_feat",
		],
	)
	interaction_feature_dim = infer_interaction_feature_dim(samples, num_candidates=num_candidates)
	logger.info("Using %s ranking samples with offline retrieval details", len(samples))
	logger.info("Interaction feature dim: %s", interaction_feature_dim)
	tokenizer = prepare_tokenizer(model_name)
	answer_prompt_ids = compute_answer_prompt_ids(tokenizer)
	label_token_ids = compute_label_token_ids(tokenizer, num_candidates=num_candidates)
	train_size = int(0.9 * len(samples))
	train_dataset = GCNRecommendationDataset(
		samples[:train_size],
		tokenizer,
		gcn_embedding_dict,
		embedding_dim,
		match_feature_dim=match_feature_dim,
		max_prompt_length=256,
		num_history=num_history,
		num_candidates=num_candidates,
		interaction_feature_dim=interaction_feature_dim,
		prompt_mode=prompt_mode,
	)
	eval_dataset = GCNRecommendationDataset(
		samples[train_size:],
		tokenizer,
		gcn_embedding_dict,
		embedding_dim,
		match_feature_dim=match_feature_dim,
		max_prompt_length=256,
		num_history=num_history,
		num_candidates=num_candidates,
		interaction_feature_dim=interaction_feature_dim,
		prompt_mode=prompt_mode,
	)
	train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
	eval_dataloader = DataLoader(eval_dataset, batch_size=16)
	ranker_params = {
		"model_name": model_name,
		"gcn_dim": embedding_dim,
		"num_history": num_history,
		"num_candidates": num_candidates,
		"match_feature_dim": match_feature_dim,
		"interaction_feature_dim": interaction_feature_dim,
		"ce_loss_weight": 1.0,
		"ranking_loss_weight": 0.5,
		"aux_loss_weight": 1.0,
		"ranking_margin": 0.3,
		"hardest_negative_k": 0,
		"gen_loss_weight": 2.0,
		"gen_inference_weight": 1.0,
		"gen_loss_warmup_epochs": 0,
	}
	model = LLMGraphRanker(
		**ranker_params,
		label_token_ids=label_token_ids,
		answer_prompt_ids=answer_prompt_ids,
	)
	trainer = GCNRecommendationTrainer(
		model,
		tokenizer,
		device=device,
		model_save_path=model_save_path,
		metrics_log_path=logging_paths["metrics_log_path"],
	)
	trainer.train(train_dataloader, epochs=5, lr=1e-4, eval_dataloader=eval_dataloader)
	if os.path.exists(model_save_path):
		trainer.load_probehead(model_save_path)
	metrics = trainer.evaluate(eval_dataloader)
	logger.info("Run metrics: %s", metrics)
	results_save_path = f"./output/{domain}/rank_save/eval_metrics_{run_tag}.json"
	with open(results_save_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=4)
	logger.info("Evaluation metrics saved to %s", results_save_path)


if __name__ == "__main__":
	main()
