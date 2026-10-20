import argparse


def build_parser():
    parser = argparse.ArgumentParser("K-RagRec-Qwen2 baseline (server KG/GCN backend)")
    parser.add_argument("--seed", type=int, default=9)

    parser.add_argument("--train_file", type=str, default="data/dataset/ML1M/kragrec_v2_train.json")
    parser.add_argument("--valid_file", type=str, default="data/dataset/ML1M/kragrec_v2_valid.json")
    parser.add_argument("--eval_file", type=str, default="data/dataset/ML1M/kragrec_v2_easy_test.json")
    parser.add_argument("--index_dir", type=str, default="data/dataset/kg_ml1m_kragrec")
    parser.add_argument("--output_dir", type=str, default="output/kragrec_qwen2_ml1m")
    parser.add_argument("--run_name", type=str, default="kragrec_qwen2_ml1m")
    parser.add_argument("--checkpoint", type=str, default="")

    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_end", type=int, default=-1)
    parser.add_argument("--eval_start", type=int, default=0)
    parser.add_argument("--eval_end", type=int, default=-1)
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_eval", type=int, default=0)

    parser.add_argument("--llm_model_name", type=str, default="qwen2_7b_instruct")
    parser.add_argument("--llm_model_path", type=str, default="/data/WY/models/Qwen2-7B-Instruct")
    parser.add_argument("--llm_frozen", type=str, default="True")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_txt_len", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=1)

    parser.add_argument("--gnn_model_name", type=str, default="gt")
    parser.add_argument("--gnn_num_layers", type=int, default=2)
    parser.add_argument("--gnn_in_dim", type=int, default=256)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout", type=float, default=0.0)

    parser.add_argument("--adaptive_ratio", type=int, default=5)
    parser.add_argument("--sub_graph_numbers", type=int, default=3)
    parser.add_argument("--reranking_numbers", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--epoch_offset", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=2)
    parser.add_argument("--log_every", type=int, default=10)
    return parser


def parse_args():
    return build_parser().parse_args()