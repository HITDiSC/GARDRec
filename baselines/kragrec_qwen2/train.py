import json
import math
import os
import random
import shutil
import sys
import time

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import parse_args
from data_utils import build_prompt, letter_to_index, load_json
from graph_llm import GraphLLM
from retrieve import GraphRetrieval


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(path, payload):
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temp, path)


def slice_rows(rows, start, end, limit):
    if start:
        rows = rows[start:]
    if end is not None and end >= 0:
        rows = rows[:end]
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


def row_sequence(row):
    sequence = row.get("sequence_ids", [])
    if isinstance(sequence, str):
        sequence = json.loads(sequence)
    return [str(item) for item in sequence]


def build_batch(rows, retrieval, args):
    ids = []
    graphs = []
    questions = []
    labels = []
    for index, row in enumerate(rows):
        sequence = row_sequence(row)
        retrieve_count = max(1, int(args.adaptive_ratio * max(1, len(sequence))))
        retrieved = retrieval.whether_retrieval(sequence, retrieve_count)
        graphs.append(
            retrieval.retrieval_topk(
                row.get("input", ""),
                retrieved,
                topk_nodes=args.sub_graph_numbers,
                topk_rerank_nodes=args.reranking_numbers,
            )
        )
        ids.append(str(row.get("sample_idx", index)))
        questions.append(build_prompt(row))
        labels.append(str(row.get("output", "A")))
    return {"id": ids, "graph": graphs, "question": questions, "label": labels}


def save_checkpoint(model, optimizer, epoch, args, path, loss=None):
    payload = {
        "model": model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "args": vars(args),
        "format": "kragrec_qwen2_trainable_only",
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    progress_path = os.path.join(args.output_dir, "progress.json")
    log_path = os.path.join(args.output_dir, "train_log.jsonl")

    rows = load_json(args.train_file)
    rows = slice_rows(rows, args.train_start, args.train_end, args.limit_train)
    if not rows:
        raise RuntimeError("No training rows selected")

    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    print("Loading retrieval index from", args.index_dir, flush=True)
    retrieval = GraphRetrieval(args.index_dir)
    print("Loading graph LLM", flush=True)
    model = GraphLLM(args)
    model.print_trainable_params()
    resume_loss = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(checkpoint.get("model", checkpoint))
        resume_loss = checkpoint.get("loss")
        print("resumed trainable weights from", args.checkpoint, flush=True)

    trainable = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd
    )

    steps_per_epoch = math.ceil(len(rows) / max(1, args.batch_size))
    global_step = 0
    started = time.time()
    best_loss = float(resume_loss) if resume_loss is not None else float("inf")
    total_epochs = args.num_epochs + args.epoch_offset

    train_log = open(log_path, "a", encoding="utf-8")
    try:
        for epoch in range(args.num_epochs):
            display_epoch = epoch + 1 + args.epoch_offset
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            running_count = 0
            accumulator = 0
            epoch_start = time.time()

            for step, start in enumerate(
                tqdm(range(0, len(rows), args.batch_size), desc="epoch %d/%d" % (display_epoch, total_epochs))
            ):
                batch_rows = rows[start:start + args.batch_size]
                batch = build_batch(batch_rows, retrieval, args)
                loss = model(batch)
                (loss / max(1, args.grad_accum_steps)).backward()
                accumulator += 1

                loss_value = float(loss.detach().cpu())
                running_loss += loss_value
                running_count += 1

                should_step = (
                    accumulator >= args.grad_accum_steps
                    or step + 1 == steps_per_epoch
                )
                if should_step:
                    clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulator = 0
                    global_step += 1

                if step % max(1, args.log_every) == 0 or step + 1 == steps_per_epoch:
                    elapsed = time.time() - started
                    average_loss = running_loss / max(1, running_count)
                    print(
                        "epoch {}/{} step {}/{} global_step {} loss {:.6f} avg_loss {:.6f}".format(
                            display_epoch,
                            total_epochs,
                            step + 1,
                            steps_per_epoch,
                            global_step,
                            loss_value,
                            average_loss,
                        ),
                        flush=True,
                    )
                    progress = {
                        "status": "running",
                        "run_name": args.run_name,
                        "epoch": display_epoch,
                        "num_epochs": total_epochs,
                        "step": step + 1,
                        "steps_per_epoch": steps_per_epoch,
                        "global_step": global_step,
                        "loss": loss_value,
                        "avg_loss": average_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_sec": round(elapsed, 2),
                        "samples_per_sec": round(running_count / max(elapsed, 1e-6), 4),
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    if torch.cuda.is_available():
                        progress["gpu_memory_allocated_gb"] = round(
                            torch.cuda.memory_allocated() / 1024 ** 3, 2
                        )
                    atomic_json(progress_path, progress)
                    train_log.write(json.dumps(progress, ensure_ascii=False) + "\n")
                    train_log.flush()

            epoch_loss = running_loss / max(1, running_count)
            epoch_payload = {
                "epoch": display_epoch,
                "num_epochs": total_epochs,
                "loss": epoch_loss,
                "elapsed_sec": round(time.time() - epoch_start, 2),
            }
            print("epoch finished:", json.dumps(epoch_payload, ensure_ascii=False), flush=True)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(
                    model, optimizer, display_epoch, args,
                    os.path.join(args.output_dir, "checkpoint_best.pth"),
                    loss=epoch_loss,
                )
            save_checkpoint(
                model, optimizer, display_epoch, args,
                os.path.join(args.output_dir, "checkpoint_last.pth"),
                loss=epoch_loss,
            )
            save_checkpoint(
                model, optimizer, display_epoch, args,
                os.path.join(args.output_dir, "checkpoint_epoch%d.pth" % display_epoch),
                loss=epoch_loss,
            )

        if not os.path.exists(os.path.join(args.output_dir, "checkpoint_best.pth")):
            shutil.copyfile(
                os.path.join(args.output_dir, "checkpoint_last.pth"),
                os.path.join(args.output_dir, "checkpoint_best.pth"),
            )
        atomic_json(
            progress_path,
            {
                "status": "completed",
                "run_name": args.run_name,
                "epoch": total_epochs,
                "num_epochs": total_epochs,
                "best_loss": best_loss,
                "elapsed_sec": round(time.time() - started, 2),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        print("training completed", flush=True)
    finally:
        train_log.close()


if __name__ == "__main__":
    main()