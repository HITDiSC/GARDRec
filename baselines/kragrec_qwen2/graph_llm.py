import contextlib

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gnn import load_gnn_model

IGNORE_INDEX = -100
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def scatter_mean(src, index, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    out = src.new_zeros((dim_size, src.size(1)))
    count = src.new_zeros((dim_size,))
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones(src.size(0), device=src.device, dtype=src.dtype))
    return out / count.clamp_min(1.0).unsqueeze(1)


class GraphLLM(nn.Module):
    """K-RagRec graph-prefix model adapted to Qwen2.

    Only the graph encoder and the projector are trainable.  The LLM is frozen
    and receives the projected graph embedding as a soft prefix token.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_txt_len = int(args.max_txt_len)
        self.use_chat_template = bool(getattr(args, "use_chat_template", True))
        self.dtype = torch.bfloat16 if getattr(args, "dtype", "bf16") == "bf16" else torch.float16
        self._device = torch.device(getattr(args, "device", "cuda"))

        print("Loading Qwen2 from", args.llm_model_path, flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        load_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        attn_implementation = getattr(args, "attn_implementation", "sdpa")
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        try:
            model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, **load_kwargs)
        except TypeError:
            load_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, **load_kwargs)
        model.to(self._device)
        model.config.use_cache = False

        for param in model.parameters():
            param.requires_grad = False
        if args.llm_frozen != "True":
            raise NotImplementedError("This baseline only supports a frozen Qwen2 LLM")
        if getattr(args, "gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        self.model = model
        self.hidden_size = int(model.config.hidden_size)
        print("Qwen2 hidden_size =", self.hidden_size, flush=True)

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self._device)

        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.hidden_size),
        ).to(self._device)

        self.word_embedding = self.model.get_input_embeddings()
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = self.tokenizer.pad_token_id
        self.letter_token_ids = self._resolve_letter_token_ids()
        print("letter token ids:", dict(zip(LETTERS[:20], self.letter_token_ids)), flush=True)

    def _resolve_letter_token_ids(self):
        ids = []
        for letter in LETTERS[:20]:
            chosen = None
            for text in (" " + letter, letter):
                token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
                if len(token_ids) == 1:
                    chosen = int(token_ids[0])
                    break
            if chosen is None:
                raise ValueError("Option letter %r is not a single token" % letter)
            ids.append(chosen)
        return ids

    @property
    def device(self):
        return self._device

    def maybe_autocast(self):
        if self._device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return contextlib.nullcontext()

    def train(self, mode=True):
        super().train(mode)
        # The LLM stays in eval mode: only the graph encoder and projector train.
        self.model.eval()
        self.graph_encoder.train(mode)
        self.projector.train(mode)
        return self

    def _encode_question(self, question):
        if self.use_chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = question
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def encode_graphs(self, samples):
        graph_embeds = []
        for graphs in samples["graph"]:
            if not graphs:
                graph_embeds.append(
                    torch.zeros((1, self.args.gnn_hidden_dim), device=self._device, dtype=torch.float32)
                )
                continue
            batch = Batch.from_data_list(graphs).to(self._device)
            node_embeds, _ = self.graph_encoder(
                batch.x.float(),
                batch.edge_index.long(),
                batch.edge_attr.float(),
            )
            graph_embeds.append(scatter_mean(node_embeds, batch.batch, dim_size=batch.num_graphs))
        return graph_embeds

    def _build_embeds(self, samples, with_labels):
        projected = [self.projector(embed) for embed in self.encode_graphs(samples)]
        embed_dtype = self.word_embedding.weight.dtype
        prefixes = [proj.mean(dim=0, keepdim=True).to(embed_dtype) for proj in projected]

        batch_inputs = []
        batch_masks = []
        batch_labels = []
        pad_id = self.tokenizer.pad_token_id

        for index, question in enumerate(samples["question"]):
            question_ids = self._encode_question(question)
            if with_labels:
                label_ids = self.tokenizer(
                    " " + str(samples["label"][index]), add_special_tokens=False
                ).input_ids + [self.eos_token_id]
            else:
                label_ids = []

            keep = self.max_txt_len - 1 - len(label_ids)
            keep = max(1, keep)
            question_ids = question_ids[-keep:]

            full_ids = question_ids + label_ids
            token_embeds = self.word_embedding(
                torch.tensor(full_ids, dtype=torch.long, device=self._device)
            )
            inputs_embeds = torch.cat([prefixes[index], token_embeds], dim=0)
            batch_inputs.append(inputs_embeds)
            batch_masks.append([1] * inputs_embeds.shape[0])
            if with_labels:
                batch_labels.append(
                    [IGNORE_INDEX] * (1 + len(question_ids)) + label_ids
                )

        max_length = max(x.shape[0] for x in batch_inputs)
        pad_embed = self.word_embedding(
            torch.tensor([pad_id], dtype=torch.long, device=self._device)
        )
        for index in range(len(batch_inputs)):
            pad_length = max_length - batch_inputs[index].shape[0]
            if pad_length > 0:
                batch_inputs[index] = torch.cat(
                    [pad_embed.repeat(pad_length, 1), batch_inputs[index]], dim=0
                )
                batch_masks[index] = [0] * pad_length + batch_masks[index]
                if with_labels:
                    batch_labels[index] = [IGNORE_INDEX] * pad_length + batch_labels[index]

        inputs_embeds = torch.stack(batch_inputs, dim=0)
        attention_mask = torch.tensor(batch_masks, dtype=torch.long, device=self._device)
        labels = None
        if with_labels:
            labels = torch.tensor(batch_labels, dtype=torch.long, device=self._device)
        return inputs_embeds, attention_mask, labels

    def forward(self, samples):
        inputs_embeds, attention_mask, labels = self._build_embeds(samples, with_labels=True)
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                use_cache=False,
            )
        return outputs.loss

    @torch.no_grad()
    def score_options(self, samples):
        inputs_embeds, attention_mask, _ = self._build_embeds(samples, with_labels=False)
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )
        logits = outputs.logits[:, -1, :].float()
        option_ids = torch.tensor(self.letter_token_ids, device=logits.device)
        scores = logits.index_select(1, option_ids).softmax(dim=-1)
        return scores.cpu()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self):
        state = {}
        for name, tensor in self.graph_encoder.state_dict().items():
            state["graph_encoder." + name] = tensor.detach().cpu()
        for name, tensor in self.projector.state_dict().items():
            state["projector." + name] = tensor.detach().cpu()
        return state

    def load_trainable_state_dict(self, state):
        graph_state = {
            key[len("graph_encoder."):]: value
            for key, value in state.items()
            if key.startswith("graph_encoder.")
        }
        projector_state = {
            key[len("projector."):]: value
            for key, value in state.items()
            if key.startswith("projector.")
        }
        graph_result = self.graph_encoder.load_state_dict(graph_state, strict=True)
        projector_result = self.projector.load_state_dict(projector_state, strict=True)
        return graph_result, projector_result

    def print_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            "trainable params: {:,} / {:,} ({:.2f}%)".format(
                trainable, total, 100.0 * trainable / max(total, 1)
            ),
            flush=True,
        )
        return trainable, total