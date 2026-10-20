from types import SimpleNamespace

import modelscope
import torch
import torch.nn as nn


class DummyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=32)
        self.embedding = nn.Embedding(16, 32)

    def get_input_embeddings(self):
        return self.embedding

    def gradient_checkpointing_enable(self, **kwargs):
        return None


class DummyAutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return DummyLLM()


if not hasattr(modelscope, 'AutoModelForCausalLM'):
    modelscope.AutoModelForCausalLM = DummyAutoModelForCausalLM
if not hasattr(modelscope, 'AutoTokenizer'):
    modelscope.AutoTokenizer = object

import components.llm_graph_ranker as ranker_module


def build_ranker(mode):
    return ranker_module.LLMGraphRanker(
        model_name='dummy',
        gcn_dim=8,
        num_history=3,
        num_candidates=4,
        match_feature_dim=5,
        interaction_feature_dim=2,
        fusion_weight_mode=mode,
        eta1_init=0.5,
        eta2_init=0.5,
        match_weight_init=0.3,
    )


def main():
    original = ranker_module.AutoModelForCausalLM.from_pretrained
    ranker_module.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: DummyLLM()
    try:
        learnable = build_ranker('learnable')
        values = learnable.fusion_weight_values()
        assert abs(values['mean_history'] - 0.5) < 1e-5, values
        assert abs(values['retrieval'] - 0.5) < 1e-5, values
        assert abs(values['match'] - 0.3) < 1e-5, values
        assert learnable.mean_history_weight.requires_grad
        assert learnable.retrieval_weight.requires_grad
        assert learnable.match_weight.requires_grad

        fixed = build_ranker('fixed')
        fixed_values = fixed.fusion_weight_values()
        assert abs(fixed_values['mean_history'] - 0.5) < 1e-5, fixed_values
        assert abs(fixed_values['retrieval'] - 0.5) < 1e-5, fixed_values
        assert abs(fixed_values['match'] - 0.3) < 1e-5, fixed_values
        assert not fixed.mean_history_weight.requires_grad
        assert not fixed.retrieval_weight.requires_grad
        assert not fixed.match_weight.requires_grad
    finally:
        ranker_module.AutoModelForCausalLM.from_pretrained = original
    print('fusion weight mode test passed')


if __name__ == '__main__':
    main()
