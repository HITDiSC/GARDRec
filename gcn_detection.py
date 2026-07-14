# Copied from workspace gcn_detection.py
from neo4j_connect import Neo4jConnection as RawNeo4jConnection
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "Wangyong123",
    "database": "kg-az-book"
}

import torch
import torch.nn as nn
from modelscope import AutoModelForCausalLM, AutoTokenizer

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim, eps=1e-4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self._init_weights()
    def _init_weights(self):
        linear_layers = [m for m in self.projector if isinstance(m, nn.Linear)]
        nn.init.normal_(linear_layers[0].weight, std=0.01)
        if linear_layers[0].bias is not None:
            nn.init.zeros_(linear_layers[0].bias)
        nn.init.normal_(linear_layers[-1].weight, std=1e-5)
        if linear_layers[-1].bias is not None:
            nn.init.zeros_(linear_layers[-1].bias)
    def forward(self, x):
        return self.projector(x)
