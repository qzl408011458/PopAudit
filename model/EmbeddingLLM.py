import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List

class QwenPersonaEmbeddingB(nn.Module):
    """
    Qwen3 hidden-state persona embedding
    Scheme B: average of last L layers + masked mean pooling
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda",
        max_length: int = 256,
        last_n_layers: int = 4,
        cond_dim: int = 256,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.device = device
        self.max_length = max_length
        self.last_n_layers = last_n_layers

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.llm = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)

        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad_(False)

        hidden_size = self.llm.config.hidden_size

        self.layernorm = nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()
        self.proj = nn.Linear(hidden_size, cond_dim)

    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        texts: List[str]
        return: Tensor [B, cond_dim]
        """
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.llm(
            **batch,
            output_hidden_states=True,
            return_dict=True
        )

        # hidden_states: tuple(len_layers+1), each [B, T, D]
        hidden_states = outputs.hidden_states

        # ---- Scheme B: average last L layers ----
        h = torch.stack(
            hidden_states[-self.last_n_layers:], dim=0
        ).mean(dim=0)  # [B, T, D]

        # ---- masked mean pooling ----
        mask = batch["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        
        pooled = self.layernorm(pooled)
        cond = pooled
        # cond = self.proj(pooled)  # [B, cond_dim]

        return cond
