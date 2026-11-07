"""Learned reward model: scores how well a candidate rule fulfils a request."""

from pathlib import Path

import torch
from torch import nn

from .llm import format_prompt, load_lora_model

MAX_TOKENS = 220


class ScoreHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class RewardModel(nn.Module):
    """The SFT backbone, which already reads the DSL, plus a scalar head on the final token."""

    def __init__(self, backbone, tokenizer):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.head = ScoreHead(backbone.config.hidden_size).to(backbone.device)

    def encode(self, requests: list[str], candidates: list[str]) -> dict:
        eos = self.tokenizer.eos_token
        texts = [format_prompt(r) + c + eos for r, c in zip(requests, candidates)]
        previous = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        batch = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS
        )
        self.tokenizer.padding_side = previous
        return batch.to(self.backbone.device)

    def forward(self, requests: list[str], candidates: list[str]) -> torch.Tensor:
        batch = self.encode(requests, candidates)
        with torch.autocast("cuda", dtype=torch.float16):
            output = self.backbone(**batch, output_hidden_states=True)
        last = batch["attention_mask"].sum(dim=1) - 1
        features = output.hidden_states[-1][torch.arange(last.numel()), last]
        return self.head(features.float())

    @torch.inference_mode()
    def score(self, requests: list[str], candidates: list[str], batch_size: int = 16) -> torch.Tensor:
        self.eval()
        scores = [
            self(requests[i : i + batch_size], candidates[i : i + batch_size])
            for i in range(0, len(requests), batch_size)
        ]
        return torch.cat(scores)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        self.backbone.save_pretrained(path)
        torch.save(self.head.state_dict(), path / "score_head.pt")

    @classmethod
    def load(cls, path: Path | str, tokenizer, device: str = "cuda", trainable: bool = False):
        backbone = load_lora_model(path, device, trainable=trainable)
        model = cls(backbone, tokenizer)
        model.head.load_state_dict(torch.load(Path(path) / "score_head.pt", map_location=device))
        return model


def bradley_terry_loss(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    """Negative log probability that the chosen response beats the rejected one."""
    return -torch.nn.functional.logsigmoid(chosen - rejected).mean()
