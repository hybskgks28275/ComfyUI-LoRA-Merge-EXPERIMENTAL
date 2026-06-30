import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from np_lora import adapter_delta, find_adapter_pairs, fuse_lora_state_dicts, np_project_content


def _state(base: str, up: torch.Tensor, down: torch.Tensor):
    return {
        base + ".lora_up.weight": up,
        base + ".lora_down.weight": down,
        base + ".alpha": torch.tensor(float(down.shape[0])),
    }


def test_mu_zero_is_direct_merge():
    torch.manual_seed(3)
    content = _state("layer", torch.randn(4, 2), torch.randn(2, 5))
    style = _state("layer", torch.randn(4, 3), torch.randn(3, 5))
    merged, warnings = fuse_lora_state_dicts(content, style, mu=0.0)
    assert not warnings
    pair = find_adapter_pairs(merged)["layer"]
    delta, _, _ = adapter_delta(merged, pair)
    expected = content["layer.lora_up.weight"] @ content["layer.lora_down.weight"]
    expected += style["layer.lora_up.weight"] @ style["layer.lora_down.weight"]
    assert torch.allclose(delta, expected, atol=2e-5, rtol=2e-5)


def test_higher_mu_suppresses_style_subspace_component():
    style = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    content = torch.tensor([[3.0, 2.0], [0.0, 0.0]])
    projected = np_project_content(content, style, mu=1.0e8)
    assert torch.allclose(projected, torch.tensor([[0.0, 2.0], [0.0, 0.0]]), atol=1e-5)
