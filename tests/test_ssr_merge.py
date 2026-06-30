import torch

from ssr_merge import SSRLayer, SSRStats


def _layer(name, down, up):
    return SSRLayer(
        lora_base=name,
        target_base="diffusion_model.layer",
        down=down,
        up=up,
        alpha=float(down.shape[0]),
    )


def test_ssr_solve_emits_standard_lora_keys():
    task_a = {
        "diffusion_model.layer": _layer(
            "diffusion_model.layer",
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )
    }
    task_b = {
        "diffusion_model.layer": _layer(
            "diffusion_model.layer",
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([[0.5], [0.25]]),
        )
    }

    stats = SSRStats([task_a, task_b], lambda_reg=1e-4, model_strengths=[1.0, 1.0])
    stats.set_active_task(0)
    stats.accumulate("diffusion_model.layer", torch.eye(2))
    stats.set_active_task(1)
    stats.accumulate("diffusion_model.layer", torch.ones(2, 2))

    merged = stats.solve()

    assert "diffusion_model.layer.lora_down.weight" in merged
    assert "diffusion_model.layer.lora_up.weight" in merged
    assert "diffusion_model.layer.alpha" in merged
    assert merged["diffusion_model.layer.lora_down.weight"].shape == (3, 2)
    assert merged["diffusion_model.layer.lora_up.weight"].shape == (2, 3)
    assert merged["diffusion_model.layer.alpha"].item() == 3.0


def test_ssr_identity_fallback_when_no_activation():
    task_a = {"diffusion_model.layer": _layer("diffusion_model.layer", torch.eye(2), torch.eye(2))}
    task_b = {
        "diffusion_model.layer": _layer(
            "diffusion_model.layer",
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([[1.0], [0.0]]),
        )
    }

    stats = SSRStats([task_a, task_b], lambda_reg=1e-4, model_strengths=[1.0, 1.0])
    merged = stats.solve()

    assert merged["diffusion_model.layer.lora_down.weight"].shape == (3, 2)
    assert any("identity router" in warning for warning in stats.warnings)
