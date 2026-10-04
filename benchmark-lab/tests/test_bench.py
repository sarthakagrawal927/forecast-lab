"""Protocol and model guards. The protocol tests pin the details that make our
numbers comparable with the published tables."""
import numpy as np
import pandas as pd
import pytest
import torch

from bench import data
from bench.models import DLinear, Hybrid, PatchTransformer, StepTransformer, naive, seasonal_naive


def test_ett_hourly_borders_match_reference_loader():
    assert data.borders("ETTh1", 17420, 336) == [(0, 8640), (8304, 11520), (11184, 14400)]


def test_custom_dataset_borders_are_70_10_20():
    (a0, a1), (b0, b1), (c0, c1) = data.borders("weather", 1000, 96)
    assert (a1, b1, c1) == (700, 800, 1000) and b0 == 700 - 96 and c0 == 800 - 96


def test_test_window_count_matches_published_protocol(monkeypatch, tmp_path):
    """2785 test windows for ETTh1 at L=336, H=96 (every window scored)."""
    idx = pd.date_range("2016-07-01", periods=17420, freq="h")
    frame = pd.DataFrame({"OT": np.arange(17420, dtype=float)}, index=idx).rename_axis("date")
    monkeypatch.setattr(data, "load_frame", lambda name: frame)
    _, _, test, _ = data.make_splits("ETTh1", 336, 96)
    assert len(test) == 2785


def test_scaler_uses_training_segment_only(monkeypatch):
    idx = pd.date_range("2016-07-01", periods=17420, freq="h")
    values = np.r_[np.zeros(8640), np.full(17420 - 8640, 100.0)] + np.tile([0.0, 1.0], 8710)
    frame = pd.DataFrame({"OT": values}, index=idx).rename_axis("date")
    monkeypatch.setattr(data, "load_frame", lambda name: frame)
    train, _, test, _ = data.make_splits("ETTh1", 336, 96)
    assert abs(train.x.mean()) < 1e-3          # train is centred
    assert test.x[-1, 0] > 100                  # test level shift is NOT normalised away


def test_baselines_shapes_and_values():
    hist = torch.arange(48, dtype=torch.float32).view(1, 48, 1)
    assert torch.all(naive(hist, 5) == 47)
    sn = seasonal_naive(hist, 30, 24)
    assert sn.shape == (1, 30, 1) and sn[0, 0, 0] == 24 and sn[0, 24, 0] == 24


@pytest.mark.parametrize("model", [
    DLinear(336, 96), StepTransformer(336, 96, 7, d_model=16, n_layers=1), PatchTransformer(336, 96, 7)])
def test_models_map_history_to_horizon(model):
    assert model(torch.randn(2, 336, 7)).shape == (2, 96, 7)


def test_dlinear_on_constant_series_learns_nothing_new():
    """The moving average of a constant is the constant: the decomposition is exact."""
    m = DLinear(48, 12)
    x = torch.full((1, 48, 3), 2.0)
    pad = (m.kernel - 1) // 2
    padded = torch.cat([x[:, :1].expand(-1, pad, -1), x, x[:, -1:].expand(-1, pad, -1)], dim=1)
    assert torch.allclose(padded.unfold(1, m.kernel, 1).mean(-1), x)


def test_hybrid_starts_exactly_at_its_linear_anchor():
    torch.manual_seed(0)
    anchor = DLinear(336, 96)
    hybrid = Hybrid(anchor, PatchTransformer(336, 96, 7, zero_head=True)).eval()
    x = torch.randn(4, 336, 7) * 3 + 5
    torch.testing.assert_close(hybrid(x), anchor(x))
