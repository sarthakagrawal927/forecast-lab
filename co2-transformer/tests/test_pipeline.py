"""Guards for the properties the results depend on: no look-ahead, padding is
invisible to the model, the head starts at persistence, losses ignore
unobserved cells, and the API's feature path equals the training path."""
import numpy as np
import pandas as pd
import pytest
import torch

from co2tx.data import AGE_CAP, Preprocessor, make_windows, observation_state, window_at
from co2tx.model import CO2Transformer, pinball_loss
from deploy.load_db import parse_tag


def synthetic_run(T=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-01", periods=T, freq="43s")
    point = np.tile(np.repeat(np.arange(1, 7), 3), T)[:T]
    return pd.DataFrame({"FT103(kg/hr)": rng.normal(250, 10, T), "TT210(0C)": rng.normal(44, 1, T),
                         "co2": point * 0.5 + rng.normal(0, 0.05, T), "point": point}, index=idx)


def test_observation_state_is_causal():
    run = synthetic_run()
    co2, point = run["co2"].to_numpy(), run["point"].to_numpy()
    fill = np.zeros(6)
    last_a, age_a, _ = observation_state(co2, point, fill)
    co2_future = co2.copy()
    co2_future[25:] += 100  # change only the future
    last_b, age_b, _ = observation_state(co2_future, point, fill)
    np.testing.assert_array_equal(last_a[:25], last_b[:25])
    np.testing.assert_array_equal(age_a, age_b)


def test_ages_count_steps_since_each_point_was_read():
    point = np.array([1, 1, 2, 3, 1])
    _, age, seen = observation_state(np.ones(5), point, np.zeros(6))
    assert age[4, 0] == 0 and age[4, 1] == 2 and age[4, 2] == 1
    assert age[4, 5] == AGE_CAP and seen[4, 5] == 0


def test_window_left_pads_and_ends_at_origin():
    feats = np.arange(10, dtype=np.float32)[:, None]
    win, pad = window_at(feats, t=2, lookback=5)
    assert pad.tolist() == [True, True, False, False, False]
    assert win[-1, 0] == 2 and win[0, 0] == 0


def test_targets_only_where_analyser_measured():
    runs = {"a": synthetic_run()}
    pre = Preprocessor.fit(runs)
    w = make_windows(pre, runs, lookback=6, horizon=4)
    assert (w.mask.sum(-1) <= 1).all()  # at most one point measured per future step
    t, h = 10, 2
    p = runs["a"]["point"].iloc[t + h] - 1
    assert w.mask[t, h - 1, p] == 1
    assert w.y[t, h - 1, p] == pytest.approx(runs["a"]["co2"].iloc[t + h])


def _model(L=8, F=5, H=3):
    torch.manual_seed(0)
    return CO2Transformer(n_features=F, lookback=L, horizon=H, d_model=16, n_heads=2, d_ff=32).eval()


def test_padded_steps_do_not_change_the_forecast():
    m = _model()
    x = torch.randn(1, 8, 5)
    pad = torch.tensor([[True] * 3 + [False] * 5])
    last = torch.zeros(1, 6)
    x2 = x.clone()
    x2[:, :3] = 999.0  # garbage in padded positions
    torch.testing.assert_close(m(x, pad, last), m(x2, pad, last))


def test_untrained_head_equals_persistence_with_ordered_quantiles():
    m = _model()
    last = torch.rand(4, 6) * 5
    out = m(torch.randn(4, 8, 5), torch.zeros(4, 8, dtype=torch.bool), last)
    torch.testing.assert_close(out[..., 1], last[:, None, :].expand(-1, 3, -1))
    assert (out[..., 0] < out[..., 1]).all() and (out[..., 1] < out[..., 2]).all()


def test_loss_ignores_unobserved_cells():
    pred = torch.zeros(2, 3, 6, 3)
    y = torch.zeros(2, 3, 6)
    mask = torch.zeros(2, 3, 6)
    mask[0, 0, 5] = 1
    y_noise = y.clone()
    y_noise[1] = 50.0  # unobserved: must not matter
    assert pinball_loss(pred, y, mask) == pinball_loss(pred, y_noise, mask)


def test_serving_features_match_training_features():
    """The API rebuilds history from Postgres; the window it feeds the model
    must equal the training window at the same origin."""
    runs = {"a": synthetic_run()}
    pre = Preprocessor.fit(runs)
    w = make_windows(pre, runs, lookback=6, horizon=4)
    t = 17
    feats, _ = pre.step_features(runs["a"].iloc[: t + 1])
    win, pad = window_at(feats, t, 6)
    np.testing.assert_array_equal(win, w.x[t])
    np.testing.assert_array_equal(pad, w.pad[t])


@pytest.mark.parametrize("tag,expected", [
    ("FT103(kg/hr)", ("FT103", "kg/hr", "flow")),
    ("TT210(0C)", ("TT210", "degC", "temperature")),
    ("FT303m3/hr", ("FT303", "m3/hr", "flow")),
    ("TT110a(0C)", ("TT110a", "degC", "temperature")),
])
def test_tag_parsing(tag, expected):
    assert parse_tag(tag) == expected


def test_seen_flags_sit_where_metrics_expect_them():
    """train.metrics reads the seen flags as columns [-12:-6] of the last step."""
    runs = {"a": synthetic_run()}
    pre = Preprocessor.fit(runs)
    pre.step_features(runs["a"])
    assert pre.names[-12:-6] == [f"seen_p{i}" for i in range(1, 7)]


def test_router_uses_soft_sensor_only_for_points_not_yet_read():
    from co2tx.train import cold_cells, route

    run = synthetic_run()  # visits points 1..6 in order, 3 steps each
    pre = Preprocessor.fit({"a": run})
    w = make_windows(pre, {"a": run}, lookback=4, horizon=2)
    warm, cold = np.zeros(w.y.shape), np.ones(w.y.shape)
    out = route(warm, cold, w)
    # At origin 4 only points 1-2 have been read: they stay warm, 3-6 are cold.
    assert out[4, :, :2].max() == 0 and out[4, :, 2:].min() == 1
    assert not cold_cells(w)[-1].any()  # by the end of the run every point has been read


def test_finalize_orders_quantiles_then_clips_at_zero():
    from co2tx.train import finalize

    q = np.array([[0.5, 0.2, -0.3], [1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(finalize(q), [[0.0, 0.2, 0.5], [1.0, 2.0, 3.0]])
