import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ea_direct_compare as m


def test_group_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(m.T_FINE, 2, m.MODEL_H, m.MODEL_W), dtype=np.int16)
    assert np.array_equal(m.ungrouped_null(m.grouped_null(X)), X)
    assert np.array_equal(m.ungrouped_free(m.grouped_free(X)), X)


def test_null_move_preserves_coarse():
    X = np.zeros((m.T_FINE, 2, m.MODEL_H, m.MODEL_W), dtype=np.int16)
    X[1, 0, 0, 0] = 3
    X[10, 1, 1, 1] = 2
    Xg = m.grouped_null(X).astype(np.int32)
    Gg = np.zeros_like(Xg, dtype=np.float32)
    Ag = Xg.copy()
    Gg[:, 2] = 1.0
    Xg2, _, shifts = m._select_moves(Xg, Gg, Ag, 2, m.S_PER_FRAME, None)
    Y = m.ungrouped_null(Xg2).astype(np.int16)
    assert int(Y.sum()) == int(X.sum())
    assert np.array_equal(m.coarse_np(X), m.coarse_np(Y))
    assert len(shifts) <= 2


def test_frame_metric_zero_on_null_equivalent():
    X = np.zeros((m.T_FINE, 2, m.MODEL_H, m.MODEL_W), dtype=np.int16)
    X[0, 0, 0, 0] = 1
    Y = X.copy()
    Y[0, 0, 0, 0] = 0
    Y[1, 0, 0, 0] = 1
    cm = m.coarse_metrics(X, Y)
    assert cm["frame_exact_equal"] is True
    assert cm["max_integer_frame_difference"] == 0


def test_model_shapes():
    x = torch.zeros(1, m.T_FINE, 2, m.MODEL_H, m.MODEL_W)
    for cls in (m.ConvSNN, m.SEWResNet18, m.FrameResNet18, m.CoarseFrameFormer):
        model = cls().eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, m.NUM_CLASSES)


def test_prior_delta_reconstruction_source_indexed():
    X = np.zeros((m.T_FINE, 2, m.MODEL_H, m.MODEL_W), dtype=np.int16)
    X[1, 0, 0, 0] = 3
    X[4, 1, 2, 3] = 2
    delta = np.zeros_like(X, dtype=np.int16)
    delta[1, 0, 0, 0] = 2
    delta[4, 1, 2, 3] = -1
    Y = np.zeros_like(X)
    Y[3, 0, 0, 0] = 3
    Y[3, 1, 2, 3] = 2
    assert m.verify_prior_delta_reconstruction(X, Y, delta) is True


def test_prior_delta_reconstruction_rejects_wrong_semantics():
    X = np.zeros((m.T_FINE, 2, m.MODEL_H, m.MODEL_W), dtype=np.int16)
    X[1, 0, 0, 0] = 1
    Y = np.zeros_like(X); Y[2, 0, 0, 0] = 1
    delta = np.zeros_like(X, dtype=np.int16)
    delta[2, 0, 0, 0] = 1  # destination-indexed: must fail
    try:
        m.verify_prior_delta_reconstruction(X, Y, delta)
    except AssertionError:
        pass
    else:
        raise AssertionError("destination-indexed delta should have failed")


def test_official_interface_gate_with_fake_class():
    class FakeAttack(nn.Module):
        def __init__(self, device, model_without_encoder, reduction="mean", steps=40,
                     alpha_phi=1.0, lambda_cap=20.0, temperature=1.0,
                     random_start=False, cap_limit=1.0, l0_moves_budget=1000,
                     lambda_B=0.0, dual_lr=0.1):
            super().__init__()
        def forward(self, x_enc_tbchw, labels, return_disp=False, use_PIL=True,
                    use_cap=True, use_penalty=True, target_label=-1):
            return x_enc_tbchw
    info = m.verify_official_prior_interface(FakeAttack)
    assert info["return_disp_verified"] is True
    assert info["nn_module_verified"] is True


def _dummy_row(method, seed=0):
    return {
        "seed": seed,
        "victim": "conv_snn",
        "sample_index": 0,
        "sample_path": "x",
        "label": 0,
        "clean_pred": 0,
        "clean_correct": True,
        "method": method,
        "requested_event_budget": 0.10,
        "total_event_units": 100,
        "unique_event_units_changed": 10,
        "unique_event_units_changed_pct": 0.10,
        "budget_match_abs_error": 0.0,
        "prior_b0_points_changed": np.nan,
        "prior_b0_points_requested": np.nan,
        "prior_b0_packets_changed": np.nan,
        "prior_b0_packets_requested": np.nan,
        "prior_budget_attempts": np.nan,
        "avg_abs_shift_bins": 1.0,
        "avg_abs_shift_ms": 2.0,
        "victim_pred_adv": 1,
        "attack_success_on_clean_correct": True,
        "protected_consumer": "coarse_frameformer",
        "clean_protected_consumer_pred": 0,
        "protected_consumer_pred_adv": 0,
        "protected_consumer_flip": False,
        "protected_consumer_logits_bit_identical": method == "null_space",
        "protected_consumer_max_abs_logit_difference": 0.0 if method == "null_space" else 1.0,
        "frame_exact_equal": method == "null_space",
        "max_integer_frame_difference": 0 if method == "null_space" else 2,
        "frame_l1_difference": 0 if method == "null_space" else 4,
        "seconds": 1.0,
    }


def test_table_order_budget_and_latex_percent_escape(tmp_path):
    rows = pd.DataFrame([
        _dummy_row("null_space"),
        _dummy_row("free_gradient"),
        _dummy_row("prior_pil_l0"),
    ])
    m.aggregate_outputs(rows, tmp_path)
    paper = pd.read_csv(tmp_path / "direct_retiming_table.csv")
    assert paper["Method"].tolist() == [
        "Yu et al. PIL-L0 retiming", "Free gradient retiming", "Null-space retiming"
    ]
    assert paper["budget"].tolist() == ["10%", "10%", "10%"]
    assert "Event units retimed" in paper.columns
    assert "Protected frame-consumer flips" in paper.columns
    assert "Max coarse-frame diff" in paper.columns
    tex = (tmp_path / "direct_retiming_table.tex").read_text()
    assert r"\%" in tex
    assert "10.00 ± 0.00%" not in tex


def test_prior_phi_preflight_flags_t160_quadratic_scale():
    z = m.prior_phi_preflight()
    assert z["our_T"] == 160
    assert z["official_example_T"] == 10
    assert z["temporal_quadratic_ratio_vs_T10"] == 256.0
    assert 0.7 < z["phi_single_tensor_fp32_gib"] < 0.9


def test_paired_protected_logits_bit_identical_for_same_coarse_view():
    class TinyCoarse(nn.Module):
        def forward(self, X):
            B = X.shape[0]
            coarse = X.reshape(B, m.N_COARSE, m.S_PER_FRAME, 2, m.MODEL_H, m.MODEL_W).sum(dim=2)
            # Deterministic scalar logits from only the protected coarse tensor.
            s = coarse.reshape(B, -1).sum(dim=1, keepdim=True)
            return torch.cat([s, -s], dim=1)

    X = np.zeros((m.T_FINE, 2, m.MODEL_H, m.MODEL_W), dtype=np.int16)
    X[0, 0, 0, 0] = 1
    Y = X.copy()
    Y[0, 0, 0, 0] = 0
    Y[1, 0, 0, 0] = 1
    assert np.array_equal(m.coarse_np(X), m.coarse_np(Y))
    z0, za = m.paired_protected_logits(TinyCoarse().eval(), X, Y)
    assert np.array_equal(z0, za)


def test_backend_matches_frozen_dvs_experiment():
    assert torch.backends.cudnn.benchmark is True


def test_protocol_fingerprint_separates_smoke_and_paper_runs():
    base = {
        "protocol_version": "v5",
        "limit": 2,
        "prior_steps": 2,
        "prior_recalibrate": False,
        "prior_attack_py_sha256": "abc",
        "protected_consumer": "coarse_frameformer",
    }
    smoke = m.protocol_fingerprint(base)
    paper_cfg = dict(base)
    paper_cfg["limit"] = 0
    paper_cfg["prior_steps"] = 40
    paper_cfg["prior_recalibrate"] = True
    paper = m.protocol_fingerprint(paper_cfg)
    assert smoke != paper
    assert smoke == m.protocol_fingerprint(dict(base))
