import pytest

from aura import (
    BetaKernelPayload,
    GaborFrequencyPayload,
    GaussianFallbackPayload,
    NeuralResidualPayload,
    SemanticFeaturePayload,
    SurfaceCellPayload,
    VolumeCellPayload,
)


def test_native_carrier_payloads_expose_typed_contracts():
    payloads = [
        SurfaceCellPayload(normal=(0.0, 0.0, 1.0), thickness=0.02).to_dict(),
        VolumeCellPayload(density=0.4, phase_anisotropy=0.1).to_dict(),
        BetaKernelPayload(alpha=2.0, beta=3.0, support_radius=(0.1, 0.2, 0.3)).to_dict(),
        GaborFrequencyPayload(frequency=(1.0, 0.0, 0.0), bandwidth=0.5).to_dict(),
        NeuralResidualPayload(latent_dim=16, residual_scale=0.25, model_ref="models/local.pt").to_dict(),
        GaussianFallbackPayload(
            mean=(0.0, 0.0, 0.0),
            covariance=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.0025)),
            source="test",
        ).to_dict(),
        SemanticFeaturePayload(label="chair", confidence=0.9, feature_refs=("clip:1",)).to_dict(),
    ]

    assert {payload["type"] for payload in payloads} == {
        "surface_cell",
        "volume_cell",
        "beta_kernel",
        "gabor_frequency",
        "neural_residual",
        "gaussian_fallback",
        "semantic_feature",
    }


def test_native_carrier_payloads_round_trip_from_dict():
    payloads = [
        (
            SurfaceCellPayload,
            SurfaceCellPayload(
                normal=(0.0, 0.0, 1.0),
                thickness=0.02,
                plane_point=(0.0, 0.0, -0.25),
            ).to_dict(),
        ),
        (VolumeCellPayload, VolumeCellPayload(density=0.4, phase_anisotropy=0.1).to_dict()),
        (BetaKernelPayload, BetaKernelPayload(alpha=2.0, beta=3.0, support_radius=(0.1, 0.2, 0.3)).to_dict()),
        (
            GaborFrequencyPayload,
            GaborFrequencyPayload(
                frequency=(1.0, 0.0, 0.0),
                bandwidth=0.5,
                plane_point=(0.0, 0.0, 0.125),
            ).to_dict(),
        ),
        (NeuralResidualPayload, NeuralResidualPayload(latent_dim=16, residual_scale=0.25, model_ref="models/local.pt").to_dict()),
        (
            GaussianFallbackPayload,
            GaussianFallbackPayload(
                mean=(0.0, 0.0, 0.0),
                covariance=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.0025)),
                source="test",
            ).to_dict(),
        ),
        (SemanticFeaturePayload, SemanticFeaturePayload(label="chair", confidence=0.9, feature_refs=("clip:1",)).to_dict()),
    ]

    for payload_type, payload in payloads:
        assert payload_type.from_dict(payload).to_dict() == payload


def test_payload_validation_rejects_invalid_values():
    with pytest.raises(ValueError, match="thickness"):
        SurfaceCellPayload(normal=(0.0, 0.0, 1.0), thickness=0.0).to_dict()
    with pytest.raises(ValueError, match="density"):
        VolumeCellPayload(density=1.5).to_dict()
    with pytest.raises(ValueError, match="latent_dim"):
        NeuralResidualPayload(latent_dim=0, residual_scale=0.2).to_dict()
    with pytest.raises(ValueError, match="payload type"):
        SurfaceCellPayload.from_dict({"type": "volume_cell", "normal": [0.0, 0.0, 1.0], "thickness": 0.1, "roughness": 0.5})


# ============================================================
# Step 4 new tests: payload-level regression tests for the
# four carrier upgrades (DBS, Gabor bank, Scaffold-GS, LangSplatV2)
# ============================================================


# --- A) DBS Beta Kernel tests ---

def test_dbs_beta_default_reproduces_prior():
    """New DBS fields at defaults must produce exactly the same to_dict() as before."""
    prior = BetaKernelPayload(alpha=2.0, beta=3.0, support_radius=(0.1, 0.2, 0.3)).to_dict()
    new = BetaKernelPayload(
        alpha=2.0, beta=3.0, support_radius=(0.1, 0.2, 0.3),
        adaptive_alpha=None, adaptive_beta=None,
        frequency_scale=1.0, appearance_shift=0.0,
    ).to_dict()
    assert prior == new
    # No DBS extension keys should appear in default output
    assert "adaptive_alpha" not in new
    assert "adaptive_beta" not in new
    assert "frequency_scale" not in new
    assert "appearance_shift" not in new


def test_dbs_beta_adaptive_shape():
    """adaptive_alpha/beta override base alpha/beta in serialised dict."""
    payload = BetaKernelPayload(
        alpha=2.0, beta=3.0, support_radius=(0.1, 0.2, 0.3),
        adaptive_alpha=4.5, adaptive_beta=1.2,
        frequency_scale=2.0, appearance_shift=0.1,
    )
    d = payload.to_dict()
    assert d["adaptive_alpha"] == pytest.approx(4.5)
    assert d["adaptive_beta"] == pytest.approx(1.2)
    assert d["frequency_scale"] == pytest.approx(2.0)
    assert d["appearance_shift"] == pytest.approx(0.1)
    # Round-trip
    rt = BetaKernelPayload.from_dict(d)
    assert rt.adaptive_alpha == pytest.approx(4.5)
    assert rt.adaptive_beta == pytest.approx(1.2)
    assert rt.frequency_scale == pytest.approx(2.0)
    assert rt.appearance_shift == pytest.approx(0.1)


# --- B) Gabor Filter Bank tests ---

def test_gabor_bank_default_single_filter():
    """num_filters=1 (default) must produce exactly the same to_dict() as before."""
    prior = GaborFrequencyPayload(frequency=(1.0, 0.0, 0.0), bandwidth=0.5).to_dict()
    new = GaborFrequencyPayload(
        frequency=(1.0, 0.0, 0.0), bandwidth=0.5,
        num_filters=1,
        frequencies=None, orientations=None, phases=None, filter_weights=None,
    ).to_dict()
    assert prior == new
    # No filter bank extension keys should appear in default output
    assert "num_filters" not in new
    assert "frequencies" not in new
    assert "orientations" not in new
    assert "phases" not in new
    assert "filter_weights" not in new


def test_gabor_bank_multi_filter():
    """num_filters=2 activates filter bank serialization."""
    payload = GaborFrequencyPayload(
        frequency=(1.0, 0.0, 0.0), bandwidth=0.5, phase=0.0,
        num_filters=2,
        frequencies=[0.5, 1.5],
        orientations=[0.0, 1.5707963],
        phases=[0.0, 0.1],
        filter_weights=[0.6, 0.4],
    )
    d = payload.to_dict()
    assert d["num_filters"] == 2
    assert d["frequencies"] == pytest.approx([0.5, 1.5])
    assert d["orientations"] == pytest.approx([0.0, 1.5707963])
    assert d["phases"] == pytest.approx([0.0, 0.1])
    assert d["filter_weights"] == pytest.approx([0.6, 0.4])
    # Round-trip
    rt = GaborFrequencyPayload.from_dict(d)
    assert rt.num_filters == 2
    assert rt.frequencies == pytest.approx([0.5, 1.5])
    assert rt.orientations == pytest.approx([0.0, 1.5707963])
    assert rt.phases == pytest.approx([0.0, 0.1])
    assert rt.filter_weights == pytest.approx([0.6, 0.4])


# --- C) Neural Residual Scaffold-GS tests ---

def test_neural_residual_scaffold_default():
    """Scaffold-GS fields at defaults must produce exactly the same to_dict() as before."""
    prior = NeuralResidualPayload(latent_dim=16, residual_scale=0.25, model_ref="models/local.pt").to_dict()
    new = NeuralResidualPayload(
        latent_dim=16, residual_scale=0.25, model_ref="models/local.pt",
        anchor_feature_dim=None, mlp_hidden_dim=64, num_mlp_layers=2,
        use_anchor_conditioning=False,
    ).to_dict()
    assert prior == new
    # No Scaffold-GS extension keys should appear in default output
    assert "anchor_feature_dim" not in new
    assert "mlp_hidden_dim" not in new
    assert "num_mlp_layers" not in new
    assert "use_anchor_conditioning" not in new


def test_neural_residual_scaffold_anchor():
    """anchor_feature_dim and other Scaffold-GS fields are serialised when set."""
    payload = NeuralResidualPayload(
        latent_dim=32, residual_scale=0.5,
        anchor_feature_dim=16, mlp_hidden_dim=128, num_mlp_layers=3,
        use_anchor_conditioning=True,
    )
    d = payload.to_dict()
    assert d["anchor_feature_dim"] == 16
    assert d["mlp_hidden_dim"] == 128
    assert d["num_mlp_layers"] == 3
    assert d["use_anchor_conditioning"] is True
    # Round-trip
    rt = NeuralResidualPayload.from_dict(d)
    assert rt.anchor_feature_dim == 16
    assert rt.mlp_hidden_dim == 128
    assert rt.num_mlp_layers == 3
    assert rt.use_anchor_conditioning is True


# --- D) LangSplatV2 Semantic Sparse Codebook tests ---

def test_semantic_sparse_codebook_default_dense():
    """use_sparse_codebook=False (default) must produce exactly the same to_dict() as before."""
    prior = SemanticFeaturePayload(label="chair", confidence=0.9, feature_refs=("clip:1",)).to_dict()
    new = SemanticFeaturePayload(
        label="chair", confidence=0.9, feature_refs=("clip:1",),
        use_sparse_codebook=False, codebook_size=256, codebook_dim=64,
        sparse_indices=None, sparse_weights=None,
    ).to_dict()
    assert prior == new
    # No sparse codebook extension keys should appear in default output
    assert "use_sparse_codebook" not in new
    assert "codebook_size" not in new
    assert "codebook_dim" not in new
    assert "sparse_indices" not in new
    assert "sparse_weights" not in new


def test_semantic_sparse_codebook_sparse():
    """use_sparse_codebook=True serialises and round-trips sparse codebook fields."""
    payload = SemanticFeaturePayload(
        label="chair", confidence=0.9,
        use_sparse_codebook=True,
        codebook_size=128, codebook_dim=32,
        sparse_indices=[0, 5, 12],
        sparse_weights=[0.5, 0.3, 0.2],
    )
    d = payload.to_dict()
    assert d["use_sparse_codebook"] is True
    assert d["codebook_size"] == 128
    assert d["codebook_dim"] == 32
    assert d["sparse_indices"] == [0, 5, 12]
    assert d["sparse_weights"] == pytest.approx([0.5, 0.3, 0.2])
    # Round-trip
    rt = SemanticFeaturePayload.from_dict(d)
    assert rt.use_sparse_codebook is True
    assert rt.codebook_size == 128
    assert rt.codebook_dim == 32
    assert rt.sparse_indices == [0, 5, 12]
    assert rt.sparse_weights == pytest.approx([0.5, 0.3, 0.2])


def test_semantic_sparse_codebook_decode():
    """decode_semantic_feature reconstructs feature vector from codebook atoms."""
    from aura.semantic import decode_semantic_feature

    payload_dense = SemanticFeaturePayload(label="chair", confidence=0.9).to_dict()
    assert decode_semantic_feature(payload_dense) is None  # dense path unchanged

    payload_sparse = SemanticFeaturePayload(
        label="chair", confidence=0.9,
        use_sparse_codebook=True,
        codebook_size=4, codebook_dim=3,
        sparse_indices=[0, 2],
        sparse_weights=[1.0, 0.5],
    ).to_dict()
    codebook = [
        [1.0, 0.0, 0.0],   # atom 0
        [0.0, 1.0, 0.0],   # atom 1
        [0.0, 0.0, 2.0],   # atom 2
        [1.0, 1.0, 1.0],   # atom 3
    ]
    result = decode_semantic_feature(payload_sparse, codebook=codebook)
    # Expected: 1.0 * atom[0] + 0.5 * atom[2] = [1.0, 0.0, 0.0] + [0.0, 0.0, 1.0] = [1.0, 0.0, 1.0]
    assert result == pytest.approx([1.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Cover lines 55, 275, 321, 398, 435 — error branches in payloads
# ---------------------------------------------------------------------------

def test_volume_cell_phase_anisotropy_out_of_range():
    """Line 55: phase_anisotropy outside [-1, 1] raises ValueError."""
    from aura.carrier_payloads import VolumeCellPayload
    with pytest.raises(ValueError, match="phase_anisotropy"):
        VolumeCellPayload(density=0.5, phase_anisotropy=1.5).to_dict()


def test_gaussian_fallback_covariance_not_3x3():
    """Line 275: covariance not 3x3 raises ValueError."""
    from aura.carrier_payloads import GaussianFallbackPayload
    bad_cov = ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
    with pytest.raises(ValueError, match="covariance must be a 3x3 matrix"):
        GaussianFallbackPayload(
            mean=(0.0, 0.0, 0.0),
            covariance=bad_cov,
        ).to_dict()


def test_semantic_feature_empty_label_raises():
    """Line 321: empty label raises ValueError."""
    from aura.carrier_payloads import SemanticFeaturePayload
    with pytest.raises(ValueError, match="label is required"):
        SemanticFeaturePayload(label="", confidence=0.9, feature_refs=()).to_dict()


def test_relighting_albedo_wrong_length_raises():
    """Line 398: albedo not a 3-element sequence raises ValueError."""
    from aura.carrier_payloads import RelightingPayload
    with pytest.raises(ValueError, match="albedo must be an RGB triple"):
        RelightingPayload(albedo=(0.5, 0.5)).to_dict()


def test_vec3_wrong_length_raises():
    """Line 435: _vec3 with wrong-length sequence raises ValueError."""
    from aura.carrier_payloads import RelightingPayload
    with pytest.raises(ValueError, match="vec3 payload fields"):
        RelightingPayload.from_dict({"type": "relighting", "albedo": [0.1, 0.2]})


def test_gaussian_fallback_non_positive_diagonal_raises():
    """Line 277: covariance with non-positive diagonal raises."""
    from aura.carrier_payloads import GaussianFallbackPayload
    bad_cov = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="diagonal entries must be positive"):
        GaussianFallbackPayload(mean=(0.0, 0.0, 0.0), covariance=bad_cov).to_dict()


def test_relighting_albedo_valid_emits_fields():
    """Lines 399-404: valid albedo + non-default roughness/metallic emitted."""
    from aura.carrier_payloads import RelightingPayload
    p = RelightingPayload(albedo=(0.8, 0.6, 0.4), shading_roughness=0.3, shading_metallic=0.2)
    d = p.to_dict()
    assert d["albedo"] == pytest.approx([0.8, 0.6, 0.4])
    assert d["shading_roughness"] == pytest.approx(0.3)
    assert d["shading_metallic"] == pytest.approx(0.2)


def test_relighting_from_dict_returns_instance():
    """Line 410: from_dict creates a RelightingPayload."""
    from aura.carrier_payloads import RelightingPayload
    p = RelightingPayload.from_dict({"type": "relighting", "albedo": [0.5, 0.5, 0.5]})
    assert p.albedo == pytest.approx((0.5, 0.5, 0.5))
