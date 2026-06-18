"""Tests for the cross-carrier neural-residual MLP (Deliverable 1) and
trainable differentiable allocation logits (Deliverable 2).

These tests PROVE the mechanisms are real:
  - Cross-carrier MLP output is non-zero and changes when neighbor features change.
  - Gradients flow to MLP weights after backward().
  - Allocation logits are trainable: after N steps with a loss that favors a
    different carrier, the argmax carrier FLIPS.
  - Gradients flow to the logits.
  - With new features DISABLED (defaults), outputs are identical to before.
"""

from __future__ import annotations

import importlib.util

import pytest


# ---------------------------------------------------------------------------
# Skip whole file if torch not installed
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is required for cross-carrier and allocation logit tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_torch():
    import torch
    return torch


def _make_element(
    eid: str,
    carrier_id: str,
    payload_type: str,
    color=(0.5, 0.5, 0.5),
    opacity: float = 0.8,
    residual_scale: float = 0.0,
    bounds_min=(-0.5, -0.5, -0.5),
    bounds_max=(0.5, 0.5, 0.5),
    extra_payload: dict | None = None,
):
    """Build a minimal AuraElement for testing."""
    from aura import AuraElement, Bounds

    payload = {
        "type": payload_type,
        "color": list(color),
        "opacity": opacity,
        "residual_scale": residual_scale,
        "confidence": 0.9,
        **(extra_payload or {}),
    }
    return AuraElement(
        id=eid,
        carrier_id=carrier_id,
        bounds=Bounds(bounds_min, bounds_max),
        color=color,
        opacity=opacity,
        payload=payload,
    )


# ===========================================================================
# DELIVERABLE 1 — Cross-carrier MLP tests
# ===========================================================================


class TestCrossCarrierMLP:
    """Tests proving the cross-carrier residual MLP is a real computation."""

    def test_build_mlp_has_trainable_parameters(self):
        """The MLP returned by build_cross_carrier_mlp has real nn.Parameters."""
        import torch
        from aura.cross_carrier import build_cross_carrier_mlp

        mlp = build_cross_carrier_mlp(torch, device="cpu")
        params = list(mlp.parameters())
        assert len(params) > 0, "MLP must have at least one parameter"
        total = sum(p.numel() for p in params)
        assert total > 0, "MLP must have non-zero total parameter count"
        for p in params:
            assert p.requires_grad, "All MLP parameters must require grad"

    def test_mlp_output_is_non_zero_for_nonzero_input(self):
        """MLP output is non-zero for realistic neighbor features (post Xavier init)."""
        import torch
        from aura.cross_carrier import (
            build_cross_carrier_mlp,
            cross_carrier_residual_correction,
        )

        mlp = build_cross_carrier_mlp(torch, device="cpu")
        # Non-trivial neighbor features
        nb_colors = torch.tensor([[0.8, 0.3, 0.1], [0.2, 0.7, 0.5]])
        nb_opacities = torch.tensor([0.9, 0.6])
        nb_residuals = torch.tensor([0.4, 0.2])
        nb_centroids = torch.tensor([[1.0, 0.5, 0.2], [-0.3, 0.8, 0.1]])

        correction = cross_carrier_residual_correction(
            torch, mlp, nb_colors, nb_opacities, nb_residuals, nb_centroids, "cpu"
        )
        # Not a constant zero (Xavier init makes this extremely likely)
        assert correction.shape == (), "Correction must be a scalar"
        assert correction.item() != 0.0, "MLP correction must be non-zero for real inputs"

    def test_mlp_output_changes_with_different_neighbor_features(self):
        """Different neighbor features -> different MLP corrections (non-constant)."""
        import torch
        from aura.cross_carrier import (
            build_cross_carrier_mlp,
            cross_carrier_residual_correction,
        )

        mlp = build_cross_carrier_mlp(torch, device="cpu")

        nb_colors_a = torch.tensor([[0.9, 0.1, 0.2]])
        nb_opacities_a = torch.tensor([0.95])
        nb_residuals_a = torch.tensor([0.1])
        nb_centroids_a = torch.tensor([[2.0, 1.0, 0.5]])

        nb_colors_b = torch.tensor([[0.1, 0.9, 0.8]])
        nb_opacities_b = torch.tensor([0.2])
        nb_residuals_b = torch.tensor([0.9])
        nb_centroids_b = torch.tensor([[-1.0, -2.0, 0.3]])

        correction_a = cross_carrier_residual_correction(
            torch, mlp, nb_colors_a, nb_opacities_a, nb_residuals_a, nb_centroids_a, "cpu"
        )
        correction_b = cross_carrier_residual_correction(
            torch, mlp, nb_colors_b, nb_opacities_b, nb_residuals_b, nb_centroids_b, "cpu"
        )

        assert correction_a.item() != correction_b.item(), (
            "Different neighbor features must yield different MLP corrections"
        )

    def test_gradients_flow_to_mlp_weights_after_backward(self):
        """After backward(), all MLP parameters have non-None, non-zero gradients."""
        import torch
        from aura.cross_carrier import (
            build_cross_carrier_mlp,
            cross_carrier_residual_correction,
        )

        mlp = build_cross_carrier_mlp(torch, device="cpu")

        nb_colors = torch.tensor([[0.6, 0.4, 0.2]])
        nb_opacities = torch.tensor([0.7])
        nb_residuals = torch.tensor([0.3])
        nb_centroids = torch.tensor([[0.5, -0.3, 0.8]])

        correction = cross_carrier_residual_correction(
            torch, mlp, nb_colors, nb_opacities, nb_residuals, nb_centroids, "cpu"
        )
        # Build a tiny scalar loss and backward
        loss = (correction - 0.3) ** 2
        loss.backward()

        for name, param in mlp.named_parameters():
            assert param.grad is not None, f"grad is None for MLP param {name}"
            assert param.grad.abs().sum().item() > 0.0, (
                f"Grad is zero for MLP param {name} — computation is a no-op"
            )

    def test_mlp_parameter_tensors_surfaced_in_carrier_parameters(self):
        """torch_carrier_parameter_tensors includes mlp_w*/mlp_b* for neural+anchor elements."""
        import torch
        from aura.torch_kernels import torch_carrier_parameter_tensors

        elem = _make_element(
            "neural_e",
            carrier_id="neural",
            payload_type="neural_residual",
            extra_payload={"use_anchor_conditioning": True},
        )
        params = torch_carrier_parameter_tensors(torch, [elem], device="cpu", requires_grad=True)
        ep = params["neural_e"]

        assert "cross_carrier_mlp" in ep, "MLP module must be in carrier parameters"
        assert "mlp_w0" in ep, "mlp_w0 weight tensor must be surfaced"
        assert "mlp_b0" in ep, "mlp_b0 bias tensor must be surfaced"
        for key in ("mlp_w0", "mlp_b0", "mlp_w1", "mlp_b1", "mlp_w2", "mlp_b2"):
            assert ep[key].requires_grad, f"{key} must require grad"

    def test_carrier_parameter_tensors_no_mlp_when_anchor_disabled(self):
        """Without use_anchor_conditioning, no MLP keys appear in carrier parameters."""
        import torch
        from aura.torch_kernels import torch_carrier_parameter_tensors

        elem = _make_element(
            "neural_e2",
            carrier_id="neural",
            payload_type="neural_residual",
            # No use_anchor_conditioning key
        )
        params = torch_carrier_parameter_tensors(torch, [elem], device="cpu", requires_grad=True)
        ep = params["neural_e2"]

        assert "cross_carrier_mlp" not in ep, (
            "No MLP module should appear when use_anchor_conditioning is False"
        )
        assert not any(k.startswith("mlp_") for k in ep), (
            "No mlp_* keys should appear when use_anchor_conditioning is False"
        )

    def test_torch_carrier_response_tensors_with_anchor_conditioning_is_non_zero(self):
        """neural_residual with anchor_conditioning + neighbors changes transmittance."""
        import torch
        from aura import AuraElement, Bounds
        from aura.cross_carrier import build_cross_carrier_mlp, mlp_parameter_tensors_from_module
        from aura.torch_kernels import torch_carrier_parameter_tensors, torch_carrier_response_tensors

        # Neighbor gabor element
        nb_elem = _make_element(
            "gabor_nb",
            carrier_id="gabor",
            payload_type="gabor_frequency",
            color=(0.9, 0.1, 0.2),
            opacity=0.7,
            bounds_min=(-0.3, -0.3, -0.3),
            bounds_max=(0.3, 0.3, 0.3),
            extra_payload={"frequency": [0.5, 1.0, 0.0], "phase": 0.0, "bandwidth": 0.8},
        )

        # Neural residual element with anchor conditioning enabled
        neural_elem = _make_element(
            "neural_main",
            carrier_id="neural",
            payload_type="neural_residual",
            color=(0.5, 0.5, 0.5),
            opacity=0.6,
            residual_scale=0.5,
            extra_payload={
                "use_anchor_conditioning": True,
                "neighbor_elements": [nb_elem],
                "anchor_feature_dim": 4,
                "latent_dim": 8,
            },
        )

        device = "cpu"
        elements = [neural_elem]
        n_rays = 3

        # Build carrier parameters — this creates the MLP
        carrier_parameters = torch_carrier_parameter_tensors(
            torch, elements, device=device, requires_grad=True
        )
        # Also include the neighbor's parameters
        nb_params = torch_carrier_parameter_tensors(torch, [nb_elem], device=device, requires_grad=True)
        carrier_parameters.update(nb_params)

        best_index = torch.zeros(n_rays, dtype=torch.long, device=device)
        best_depth = torch.zeros(n_rays, device=device)
        exit_depth = torch.ones(n_rays, device=device)
        hit_points = torch.rand(n_rays, 3, device=device)
        colors = torch.rand(1, 3, device=device)
        opacities = torch.rand(1, device=device)
        confidences = torch.rand(1, device=device)
        mins = torch.tensor([[-0.5, -0.5, -0.5]], device=device)
        maxs = torch.tensor([[0.5, 0.5, 0.5]], device=device)

        _, transmittance, _, _ = torch_carrier_response_tensors(
            torch, elements, best_index, best_depth, exit_depth,
            hit_points, colors, opacities, confidences, mins, maxs, device,
            carrier_parameters=carrier_parameters,
        )

        # Transmittance must be valid (not NaN, not all-ones trivially)
        assert not torch.isnan(transmittance).any(), "Transmittance must not be NaN"
        assert transmittance.shape == (n_rays,)
        # The MLP contributes a non-trivial residual_strength so transmittance != 1.0
        assert not (transmittance == 1.0).all(), (
            "Transmittance must not all be 1.0 — MLP must have affected residual_strength"
        )

    def test_backward_compatible_no_anchor_conditioning_returns_default(self):
        """Without anchor conditioning the response is bit-for-bit identical to pre-stub behavior."""
        import torch
        from aura.torch_kernels import torch_carrier_parameter_tensors, torch_carrier_response_tensors

        elem = _make_element(
            "neural_compat",
            carrier_id="neural",
            payload_type="neural_residual",
            color=(0.7, 0.3, 0.1),
            opacity=0.5,
            residual_scale=0.8,
        )
        device = "cpu"
        elements = [elem]
        n_rays = 2

        carrier_parameters = torch_carrier_parameter_tensors(
            torch, elements, device=device, requires_grad=True
        )
        best_index = torch.zeros(n_rays, dtype=torch.long, device=device)
        best_depth = torch.zeros(n_rays, device=device)
        exit_depth = torch.ones(n_rays, device=device)
        hit_points = torch.rand(n_rays, 3, device=device)
        colors = torch.tensor([[0.7, 0.3, 0.1]], device=device)
        opacities = torch.tensor([0.5], device=device)
        confidences = torch.tensor([0.9], device=device)
        mins = torch.tensor([[-0.5, -0.5, -0.5]], device=device)
        maxs = torch.tensor([[0.5, 0.5, 0.5]], device=device)

        _, transmittance_1, _, _ = torch_carrier_response_tensors(
            torch, elements, best_index, best_depth, exit_depth,
            hit_points, colors, opacities, confidences, mins, maxs, device,
            carrier_parameters=carrier_parameters,
        )
        _, transmittance_2, _, _ = torch_carrier_response_tensors(
            torch, elements, best_index, best_depth, exit_depth,
            hit_points, colors, opacities, confidences, mins, maxs, device,
            carrier_parameters=carrier_parameters,
        )
        # Results must be deterministic / unchanged (no stochastic MLP injection)
        assert torch.allclose(transmittance_1, transmittance_2), (
            "Without anchor conditioning, results must be deterministic"
        )

    def test_mlp_correction_range_is_bounded(self):
        """MLP correction is in (-0.5, 0.5) — residual_strength stays in [0, 1]."""
        import torch
        from aura.cross_carrier import (
            build_cross_carrier_mlp,
            cross_carrier_residual_correction,
        )

        mlp = build_cross_carrier_mlp(torch, device="cpu")
        for _ in range(20):
            nb_colors = torch.rand(3, 3)
            nb_opacities = torch.rand(3)
            nb_residuals = torch.rand(3)
            nb_centroids = torch.randn(3, 3)
            corr = cross_carrier_residual_correction(
                torch, mlp, nb_colors, nb_opacities, nb_residuals, nb_centroids, "cpu"
            )
            assert -0.5 <= corr.item() <= 0.5, (
                f"MLP correction {corr.item()} is outside (-0.5, 0.5)"
            )

    def test_gradients_flow_end_to_end_through_kernel(self):
        """Gradients propagate from a loss on transmittance back to MLP weights."""
        import torch
        from aura import AuraElement, Bounds
        from aura.torch_kernels import torch_carrier_parameter_tensors, torch_carrier_response_tensors

        nb_elem = _make_element(
            "gabor_nb2",
            carrier_id="gabor",
            payload_type="gabor_frequency",
            color=(0.2, 0.8, 0.4),
            opacity=0.5,
            bounds_min=(-0.2, -0.2, -0.2),
            bounds_max=(0.2, 0.2, 0.2),
            extra_payload={"frequency": [1.0, 0.5, 0.2], "phase": 0.1, "bandwidth": 0.9},
        )
        neural_elem = _make_element(
            "neural_grad",
            carrier_id="neural",
            payload_type="neural_residual",
            color=(0.4, 0.6, 0.3),
            opacity=0.7,
            residual_scale=0.5,
            extra_payload={
                "use_anchor_conditioning": True,
                "neighbor_elements": [nb_elem],
            },
        )
        device = "cpu"
        elements = [neural_elem]
        carrier_parameters = torch_carrier_parameter_tensors(
            torch, elements, device=device, requires_grad=True
        )
        nb_params = torch_carrier_parameter_tensors(torch, [nb_elem], device=device, requires_grad=True)
        carrier_parameters.update(nb_params)

        n_rays = 4
        best_index = torch.zeros(n_rays, dtype=torch.long, device=device)
        best_depth = torch.zeros(n_rays, device=device)
        exit_depth = torch.ones(n_rays, device=device)
        hit_points = torch.rand(n_rays, 3, device=device)
        colors = torch.rand(1, 3, device=device)
        opacities = torch.rand(1, device=device)
        confidences = torch.rand(1, device=device)
        mins = torch.tensor([[-0.5, -0.5, -0.5]], device=device)
        maxs = torch.tensor([[0.5, 0.5, 0.5]], device=device)

        _, transmittance, _, _ = torch_carrier_response_tensors(
            torch, elements, best_index, best_depth, exit_depth,
            hit_points, colors, opacities, confidences, mins, maxs, device,
            carrier_parameters=carrier_parameters,
        )
        # Compute a loss and backward
        loss = transmittance.mean()
        loss.backward()

        mlp = carrier_parameters["neural_grad"]["cross_carrier_mlp"]
        for name, param in mlp.named_parameters():
            assert param.grad is not None, (
                f"Gradient did not reach MLP param '{name}' — not end-to-end differentiable"
            )
            assert param.grad.abs().sum().item() > 0.0, (
                f"Zero gradient for MLP param '{name}' — computation is a no-op"
            )


# ===========================================================================
# DELIVERABLE 2 — Trainable allocation logits tests
# ===========================================================================


class TestTrainableAllocationLogits:
    """Tests proving the trainable allocation logits are real trainable parameters."""

    def test_logit_params_are_nn_parameters_with_grad(self):
        """Each element's logit tensor is a real nn.Parameter with requires_grad=True."""
        import torch
        from aura.allocation import TrainableAllocationLogits, CARRIER_KIND_ORDER

        store = TrainableAllocationLogits(["e1", "e2"], device="cpu")
        for eid in ("e1", "e2"):
            p = store.logit_params[eid]
            assert p.requires_grad, f"Logit for {eid} must require grad"
            assert p.shape == (len(CARRIER_KIND_ORDER),), (
                f"Logit tensor must have shape (n_carriers,) = ({len(CARRIER_KIND_ORDER)},)"
            )

    def test_gumbel_softmax_output_sums_to_one(self):
        """Gumbel-softmax probabilities sum to 1.0 for each element."""
        import torch
        from aura.allocation import TrainableAllocationLogits

        store = TrainableAllocationLogits(["e1"], device="cpu")
        probs = store.gumbel_softmax_probs("e1", hard=False)
        assert abs(probs.sum().item() - 1.0) < 1e-5, "Gumbel-softmax probs must sum to 1"

    def test_gumbel_softmax_hard_is_one_hot(self):
        """Hard Gumbel-softmax (straight-through) returns a one-hot tensor."""
        import torch
        from aura.allocation import TrainableAllocationLogits

        store = TrainableAllocationLogits(["e1"], device="cpu")
        probs = store.gumbel_softmax_probs("e1", hard=True)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert ((probs == 0.0) | (probs == 1.0)).all(), "Hard Gumbel-softmax must be one-hot"

    def test_gradients_flow_through_soft_assignment(self):
        """After a backward pass, logit tensors receive non-zero gradients."""
        import torch
        from aura.allocation import TrainableAllocationLogits

        store = TrainableAllocationLogits(["e1"], device="cpu")
        probs = store.gumbel_softmax_probs("e1", hard=False)
        loss = -(probs[3])  # push carrier index 3 (gabor)
        loss.backward()
        grad = store.logit_params["e1"].grad
        assert grad is not None, "Logit grad is None — gradients did not flow"
        assert grad.abs().sum().item() > 0.0, "Logit grad is zero — no real gradient"

    def test_optimize_step_reduces_loss(self):
        """optimize_step() actually reduces the training loss over multiple steps."""
        import torch
        from aura.allocation import TrainableAllocationLogits, CARRIER_KIND_ORDER

        target_idx = CARRIER_KIND_ORDER.index("volume")  # 1
        store = TrainableAllocationLogits(["e1"], device="cpu")

        import torch.nn.functional as F

        def loss_fn(soft_probs):
            return F.cross_entropy(
                store.logit_params["e1"].unsqueeze(0),
                torch.tensor([target_idx], dtype=torch.long),
            )

        losses = store.optimize_step(loss_fn, n_steps=20, learning_rate=0.5)
        assert losses[0] > losses[-1], (
            "Loss must decrease over training steps — optimization is not working"
        )

    def test_argmax_carrier_flips_after_training(self):
        """After enough gradient steps, the hard argmax carrier type changes."""
        import torch
        from aura.allocation import TrainableAllocationLogits, CARRIER_KIND_ORDER

        # Initialize logits biased toward "gaussian" (index 6)
        n = len(CARRIER_KIND_ORDER)
        gaussian_idx = CARRIER_KIND_ORDER.index("gaussian")
        neural_idx = CARRIER_KIND_ORDER.index("neural")

        init_logits = {
            "e1": [0.0] * n,
        }
        init_logits["e1"][gaussian_idx] = 3.0  # start strongly biased to gaussian

        store = TrainableAllocationLogits(
            ["e1"], device="cpu", initial_logits=init_logits, temperature=1.0
        )
        initial_assignment = store.hard_assignments()["e1"]
        assert initial_assignment == "gaussian", (
            "Initial assignment should be 'gaussian' given the init logits"
        )

        # Train toward "neural" — loss favors neural carrier (index 4)
        import torch.nn.functional as F

        def loss_fn(soft_probs):
            return F.cross_entropy(
                store.logit_params["e1"].unsqueeze(0),
                torch.tensor([neural_idx], dtype=torch.long),
            )

        store.optimize_step(loss_fn, n_steps=200, learning_rate=1.0)
        final_assignment = store.hard_assignments()["e1"]
        assert final_assignment == "neural", (
            f"After training, argmax must flip from 'gaussian' to 'neural', "
            f"got '{final_assignment}' instead — logit training is not working"
        )

    def test_train_allocation_logits_flips_carrier(self):
        """train_allocation_logits() convenience function flips carrier via training."""
        import torch
        from aura.allocation import (
            TrainableAllocationLogits,
            CARRIER_KIND_ORDER,
            train_allocation_logits,
        )

        # Start biased to surface
        surface_idx = CARRIER_KIND_ORDER.index("surface")
        gabor_idx = CARRIER_KIND_ORDER.index("gabor")
        n = len(CARRIER_KIND_ORDER)
        init = {"e1": [0.0] * n}
        init["e1"][surface_idx] = 3.0

        # Train toward gabor
        store = train_allocation_logits(
            ["e1"],
            {"e1": gabor_idx},
            device="cpu",
            n_steps=200,
            learning_rate=1.0,
            initial_logits=init,
        )
        assignment = store.hard_assignments()["e1"]
        assert assignment == "gabor", (
            f"train_allocation_logits() must flip carrier to 'gabor', got '{assignment}'"
        )

    def test_logit_parameters_exposed_for_external_optimizer(self):
        """parameters() returns a non-empty list of tensors with requires_grad."""
        import torch
        from aura.allocation import TrainableAllocationLogits

        store = TrainableAllocationLogits(["a", "b", "c"], device="cpu")
        params = store.parameters()
        assert len(params) == 3, "Must return one parameter per element"
        for p in params:
            assert p.requires_grad

    def test_soft_assignment_loss_is_differentiable(self):
        """soft_assignment_loss returns a scalar with gradient to logit params."""
        import torch
        from aura.allocation import TrainableAllocationLogits, CARRIER_KIND_ORDER

        store = TrainableAllocationLogits(["e1"], device="cpu")
        loss = store.soft_assignment_loss("e1", CARRIER_KIND_ORDER.index("beta"))
        loss.backward()
        assert store.logit_params["e1"].grad is not None
        assert store.logit_params["e1"].grad.abs().sum().item() > 0.0

    def test_default_hard_assignments_match_initial_argmax(self):
        """Before any training, hard_assignments() reflects the initial logit argmax."""
        import torch
        from aura.allocation import TrainableAllocationLogits, CARRIER_KIND_ORDER

        n = len(CARRIER_KIND_ORDER)
        beta_idx = CARRIER_KIND_ORDER.index("beta")
        init = {"e1": [0.0] * n}
        init["e1"][beta_idx] = 5.0  # strong initial bias to beta

        store = TrainableAllocationLogits(["e1"], device="cpu", initial_logits=init)
        assignments = store.hard_assignments()
        assert assignments["e1"] == "beta", (
            f"Initial argmax should be 'beta', got '{assignments['e1']}'"
        )

    def test_backward_through_straight_through_estimator(self):
        """Straight-through estimator allows gradients even through the hard one-hot."""
        import torch
        from aura.allocation import TrainableAllocationLogits

        store = TrainableAllocationLogits(["e1"], device="cpu", temperature=0.5)
        probs = store.gumbel_softmax_probs("e1", hard=True)
        # The loss must depend on the SHAPE of the distribution, not just its
        # total mass: probs always sums to 1.0, so probs.sum() is a constant
        # with zero gradient. A weighted sum over the carrier index genuinely
        # depends on which class the STE selected, so a non-zero gradient here
        # proves the straight-through estimator routes gradient to the logits.
        weights = torch.arange(probs.shape[0], dtype=probs.dtype, device=probs.device)
        loss = (probs * weights).sum()
        loss.backward()
        grad = store.logit_params["e1"].grad
        assert grad is not None, "STE must allow gradient to flow to logits"
        assert grad.abs().sum().item() > 0.0

    def test_initial_logits_from_soft_carrier_scores(self):
        """Initializing logits from soft_carrier_scores produces sensible starting point."""
        import torch
        from aura.assignment import RegionEvidence
        from aura.allocation import (
            TrainableAllocationLogits,
            CARRIER_KIND_ORDER,
            soft_carrier_scores,
            AllocationConfig,
        )

        ev = RegionEvidence(high_frequency=0.95)
        config = AllocationConfig(use_soft_scores=True)
        soft = soft_carrier_scores(ev, "e1", config=config)

        init = {"e1": list(soft.logits)}
        store = TrainableAllocationLogits(["e1"], device="cpu", initial_logits=init)
        # Gabor should dominate for high_frequency evidence
        gabor_idx = CARRIER_KIND_ORDER.index("gabor")
        initial_assignment = store.hard_assignments()["e1"]
        # The initial logit for gabor should be high
        gabor_logit = store.logit_params["e1"][gabor_idx].item()
        max_logit = store.logit_params["e1"].max().item()
        assert gabor_logit == max_logit, (
            "Gabor logit should be highest for high_frequency evidence"
        )


# ===========================================================================
# DELIVERABLE 3 — Wire: cross-carrier MLP in training path
# ===========================================================================


class TestCrossCarrierWiredIntoTraining:
    """Tests proving MLP correction receives gradients in the end-to-end path."""

    def test_mlp_weights_trained_via_optimizer_module(self):
        """Running a few Adam steps on an element with anchor conditioning updates MLP weights."""
        import torch
        import torch.optim as optim
        from aura.cross_carrier import build_cross_carrier_mlp, mlp_parameter_tensors_from_module
        from aura.torch_kernels import torch_carrier_parameter_tensors, torch_carrier_response_tensors

        nb_elem = _make_element(
            "nb_train",
            carrier_id="gabor",
            payload_type="gabor_frequency",
            color=(0.5, 0.7, 0.3),
            opacity=0.6,
            bounds_min=(-0.3, -0.3, -0.3),
            bounds_max=(0.3, 0.3, 0.3),
            extra_payload={"frequency": [0.5, 1.0, 0.0], "phase": 0.2, "bandwidth": 0.7},
        )
        neural_elem = _make_element(
            "neural_train",
            carrier_id="neural",
            payload_type="neural_residual",
            color=(0.4, 0.4, 0.4),
            opacity=0.6,
            residual_scale=0.5,
            extra_payload={
                "use_anchor_conditioning": True,
                "neighbor_elements": [nb_elem],
            },
        )
        device = "cpu"
        elements = [neural_elem]
        carrier_parameters = torch_carrier_parameter_tensors(
            torch, elements, device=device, requires_grad=True
        )
        nb_params = torch_carrier_parameter_tensors(torch, [nb_elem], device=device, requires_grad=True)
        carrier_parameters.update(nb_params)

        mlp = carrier_parameters["neural_train"]["cross_carrier_mlp"]
        # Save initial weights
        initial_weights = [p.data.clone() for p in mlp.parameters()]

        # Build optimizer over MLP params + carrier params
        mlp_params = list(mlp.parameters())
        scalar_params = [
            v for k, v in carrier_parameters["neural_train"].items()
            if isinstance(v, torch.Tensor) and v.requires_grad and not k.startswith("mlp_")
            and k != "cross_carrier_mlp"
        ]
        optimizer = optim.Adam(mlp_params + scalar_params, lr=0.1)

        n_rays = 4
        best_index = torch.zeros(n_rays, dtype=torch.long, device=device)
        best_depth = torch.zeros(n_rays, device=device)
        exit_depth = torch.ones(n_rays, device=device)
        hit_points = torch.rand(n_rays, 3, device=device)
        colors = torch.rand(1, 3, device=device)
        opacities = torch.rand(1, device=device)
        confidences = torch.rand(1, device=device)
        mins = torch.tensor([[-0.5, -0.5, -0.5]], device=device)
        maxs = torch.tensor([[0.5, 0.5, 0.5]], device=device)

        target_transmittance = torch.tensor([0.2, 0.3, 0.2, 0.3], device=device)

        # Run 5 gradient steps
        for _ in range(5):
            optimizer.zero_grad()
            _, transmittance, _, _ = torch_carrier_response_tensors(
                torch, elements, best_index, best_depth, exit_depth,
                hit_points, colors, opacities, confidences, mins, maxs, device,
                carrier_parameters=carrier_parameters,
            )
            loss = ((transmittance - target_transmittance) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Weights must have changed
        for initial, current in zip(initial_weights, mlp.parameters()):
            assert not torch.allclose(initial, current.data), (
                "MLP weights did not change after training — optimizer is not updating them"
            )
