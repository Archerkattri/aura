from aura import CarrierKind, RegionEvidence, choose_carrier


def test_surface_like_region_selects_surface_cell():
    evidence = RegionEvidence(
        geometry_confidence=0.92,
        material_confidence=0.75,
        edit_need=0.8,
    )

    assert choose_carrier(evidence).kind is CarrierKind.SURFACE_CELL


def test_fuzzy_region_selects_volume_cell():
    evidence = RegionEvidence(
        fuzzy_confidence=0.9,
        geometry_confidence=0.2,
    )

    assert choose_carrier(evidence).kind is CarrierKind.VOLUME_CELL


def test_high_frequency_region_selects_gabor_carrier():
    evidence = RegionEvidence(
        high_frequency=0.95,
        geometry_confidence=0.65,
    )

    assert choose_carrier(evidence).kind is CarrierKind.GABOR_FREQUENCY


def test_view_dependent_region_selects_neural_residual():
    evidence = RegionEvidence(
        view_dependent=0.9,
        material_confidence=0.2,
    )

    assert choose_carrier(evidence).kind is CarrierKind.NEURAL_RESIDUAL


def test_low_demand_region_selects_gaussian_fallback():
    evidence = RegionEvidence(
        image_error=0.05,
        geometry_confidence=0.45,
        ray_need=0.1,
        edit_need=0.1,
    )

    assert choose_carrier(evidence).kind is CarrierKind.GAUSSIAN_FALLBACK


def test_region_evidence_rejects_value_out_of_unit_interval():
    """Line 12: _unit() raises ValueError when value is outside [0, 1]."""
    import pytest
    from aura.assignment import RegionEvidence
    with pytest.raises((ValueError, TypeError)):
        RegionEvidence(image_error=1.5)

