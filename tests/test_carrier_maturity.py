"""Carrier-registry honesty: maturity contract, hybrid routing provenance, and the
quarantined neural footprint. All CPU/CI-safe (no CUDA, no gsplat, no torch tensors)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura import carriers as CA  # noqa: E402
from aura import hybrid as H  # noqa: E402
from aura import runtime_export as RE  # noqa: E402


# --- registry maturity ----------------------------------------------------------

def test_every_carrier_has_a_valid_maturity():
    reg = CA.default_registry()
    for cid, spec in reg.items():
        assert spec.maturity in CA.CARRIER_MATURITIES, cid


def test_maturity_map_matches_expected_contract():
    m = CA.carrier_maturity_map()
    assert m["gaussian"] == CA.MATURITY_TRAINED
    assert m["beta"] == CA.MATURITY_TRAINED
    assert m["gabor"] == CA.MATURITY_DEMO
    assert m["neural"] == CA.MATURITY_DEMO
    assert m["surface"] == CA.MATURITY_METADATA
    assert m["volume"] == CA.MATURITY_METADATA
    assert m["semantic"] == CA.MATURITY_METADATA


def test_only_gaussian_and_beta_are_trained():
    trained = {c for c, m in CA.carrier_maturity_map().items() if m == CA.MATURITY_TRAINED}
    assert trained == {"gaussian", "beta"}


def test_unknown_maturity_rejected_at_construction():
    with pytest.raises(ValueError):
        CA.CarrierSpec(id="x", kind=CA.CarrierKind.GAUSSIAN_FALLBACK,
                       description="", primary_render=True, ray_query=True,
                       maturity="production")


# --- hybrid routing provenance (the anti-silent-fallback contract) --------------

def test_routing_keeps_gaussian_and_beta_on_primary_backend():
    routes = {r["carrier"]: r for r in H.footprint_routing(
        [H.FOOTPRINT_CODES["gaussian"], H.FOOTPRINT_CODES["beta"]])}
    assert routes["gaussian"]["layer"] == "primary" and routes["gaussian"]["fallback"] is None
    assert routes["beta"]["layer"] == "primary" and routes["beta"]["fallback"] is None


def test_routing_marks_gabor_as_implemented_prism_extension():
    (r,) = H.footprint_routing([H.FOOTPRINT_CODES["gabor"]])
    assert r["layer"] == "prism"
    assert r["footprint"] == "gabor"
    assert r["fallback"] is None


def test_routing_marks_neural_as_explicit_gaussian_fallback():
    (r,) = H.footprint_routing([H.FOOTPRINT_CODES["neural"]])
    assert r["layer"] == "prism"
    assert r["footprint"] == "gaussian"          # composited via gaussian, not a neural kernel
    assert r["fallback"] == "fallback:gaussian"  # and it SAYS SO
    assert H.fallback_carrier_codes([H.FOOTPRINT_CODES["neural"]]) == [H.FOOTPRINT_CODES["neural"]]


def test_beta_in_prism_is_opt_in():
    codes = [H.FOOTPRINT_CODES["beta"]]
    assert H.footprint_routing(codes)[0]["layer"] == "primary"
    assert H.footprint_routing(codes, include_beta=True)[0]["layer"] == "prism"
    # Beta has a real PRISM kernel, so even as an extension it is not a fallback.
    assert H.footprint_routing(codes, include_beta=True)[0]["fallback"] is None


def test_no_gaussian_carrier_reports_a_fallback():
    codes = [H.FOOTPRINT_CODES["gaussian"]] * 3
    assert H.fallback_carrier_codes(codes) == []


# --- runtime export advertises maturity -----------------------------------------

def test_runtime_export_carrier_entry_exposes_registry_maturity():
    assert RE._carrier_export_entry("gaussian")["maturity"] == "trained"
    assert RE._carrier_export_entry("beta")["maturity"] == "trained"
    assert RE._carrier_export_entry("gabor")["maturity"] == "demo"
    assert RE._carrier_export_entry("neural")["maturity"] == "demo"
    assert RE._carrier_export_entry("semantic")["maturity"] == "metadata"


# --- neural footprint is quarantined --------------------------------------------

def test_make_neural_footprint_refuses_without_experimental_flag():
    from aura import prism

    # The guard fires before touching torch, so a dummy stand-in is fine here.
    with pytest.raises(NotImplementedError):
        prism.make_neural_footprint(object(), enable_experimental=False)
