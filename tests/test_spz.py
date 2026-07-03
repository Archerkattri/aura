"""Tests for the SPZ v4 (.spz) reader/writer (aura.spz).

Covers: exact 32-byte header bytes, six-stream ZSTD layout + TOC, per-attribute
round-trip within *quantization tolerance* (SPZ is lossy — equality is impossible),
the confidence sidecar (alignment + provenance), and malformed-input rejection.
All CI-runnable (small synthetic arrays, pure CPU, zstandard is a core dependency).
The one end-to-end fixture test is gated behind ``local_data``.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura import spz as S  # noqa: E402


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _cloud(n=64, deg=1, seed=0):
    """A low-level SPZ cloud with values inside every quantizer's representable
    range so round-trip errors reflect the quantization step, not clamping."""
    rng = np.random.default_rng(seed)
    shd = S._sh_dim(deg)
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return dict(
        positions=(rng.uniform(-3.0, 3.0, (n, 3))),
        scales=(rng.uniform(-5.0, 1.0, (n, 3))),          # log-scale, in range
        rotations=q,                                       # xyzw, unit
        alphas=(rng.uniform(-4.0, 4.0, n)),                # logit
        colors=(rng.uniform(-2.0, 2.0, (n, 3))),           # f_dc, in ±3.33
        sh=(rng.uniform(-0.9, 0.9, (n, shd, 3)) if shd else None),
        sh_degree=deg,
    )


def _carriers(n=64, seed=1, with_sh=False, with_conf=True):
    """An AURA carrier dict (linear scales, wxyz quats, opacity in [0,1])."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4)).astype("float32")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    c = dict(
        means=(rng.uniform(-3, 3, (n, 3))).astype("float32"),
        scales=np.exp(rng.uniform(-5, 0, (n, 3))).astype("float32"),   # linear
        quats=q,
        opacity=rng.uniform(0.02, 0.98, n).astype("float32"),
        sh_degree=0,
    )
    if with_sh:
        deg = 3
        K = (deg + 1) ** 2
        sh = (rng.uniform(-0.9, 0.9, (n, K, 3))).astype("float32")
        sh[:, 0, :] = rng.uniform(-2, 2, (n, 3))          # DC = f_dc
        c["sh"] = sh
        c["sh_degree"] = deg
    else:
        c["colors"] = rng.uniform(0, 1, (n, 3)).astype("float32")
    if with_conf:
        c["confidence"] = rng.uniform(0, 1, n).astype("float32")
    return c


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------
def test_header_bytes_exact_degree0():
    """Degree 0 -> 5 streams (no SH); exact 32-byte little-endian NgspFileHeader."""
    c = _cloud(n=10, deg=0)
    data = S.serialize_spz(c)
    expect = struct.pack("<IIIBBBBI12s", S.NGSP_MAGIC, 4, 10, 0,
                         S.DEFAULT_FRACTIONAL_BITS, 0, 5, S.HEADER_SIZE, b"\x00" * 12)
    assert data[:32] == expect
    assert data[:4] == b"NGSP"


def test_header_bytes_exact_degree2():
    """Degree >= 1 -> 6 streams; antialiased flag lands in the flags byte."""
    c = _cloud(n=7, deg=2)
    data = S.serialize_spz(c, antialiased=True)
    expect = struct.pack("<IIIBBBBI12s", S.NGSP_MAGIC, 4, 7, 2,
                         S.DEFAULT_FRACTIONAL_BITS, S.FLAG_ANTIALIASED, 6,
                         S.HEADER_SIZE, b"\x00" * 12)
    assert data[:32] == expect


def test_header_readable_without_decompression():
    c = _cloud(n=33, deg=3)
    data = S.serialize_spz(c)
    h = S.read_spz_header(data)                      # from bytes, no stream decode
    assert h["magic"] == S.NGSP_MAGIC and h["version"] == 4
    assert h["numPoints"] == 33 and h["shDegree"] == 3
    assert h["fractionalBits"] == 12 and h["numStreams"] == 6
    assert h["tocByteOffset"] == 32 and not h["hasExtensions"]


def test_header_readable_from_path(tmp_path):
    p = tmp_path / "x.spz"
    p.write_bytes(S.serialize_spz(_cloud(n=5, deg=0)))
    h = S.read_spz_header(p)
    assert h["numPoints"] == 5 and h["numStreams"] == 5


# ---------------------------------------------------------------------------
# stream / TOC layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("deg,nstreams", [(0, 5), (1, 6), (2, 6), (3, 6), (4, 6)])
def test_stream_layout_and_toc(deg, nstreams):
    n = 40
    c = _cloud(n=n, deg=deg)
    data = S.serialize_spz(c)
    h = S.read_spz_header(data)
    assert h["numStreams"] == nstreams

    toc_off = h["tocByteOffset"]
    entries = [struct.unpack("<QQ", data[toc_off + i * 16: toc_off + i * 16 + 16])
               for i in range(nstreams)]
    shd = S._sh_dim(deg)
    # uncompressed sizes, in reference stream order: pos, alpha, color, scale, rot, sh
    expected_uncompressed = [n * 9, n * 1, n * 3, n * 3, n * 4] + ([n * shd * 3] if shd else [])
    assert [us for _cs, us in entries] == expected_uncompressed
    # compressed chunks are laid back-to-back and exactly fill the file
    total = toc_off + nstreams * 16 + sum(cs for cs, _us in entries)
    assert total == len(data)
    # each compressed chunk decodes to its declared uncompressed size
    import zstandard
    off = toc_off + nstreams * 16
    for cs, us in entries:
        raw = zstandard.ZstdDecompressor().decompress(data[off:off + cs], max_output_size=us)
        assert len(raw) == us
        off += cs


# ---------------------------------------------------------------------------
# round-trip within quantization tolerance (cloud level)
# ---------------------------------------------------------------------------
def test_roundtrip_cloud_within_quantization_bounds():
    c = _cloud(n=256, deg=2, seed=3)
    back = S.deserialize_spz(S.serialize_spz(c))
    assert back["numPoints"] == 256 and back["sh_degree"] == 2

    eps = 1e-6
    pos_err = np.abs(back["positions"] - c["positions"]).max()
    assert pos_err <= 0.5 / (1 << 12) + eps                      # 1/2 LSB fixed-point

    scale_err = np.abs(back["scales"] - c["scales"]).max()
    assert scale_err <= 0.5 / 16.0 + eps                          # 1/2 LSB log-scale

    color_err = np.abs(back["colors"] - c["colors"]).max()
    assert color_err <= 0.5 / (S._COLOR_SCALE * 255.0) + eps      # 1/2 LSB wide-RGB

    op_in = 1.0 / (1.0 + np.exp(-c["alphas"]))
    op_out = 1.0 / (1.0 + np.exp(-back["alphas"]))
    assert np.abs(op_in - op_out).max() <= 0.5 / 255.0 + eps      # 1/2 LSB alpha

    # smallest-three quaternion: within-quantization, |cos angle| ~ 1
    a = c["rotations"] / np.linalg.norm(c["rotations"], axis=1, keepdims=True)
    b = back["rotations"] / np.linalg.norm(back["rotations"], axis=1, keepdims=True)
    assert np.abs((a * b).sum(1)).min() >= 0.9999

    sh_err = np.abs(back["sh"] - c["sh"]).max()
    assert sh_err <= 0.07                                          # bucketed SH step/2


def test_positions_fixedpoint_signed_symmetric():
    """24-bit signed fixed-point recovers both signs symmetrically."""
    c = _cloud(n=50, deg=0, seed=9)
    c["positions"] = np.linspace(-2.0, 2.0, 150).reshape(50, 3)
    back = S.deserialize_spz(S.serialize_spz(c))
    assert np.abs(back["positions"] - c["positions"]).max() <= 0.5 / (1 << 12) + 1e-6


# ---------------------------------------------------------------------------
# round-trip (AURA carrier level)
# ---------------------------------------------------------------------------
def test_roundtrip_carriers_flat_color(tmp_path):
    c = _carriers(n=128, with_sh=False)
    out = S.write_spz(c, tmp_path / "s.spz")
    assert out.suffix == ".spz" and out.exists()
    back = S.read_spz(out)
    assert np.abs(back["means"] - c["means"]).max() <= 0.5 / (1 << 12) + 1e-4
    assert (np.abs(back["scales"] - c["scales"]) / c["scales"]).max() <= 0.05   # ~3% log LSB
    assert np.abs(back["opacity"] - c["opacity"]).max() <= 0.5 / 255.0 + 1e-6
    assert np.abs(back["colors"] - c["colors"]).max() <= S._C0 * 0.5 / (S._COLOR_SCALE * 255.0) + 1e-4
    a = c["quats"] / np.linalg.norm(c["quats"], axis=1, keepdims=True)
    b = back["quats"] / np.linalg.norm(back["quats"], axis=1, keepdims=True)
    assert np.abs((a * b).sum(1)).min() >= 0.9999


def test_roundtrip_carriers_with_sh(tmp_path):
    c = _carriers(n=96, with_sh=True)
    back = S.read_spz(S.write_spz(c, tmp_path / "sh.spz"))
    assert back["sh_degree"] == 3 and "sh" in back
    assert back["sh"].shape == (96, 16, 3)
    # DC channel (f_dc) round-trips at wide-RGB precision; higher orders at bucket step
    assert np.abs(back["sh"][:, 0, :] - c["sh"][:, 0, :]).max() <= 0.5 / (S._COLOR_SCALE * 255.0) + 1e-4
    assert np.abs(back["sh"][:, 1:, :] - c["sh"][:, 1:, :]).max() <= 0.07


def test_write_spz_forces_suffix(tmp_path):
    out = S.write_spz(_carriers(n=4, with_conf=False), tmp_path / "noext")
    assert out.name == "noext.spz" and out.exists()


# ---------------------------------------------------------------------------
# confidence sidecar
# ---------------------------------------------------------------------------
def test_confidence_sidecar_written_and_aligned(tmp_path):
    c = _carriers(n=200, with_conf=True)
    out = S.write_spz(c, tmp_path / "scene.spz")
    side = S.confidence_sidecar_path(out)
    assert side.name == "scene.spz.confidence.npz" and side.exists()

    conf, prov = S.read_confidence_sidecar(out)
    np.testing.assert_array_equal(conf, c["confidence"])
    assert prov["count"] == 200 and prov["channel"] == "confidence"
    assert prov["spz_file"] == "scene.spz"

    # read_spz re-attaches the sidecar, aligned 1:1 to point order.
    back = S.read_spz(out)
    np.testing.assert_array_equal(back["confidence"], c["confidence"])


def test_confidence_sidecar_preserves_point_order(tmp_path):
    """A permutation of the carriers permutes the confidence identically —
    proving the join is positional (numPoints/point order)."""
    c = _carriers(n=64, with_conf=True)
    perm = np.random.default_rng(0).permutation(64)
    cp = {k: (v[perm] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == 64 else v)
          for k, v in c.items()}
    back = S.read_spz(S.write_spz(cp, tmp_path / "p.spz"))
    np.testing.assert_array_equal(back["confidence"], c["confidence"][perm])
    # positions moved together with confidence
    assert np.abs(back["means"] - cp["means"]).max() <= 0.5 / (1 << 12) + 1e-4


def test_confidence_provenance_is_json_able(tmp_path):
    out = S.write_spz(_carriers(n=8), tmp_path / "j.spz",
                      provenance={"scene": "unit", "calibrator": "isotonic"})
    _conf, prov = S.read_confidence_sidecar(out)
    assert json.loads(json.dumps(prov))["scene"] == "unit"          # survives JSON


def test_no_confidence_no_sidecar(tmp_path):
    out = S.write_spz(_carriers(n=8, with_conf=False), tmp_path / "n.spz")
    assert not S.confidence_sidecar_path(out).exists()
    conf, prov = S.read_confidence_sidecar(out)
    assert conf is None and prov is None


def test_confidence_length_mismatch_raises(tmp_path):
    with pytest.raises(ValueError):
        S.write_confidence_sidecar(tmp_path / "b.spz", np.zeros(5, "float32"), num_points=6)


# ---------------------------------------------------------------------------
# malformed input rejection
# ---------------------------------------------------------------------------
def _valid_bytes(n=16, deg=1):
    return bytearray(S.serialize_spz(_cloud(n=n, deg=deg)))


def test_reject_too_short():
    with pytest.raises(ValueError):
        S.deserialize_spz(b"NGSP\x04\x00\x00\x00")


def test_reject_bad_magic():
    d = _valid_bytes()
    d[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="not an NGSP"):
        S.deserialize_spz(bytes(d))


def test_reject_legacy_gzip():
    with pytest.raises(ValueError, match="legacy gzip"):
        S.deserialize_spz(b"\x1f\x8b\x08\x00" + b"\x00" * 40)


def test_reject_wrong_version():
    d = _valid_bytes()
    struct.pack_into("<I", d, 4, 3)                 # version 3
    with pytest.raises(ValueError, match="version"):
        S.deserialize_spz(bytes(d))


def test_reject_numstreams_mismatch():
    d = _valid_bytes(deg=1)                          # 6 streams
    d[15] = 5                                         # claim 5
    with pytest.raises(ValueError, match="streams"):
        S.deserialize_spz(bytes(d))


def test_reject_toc_offset_in_header():
    d = _valid_bytes()
    struct.pack_into("<I", d, 16, 8)                 # tocByteOffset < 32
    with pytest.raises(ValueError):
        S.deserialize_spz(bytes(d))


def test_reject_toc_past_eof():
    d = _valid_bytes()
    struct.pack_into("<I", d, 16, len(d) + 100)      # toc beyond file
    with pytest.raises(ValueError):
        S.deserialize_spz(bytes(d))


def test_reject_truncated_stream_data():
    d = _valid_bytes()
    with pytest.raises(ValueError):
        S.deserialize_spz(bytes(d[:-3]))             # drop tail of last chunk


def test_reject_uncompressed_size_mismatch():
    d = _valid_bytes()
    h = S.read_spz_header(bytes(d))
    # corrupt the first TOC entry's uncompressed size
    struct.pack_into("<Q", d, h["tocByteOffset"] + 8, 999999)
    with pytest.raises(ValueError, match="uncompressed size"):
        S.deserialize_spz(bytes(d))


def test_reject_empty_cloud():
    with pytest.raises(ValueError):
        S.serialize_spz(dict(positions=np.zeros((0, 3)), scales=np.zeros((0, 3)),
                             rotations=np.zeros((0, 4)), alphas=np.zeros(0),
                             colors=np.zeros((0, 3)), sh_degree=0))


def test_reject_bad_sh_bits():
    with pytest.raises(ValueError):
        S.serialize_spz(_cloud(n=4, deg=1), sh1_bits=0)
    with pytest.raises(ValueError):
        S.serialize_spz(_cloud(n=4, deg=1), sh_rest_bits=9)


# ---------------------------------------------------------------------------
# end-to-end fixture (gitignored real carriers; excluded from CI)
# ---------------------------------------------------------------------------
_TRUCK = Path(__file__).resolve().parent.parent / "outputs" / "truck-sidecar.aura" / "carriers.npz"


@pytest.mark.local_data
@pytest.mark.skipif(not _TRUCK.exists(), reason="truck carriers fixture not present")
def test_truck_fixture_spz_roundtrip(tmp_path):
    """Real 129k-carrier scene: write SPZ4 + confidence sidecar, read back, and
    confirm every attribute recovers within its quantization tolerance and the
    confidence sidecar stays aligned to point order."""
    z = np.load(_TRUCK)
    carriers = {k: z[k] for k in z.files}
    n = carriers["means"].shape[0]

    out = S.write_spz(carriers, tmp_path / "truck.spz")
    h = S.read_spz_header(out)
    assert h["numPoints"] == n and h["version"] == 4
    assert S.confidence_sidecar_path(out).exists()

    back = S.read_spz(out)
    assert back["means"].shape == (n, 3)
    assert np.abs(back["means"] - carriers["means"]).max() <= 0.5 / (1 << 12) + 1e-3
    assert np.abs(back["opacity"] - carriers["opacity"]).max() <= 0.5 / 255.0 + 1e-5
    np.testing.assert_array_equal(back["confidence"], carriers["confidence"].astype("float32"))
