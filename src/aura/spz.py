"""SPZ **version 4** reader/writer (Niantic ``.spz`` gaussian-splat container).

What this is
------------
A pure-``numpy`` implementation of the SPZ v4 (``NGSP``) container: a 32-byte
plaintext little-endian header, an optional extension zone, a table-of-contents,
and six independently ZSTD-compressed per-attribute streams (positions, alphas,
colors, scales, rotations, spherical-harmonics). Both **write** and **read** are
implemented, plus an AURA-carrier adapter (:func:`write_spz` / :func:`read_spz`)
and a confidence **sidecar** (SPZ v4 cannot carry a per-splat confidence channel
natively — see below).

Per-gaussian quantization (all matched to the reference):
  * positions — 24-bit signed fixed-point, ``fractionalBits`` (=12) fractional bits
  * scales    — 8-bit, log-scale, ``round((log_scale + 10) * 16)``
  * rotations — smallest-three quaternion (2-bit largest-index + 3x10-bit)
  * alphas    — 8-bit ``sigmoid(a) * 255``
  * colors    — 8-bit SH-DC "wide RGB", ``fdc * 0.15 * 255 + 127.5``
  * SH rest   — 8-bit signed, bucket-quantized (default 5 bits deg-1, 4 bits deg-2+)

Confidence
----------
SPZ v4 has **no** per-splat confidence attribute and no per-point extension
indexing (extensions are file-level global metadata blobs living uncompressed in
the plaintext header zone; brief section 1). AURA therefore writes its calibrated
per-carrier confidence to a **sidecar** ``<name>.spz.confidence.npz`` (a
``confidence`` float32 array aligned 1:1 to SPZ point order plus a JSON provenance
dict), mirroring the repo's ``carriers.npz`` sidecar pattern. ``numPoints`` from
the header is the join key.

Validation / honest scope
-------------------------
Byte-level layout and every quantizer follow the reference implementation
``github.com/nianticlabs/spz`` (``src/cc/load-spz.{h,cc}``, ``splat-types.h``) as
read on 2026-07-03 — not merely the README prose. The container written here has
been **cross-validated against the reference C++ core** (2026-07-03, spz commit
``bb0efad``): files this module writes decode correctly through the reference
``loadSpz``, files the reference ``saveSpz`` writes decode correctly here (both
within quantization tolerance), and on identical bytes the two decoders agree
bit-exactly on positions/scales/SH. The harness and its build/run instructions
are preserved at ``experiments/spz_reference_crossval.cc``; the validation is
not re-run in CI (CI covers the pure-python round-trip and byte-layout tests in
``tests/test_spz.py``). Quantization is intrinsically lossy, so round-trips recover values
within their quantization step, not exactly. Coordinate-system conversion is NOT
performed: input is stored verbatim (equivalent to the reference's default
``CoordinateSystem.UNSPECIFIED`` pack/unpack, which is identity). Legacy v1-3
(single-stream gzip) files are **not** read; this is a v4-only codec.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

# --- format constants (from load-spz.h / load-spz.cc) ----------------------
NGSP_MAGIC = 0x5053474E                 # "NGSP" little-endian
SPZ_VERSION = 4                         # LATEST_SPZ_HEADER_VERSION
HEADER_SIZE = 32                        # sizeof(NgspFileHeader)
DEFAULT_FRACTIONAL_BITS = 12            # ~0.25 mm at unit scale
DEFAULT_SH1_BITS = 5                    # PackOptions.sh1Bits
DEFAULT_SH_REST_BITS = 4               # PackOptions.shRestBits
FLAG_ANTIALIASED = 0x1
FLAG_HAS_EXTENSIONS = 0x2
_COLOR_SCALE = 0.15                     # spz colorScale for DC "wide RGB"
_SQRT1_2 = 0.7071067811865476          # 1/sqrt(2)
_ZSTD_LEVEL = 12                        # reference compressZstd default level

#: SH band-0 basis constant Y_0^0 (rgb = 0.5 + C0*f_dc); used only for the
#: AURA flat-colour <-> SH-DC bridge, matching gltf_splat / usd_writer.
_C0 = 0.28209479177387814

#: SH coefficient count (excluding DC) per SH degree — spz dimForDegree().
_SH_DIM_FOR_DEGREE = {0: 0, 1: 3, 2: 8, 3: 15, 4: 24}

CONFIDENCE_SIDECAR_SUFFIX = ".spz.confidence.npz"


# ---------------------------------------------------------------------------
# small numeric helpers (match the C++ semantics exactly)
# ---------------------------------------------------------------------------
def _sh_dim(degree: int) -> int:
    try:
        return _SH_DIM_FOR_DEGREE[int(degree)]
    except KeyError as exc:
        raise ValueError(f"unsupported SH degree {degree} (must be 0..4)") from exc


def _cround(x):
    """``std::round`` — round half **away from zero** (numpy rounds half-to-even)."""
    x = np.asarray(x, dtype=np.float64)
    return np.trunc(x + np.where(x >= 0.0, 0.5, -0.5))


def _to_uint8(x):
    """``toUint8``: clamp(round(x), 0, 255)."""
    return np.clip(_cround(x), 0.0, 255.0).astype(np.uint8)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _inv_sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log(x / (1.0 - x))


# ---------------------------------------------------------------------------
# per-attribute quantizers (vectorised; byte-identical to load-spz.cc)
# ---------------------------------------------------------------------------
def _pack_positions(positions, fractional_bits):
    scale = float(1 << fractional_bits)
    fixed = _cround(np.asarray(positions, dtype=np.float64) * scale).astype(np.int64)
    fixed &= 0xFFFFFF  # keep low 24 bits (two's-complement), like int32 -> 3 bytes
    out = np.empty((fixed.size, 3), dtype=np.uint8)
    flat = fixed.reshape(-1)
    out[:, 0] = flat & 0xFF
    out[:, 1] = (flat >> 8) & 0xFF
    out[:, 2] = (flat >> 16) & 0xFF
    return out.reshape(-1)  # [N*3*3] bytes, xyz-major then byte-minor


def _unpack_positions(buf, n, fractional_bits):
    b = np.asarray(buf, dtype=np.int64).reshape(n * 3, 3)
    u = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    u = np.where(u & 0x800000, u - 0x1000000, u)  # sign-extend bit 23
    return (u.astype(np.float64) / float(1 << fractional_bits)).reshape(n, 3)


def _pack_scales(scales_log):
    return _to_uint8((np.asarray(scales_log, dtype=np.float64) + 10.0) * 16.0).reshape(-1)


def _unpack_scales(buf, n):
    return (np.asarray(buf, dtype=np.float64) / 16.0 - 10.0).reshape(n, 3)


def _pack_alphas(alphas_logit):
    return _to_uint8(_sigmoid(alphas_logit) * 255.0).reshape(-1)


def _unpack_alphas(buf, n):
    return _inv_sigmoid(np.asarray(buf, dtype=np.float64) / 255.0).reshape(n)


def _pack_colors(colors_fdc):
    return _to_uint8(np.asarray(colors_fdc, dtype=np.float64) * (_COLOR_SCALE * 255.0)
                     + (0.5 * 255.0)).reshape(-1)


def _unpack_colors(buf, n):
    return ((np.asarray(buf, dtype=np.float64) / 255.0 - 0.5) / _COLOR_SCALE).reshape(n, 3)


def _pack_rotations_smallest_three(rotations_xyzw):
    """Smallest-three quaternion packing (packQuaternionSmallestThree)."""
    q = np.asarray(rotations_xyzw, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    absq = np.abs(q)
    i_largest = np.argmax(absq, axis=1)                       # [N]
    n = q.shape[0]
    rows = np.arange(n)
    negate = q[rows, i_largest] < 0.0                          # [N] bool
    comp = i_largest.astype(np.uint32)
    for i in range(4):
        mask = i != i_largest
        negbit = (q[:, i] < 0.0) ^ negate
        mag = (511.0 * (np.abs(q[:, i]) / _SQRT1_2) + 0.5).astype(np.uint32)
        contrib = (negbit.astype(np.uint32) << np.uint32(9)) | mag
        shifted = (comp << np.uint32(10)) | contrib
        comp = np.where(mask, shifted, comp).astype(np.uint32)
    out = np.empty((n, 4), dtype=np.uint8)
    out[:, 0] = comp & 0xFF
    out[:, 1] = (comp >> 8) & 0xFF
    out[:, 2] = (comp >> 16) & 0xFF
    out[:, 3] = (comp >> 24) & 0xFF
    return out.reshape(-1)


def _unpack_rotations_smallest_three(buf, n):
    b = np.asarray(buf, dtype=np.uint32).reshape(n, 4)
    comp = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)
    c_mask = np.uint32((1 << 9) - 1)
    i_largest = (comp >> np.uint32(30)).astype(np.int64)
    out = np.zeros((n, 4), dtype=np.float64)
    work = comp.copy()
    sum_sq = np.zeros(n, dtype=np.float64)
    rows = np.arange(n)
    for i in range(3, -1, -1):
        mask = i != i_largest
        mag = (work & c_mask).astype(np.float64)
        negbit = (work >> np.uint32(9)) & np.uint32(1)
        work = np.where(mask, work >> np.uint32(10), work).astype(np.uint32)
        val = _SQRT1_2 * mag / float((1 << 9) - 1)
        val = np.where(negbit == 1, -val, val)
        val = np.where(mask, val, 0.0)
        out[:, i] = np.where(mask, val, out[:, i])
        sum_sq += np.where(mask, val * val, 0.0)
    largest_val = np.sqrt(np.clip(1.0 - sum_sq, 0.0, None))
    out[rows, i_largest] = largest_val
    return out


def _quantize_sh(x, bucket_size):
    """``quantizeSH``: 8-bit then round to nearest bucket centre (0 -> a centre)."""
    x = np.asarray(x, dtype=np.float64)
    q = (_cround(x * 128.0) + 128.0).astype(np.int64)
    half = np.asarray(bucket_size, dtype=np.int64) // 2
    bs = np.asarray(bucket_size, dtype=np.int64)
    q = np.trunc((q + half) / bs).astype(np.int64) * bs   # C integer truncation
    return np.clip(q, 0, 255).astype(np.uint8)


def _unquantize_sh(buf):
    return (np.asarray(buf, dtype=np.float64) - 128.0) / 128.0


def _pack_sh(sh_rest, sh_degree, sh1_bits, sh_rest_bits):
    """``sh_rest`` is [N, shDim, 3] (channel fastest). Returns flat [N*shDim*3]."""
    sh_dim = _sh_dim(sh_degree)
    if sh_dim == 0:
        return np.zeros(0, dtype=np.uint8)
    arr = np.asarray(sh_rest, dtype=np.float64).reshape(-1, sh_dim, 3)
    coeff_idx = np.arange(sh_dim)
    bucket = np.where(coeff_idx < 3, 1 << (8 - sh1_bits), 1 << (8 - sh_rest_bits))
    return _quantize_sh(arr, bucket[None, :, None]).reshape(-1)


def _unpack_sh(buf, n, sh_degree):
    sh_dim = _sh_dim(sh_degree)
    if sh_dim == 0:
        return np.zeros((n, 0, 3), dtype=np.float64)
    return _unquantize_sh(buf).reshape(n, sh_dim, 3)


# ---------------------------------------------------------------------------
# ZSTD codec (require a real zstd binding; zlib is NOT acceptable)
# ---------------------------------------------------------------------------
def _zstd():
    """Return (compress(bytes,level)->bytes, decompress(bytes,expected_size)->bytes)."""
    try:
        import zstandard  # type: ignore
    except ImportError:
        zstandard = None
    if zstandard is not None:
        def _c(data, level):
            return zstandard.ZstdCompressor(level=level).compress(data)

        def _d(data, expected):
            return zstandard.ZstdDecompressor().decompress(data, max_output_size=expected)
        return _c, _d
    try:
        import pyzstd  # type: ignore
    except ImportError:
        pyzstd = None
    if pyzstd is not None:
        def _c(data, level):
            return pyzstd.compress(data, level)

        def _d(data, expected):
            return pyzstd.decompress(data)
        return _c, _d
    raise RuntimeError(
        "SPZ v4 requires a real ZSTD codec: install the 'zstandard' (preferred) or "
        "'pyzstd' package. zlib is not an acceptable substitute (the SPZ v4 container "
        "mandates ZSTD-compressed streams)."
    )


# ---------------------------------------------------------------------------
# container serialize / deserialize (low-level GaussianCloud dict)
# ---------------------------------------------------------------------------
def _cloud_stream_order(packed, sh_dim):
    """Streams in reference saveSpz order, skipping zero-length ones."""
    streams = [
        ("positions", packed["positions"]),
        ("alphas", packed["alphas"]),
        ("colors", packed["colors"]),
        ("scales", packed["scales"]),
        ("rotations", packed["rotations"]),
        ("sh", packed["sh"]),
    ]
    return [(name, buf) for name, buf in streams if buf.size > 0]


def serialize_spz(cloud, *, fractional_bits=DEFAULT_FRACTIONAL_BITS,
                  sh1_bits=DEFAULT_SH1_BITS, sh_rest_bits=DEFAULT_SH_REST_BITS,
                  antialiased=False):
    """Serialize a low-level SPZ ``cloud`` dict to v4 ``.spz`` bytes.

    ``cloud`` keys (SPZ-native semantics): ``positions`` [N,3] world coords,
    ``scales`` [N,3] **log**-scale, ``rotations`` [N,4] **xyzw** quaternion,
    ``alphas`` [N] **logit** (pre-sigmoid), ``colors`` [N,3] SH-DC f_dc,
    ``sh`` [N,shDim,3] higher-order SH (or ``None``), ``sh_degree`` int.
    """
    if not (1 <= sh1_bits <= 8 and 1 <= sh_rest_bits <= 8):
        raise ValueError("sh1_bits and sh_rest_bits must be in 1..8")
    positions = np.asarray(cloud["positions"], dtype=np.float64).reshape(-1, 3)
    n = positions.shape[0]
    if n == 0:
        raise ValueError("cannot serialize an empty cloud (numPoints == 0)")
    sh_degree = int(cloud.get("sh_degree", 0) or 0)
    sh_dim = _sh_dim(sh_degree)
    scales_log = np.asarray(cloud["scales"], dtype=np.float64).reshape(-1, 3)
    rotations = np.asarray(cloud["rotations"], dtype=np.float64).reshape(-1, 4)
    alphas = np.asarray(cloud["alphas"], dtype=np.float64).reshape(-1)
    colors = np.asarray(cloud["colors"], dtype=np.float64).reshape(-1, 3)
    for name, arr, expect in (("scales", scales_log, n), ("rotations", rotations, n),
                              ("alphas", alphas, n), ("colors", colors, n)):
        if arr.shape[0] != expect:
            raise ValueError(f"{name} has {arr.shape[0]} rows, expected {expect}")
    sh = cloud.get("sh")
    if sh_dim > 0:
        if sh is None:
            raise ValueError(f"sh_degree={sh_degree} requires an 'sh' array")
        sh = np.asarray(sh, dtype=np.float64).reshape(n, sh_dim, 3)

    packed = {
        "positions": _pack_positions(positions, fractional_bits),
        "alphas": _pack_alphas(alphas),
        "colors": _pack_colors(colors),
        "scales": _pack_scales(scales_log),
        "rotations": _pack_rotations_smallest_three(rotations),
        "sh": _pack_sh(sh, sh_degree, sh1_bits, sh_rest_bits) if sh_dim > 0
        else np.zeros(0, dtype=np.uint8),
    }

    compress, _ = _zstd()
    streams = _cloud_stream_order(packed, sh_dim)
    chunks, toc = [], []
    for _name, buf in streams:
        raw = np.ascontiguousarray(buf, dtype=np.uint8).tobytes()
        comp = compress(raw, _ZSTD_LEVEL)
        chunks.append(comp)
        toc.append((len(comp), len(raw)))

    num_streams = len(streams)
    toc_byte_offset = HEADER_SIZE  # no extensions
    flags = FLAG_ANTIALIASED if antialiased else 0
    header = struct.pack(
        "<IIIBBBBI12s", NGSP_MAGIC, SPZ_VERSION, n, sh_degree & 0xFF,
        fractional_bits & 0xFF, flags & 0xFF, num_streams & 0xFF,
        toc_byte_offset, b"\x00" * 12)
    assert len(header) == HEADER_SIZE
    toc_bytes = b"".join(struct.pack("<QQ", cs, us) for cs, us in toc)
    return header + toc_bytes + b"".join(chunks)


def read_spz_header(source):
    """Parse the 32-byte v4 header without decompressing any stream.

    ``source`` is a path or raw ``bytes``. Returns a dict of header fields.
    Raises ``ValueError`` on a non-v4 / malformed header.
    """
    data = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    if len(data) < HEADER_SIZE:
        raise ValueError(f"SPZ file too short: {len(data)} < {HEADER_SIZE} byte header")
    if data[:2] == b"\x1f\x8b":
        raise ValueError("legacy gzip SPZ (v1-3) is not supported by this v4-only reader")
    (magic, version, num_points, sh_degree, fractional_bits, flags,
     num_streams, toc_byte_offset) = struct.unpack("<IIIBBBBI", data[:20])
    if magic != NGSP_MAGIC:
        raise ValueError(f"not an NGSP file: magic=0x{magic:08x} != 0x{NGSP_MAGIC:08x}")
    if version != SPZ_VERSION:
        raise ValueError(f"unsupported SPZ version {version} (this reader is v4 only)")
    return {
        "magic": magic, "version": version, "numPoints": num_points,
        "shDegree": sh_degree, "fractionalBits": fractional_bits, "flags": flags,
        "numStreams": num_streams, "tocByteOffset": toc_byte_offset,
        "antialiased": bool(flags & FLAG_ANTIALIASED),
        "hasExtensions": bool(flags & FLAG_HAS_EXTENSIONS),
    }


def deserialize_spz(data):
    """Deserialize v4 ``.spz`` bytes into a low-level SPZ ``cloud`` dict.

    Inverse of :func:`serialize_spz`. Returns SPZ-native fields (``positions``,
    ``scales`` log, ``rotations`` xyzw, ``alphas`` logit, ``colors`` f_dc, ``sh``,
    ``sh_degree``, ``antialiased``, ``fractionalBits``, ``numPoints``).
    """
    data = bytes(data)
    header = read_spz_header(data)
    n = header["numPoints"]
    if n <= 0:
        raise ValueError(f"invalid point count: {n}")
    sh_degree = header["shDegree"]
    sh_dim = _sh_dim(sh_degree)
    frac = header["fractionalBits"]
    toc_off = header["tocByteOffset"]
    num_streams = header["numStreams"]
    if toc_off < HEADER_SIZE:
        raise ValueError("tocByteOffset is inside the header")

    # Expected (name, uncompressed byte count) in stream order; drop zero-size.
    expected = [
        ("positions", n * 9), ("alphas", n * 1), ("colors", n * 3),
        ("scales", n * 3), ("rotations", n * 4), ("sh", n * sh_dim * 3),
    ]
    expected = [(nm, sz) for nm, sz in expected if sz > 0]
    if num_streams != len(expected):
        raise ValueError(
            f"numStreams={num_streams} but v4 SPZ at sh_degree={sh_degree} "
            f"expects {len(expected)} streams")

    toc_size = num_streams * 16
    if toc_off + toc_size > len(data):
        raise ValueError("TOC extends past end of file")
    _, decompress = _zstd()

    streams = {}
    comp_offset = toc_off + toc_size
    for i, (name, exp_size) in enumerate(expected):
        e = toc_off + i * 16
        comp_size, uncomp_size = struct.unpack("<QQ", data[e:e + 16])
        if uncomp_size != exp_size:
            raise ValueError(
                f"stream '{name}' uncompressed size {uncomp_size} != expected {exp_size}")
        if comp_offset + comp_size > len(data):
            raise ValueError(f"stream '{name}' extends past end of file")
        raw = decompress(data[comp_offset:comp_offset + comp_size], exp_size)
        if len(raw) != exp_size:
            raise ValueError(
                f"stream '{name}' decompressed to {len(raw)} bytes, expected {exp_size}")
        streams[name] = np.frombuffer(raw, dtype=np.uint8)
        comp_offset += comp_size

    cloud = {
        "numPoints": n,
        "sh_degree": sh_degree,
        "antialiased": header["antialiased"],
        "fractionalBits": frac,
        "positions": _unpack_positions(streams["positions"], n, frac),
        "scales": _unpack_scales(streams["scales"], n),
        "rotations": _unpack_rotations_smallest_three(streams["rotations"], n),
        "alphas": _unpack_alphas(streams["alphas"], n),
        "colors": _unpack_colors(streams["colors"], n),
        "sh": _unpack_sh(streams["sh"], n, sh_degree) if sh_dim > 0 else None,
    }
    return cloud


# ---------------------------------------------------------------------------
# AURA-carrier  <->  SPZ-cloud adapter
# ---------------------------------------------------------------------------
def _np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _quat_wxyz_to_xyzw(q):
    q = _np(q).reshape(-1, 4)
    return np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=1)


def _quat_xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    return np.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], axis=1)


def carriers_to_cloud(carriers):
    """Convert an AURA carrier dict (means/scales-linear/quats-wxyz/opacity[0,1]/
    colors-or-sh) into a low-level SPZ cloud dict (see :func:`serialize_spz`)."""
    means = _np(carriers["means"]).reshape(-1, 3)
    n = means.shape[0]
    scales_lin = np.clip(_np(carriers["scales"]).reshape(-1, 3), 1e-12, None)
    scales_log = np.log(scales_lin)
    rot_xyzw = _quat_wxyz_to_xyzw(carriers["quats"])
    opacity = np.clip(_np(carriers["opacity"]).reshape(-1), 1e-6, 1.0 - 1e-6)
    alphas_logit = _inv_sigmoid(opacity)

    sh_full = carriers.get("sh") if hasattr(carriers, "get") else None
    sh_full = _np(sh_full)
    if sh_full is not None:
        sh_full = sh_full.reshape(n, -1, 3)
        sh_degree = int(carriers.get("sh_degree", 0) or 0)
        if sh_degree == 0:
            sh_degree = {1: 0, 4: 1, 9: 2, 16: 3, 25: 4}.get(sh_full.shape[1], 0)
        colors_fdc = sh_full[:, 0, :]
        sh_rest = sh_full[:, 1:1 + _sh_dim(sh_degree), :] if _sh_dim(sh_degree) else None
    else:
        rgb = np.clip(_np(carriers["colors"]).reshape(-1, 3), 0.0, 1.0)
        colors_fdc = (rgb - 0.5) / _C0
        sh_degree = 0
        sh_rest = None

    return {
        "positions": means, "scales": scales_log, "rotations": rot_xyzw,
        "alphas": alphas_logit, "colors": colors_fdc, "sh": sh_rest,
        "sh_degree": sh_degree,
    }


def cloud_to_carriers(cloud):
    """Convert a low-level SPZ cloud dict back into an AURA carrier dict (numpy,
    float32): means/scales-linear/quats-wxyz/opacity[0,1] + colors or sh."""
    n = cloud["numPoints"]
    sh_degree = cloud["sh_degree"]
    out = {
        "means": cloud["positions"].astype(np.float32),
        "scales": np.exp(cloud["scales"]).astype(np.float32),
        "quats": _quat_xyzw_to_wxyz(cloud["rotations"]).astype(np.float32),
        "opacity": _sigmoid(cloud["alphas"]).astype(np.float32),
        "sh_degree": sh_degree,
    }
    fdc = cloud["colors"]                                  # [N,3] f_dc
    if sh_degree > 0 and cloud["sh"] is not None:
        sh_dim = _sh_dim(sh_degree)
        sh_full = np.empty((n, 1 + sh_dim, 3), dtype=np.float32)
        sh_full[:, 0, :] = fdc
        sh_full[:, 1:, :] = cloud["sh"]
        out["sh"] = sh_full
    else:
        out["colors"] = np.clip(0.5 + _C0 * fdc, 0.0, 1.0).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# confidence sidecar (mirrors the carriers.npz sidecar pattern)
# ---------------------------------------------------------------------------
def confidence_sidecar_path(spz_path):
    """`<name>.spz` -> `<name>.spz.confidence.npz`."""
    p = Path(spz_path)
    return p.with_name(p.name + ".confidence.npz") if p.suffix == ".spz" \
        else p.with_suffix(p.suffix + CONFIDENCE_SIDECAR_SUFFIX)


def write_confidence_sidecar(spz_path, confidence, *, num_points=None, provenance=None):
    """Write ``<name>.spz.confidence.npz`` with a ``confidence`` float32 array
    (aligned 1:1 to SPZ point order) and a JSON-able ``provenance`` dict."""
    conf = np.asarray(_np(confidence), dtype=np.float32).reshape(-1)
    if num_points is not None and conf.shape[0] != num_points:
        raise ValueError(
            f"confidence length {conf.shape[0]} != numPoints {num_points}")
    prov = dict(provenance or {})
    prov.setdefault("format", "aura.spz.confidence.v1")
    prov.setdefault("channel", "confidence")
    prov.setdefault("count", int(conf.shape[0]))
    prov.setdefault("aligned_to", "spz numPoints / point order")
    prov.setdefault("note", "SPZ v4 cannot carry per-splat confidence natively; sidecar.")
    side = confidence_sidecar_path(spz_path)
    np.savez(side, confidence=conf, provenance=np.asarray(json.dumps(prov)))
    return side


def read_confidence_sidecar(spz_path):
    """Read ``(confidence[N] float32, provenance dict)`` or ``(None, None)`` if absent."""
    side = confidence_sidecar_path(spz_path)
    if not side.exists():
        return None, None
    z = np.load(side, allow_pickle=False)
    conf = np.asarray(z["confidence"], dtype=np.float32)
    prov = json.loads(str(z["provenance"])) if "provenance" in z else {}
    return conf, prov


# ---------------------------------------------------------------------------
# public high-level API
# ---------------------------------------------------------------------------
def write_spz(carriers, output_path, *, fractional_bits=DEFAULT_FRACTIONAL_BITS,
              sh1_bits=DEFAULT_SH1_BITS, sh_rest_bits=DEFAULT_SH_REST_BITS,
              antialiased=False, confidence=None, provenance=None,
              write_confidence=True):
    """Write AURA carriers to a v4 ``.spz`` file (+ confidence sidecar if present).

    Confidence is taken from the ``confidence`` kwarg, else ``carriers['confidence']``.
    Because SPZ v4 has no native per-splat confidence channel, it is written to
    ``<name>.spz.confidence.npz`` (see module docstring).
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".spz":
        output_path = output_path.with_suffix(".spz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cloud = carriers_to_cloud(carriers)
    data = serialize_spz(cloud, fractional_bits=fractional_bits, sh1_bits=sh1_bits,
                         sh_rest_bits=sh_rest_bits, antialiased=antialiased)
    output_path.write_bytes(data)

    conf = confidence
    if conf is None and hasattr(carriers, "get"):
        conf = carriers.get("confidence")
    if write_confidence and conf is not None:
        n = cloud["positions"].shape[0]
        prov = dict(provenance or {})
        prov.setdefault("spz_file", output_path.name)
        prov.setdefault("generator", "aura.spz.write_spz")
        write_confidence_sidecar(output_path, conf, num_points=n, provenance=prov)
    return output_path


def read_spz(source, *, load_confidence=True):
    """Read a v4 ``.spz`` file into an AURA carrier dict (numpy float32).

    If a ``<name>.spz.confidence.npz`` sidecar sits next to a file ``source`` and
    ``load_confidence`` is set, its ``confidence`` array is attached (length-checked
    against ``numPoints``).
    """
    if isinstance(source, (bytes, bytearray)):
        cloud = deserialize_spz(source)
        carriers = cloud_to_carriers(cloud)
        return carriers
    path = Path(source)
    cloud = deserialize_spz(path.read_bytes())
    carriers = cloud_to_carriers(cloud)
    if load_confidence:
        conf, prov = read_confidence_sidecar(path)
        if conf is not None:
            if conf.shape[0] != cloud["numPoints"]:
                raise ValueError(
                    f"confidence sidecar length {conf.shape[0]} != numPoints "
                    f"{cloud['numPoints']}")
            carriers["confidence"] = conf
            carriers["confidence_provenance"] = prov
    return carriers
