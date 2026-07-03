"""Aggregate the P2 per-scene artifacts into one summary JSON.

Reads, per scene, the calibration reports and reliability/label summaries produced
by the P2 pipeline and folds them into ``outputs/p2_summary.json`` — the single
artifact ``docs/P2_FULLRES_RENDERLOSS.md`` traces its head-to-head tables to. Pure
file IO, no GPU. Missing scenes/labels are skipped so partial runs still collect.

Per scene it records four conditions:
  * ``fullres_color``   : full-resolution (scale 1.0) colour-agreement label  [P2a]
  * ``fullres_depth``   : full-resolution occlusion-aware label
  * ``quarter_color``   : quarter-resolution (scale 0.25) colour label control [resolution A/B]
  * ``fullres_renderloss``: full-resolution render-loss label                 [P2b]
"""
from __future__ import annotations

import json
from pathlib import Path

SCENES = ["truck", "garden", "kitchen", "room"]
OUT = Path("outputs")

# condition -> (calib json, reliability-summary json)
CONDITIONS = {
    "fullres_color": ("calib_{s}_fr.json", "rel_{s}_fr.json"),
    "fullres_depth": ("calib_{s}_fr_depth.json", "rel_{s}_fr_depth.json"),
    "quarter_color": ("calib_{s}_q.json", "rel_{s}_q.json"),
    "fullres_renderloss": ("calib_{s}_renderloss.json", "rl_{s}_fr.json"),
}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def build_p2_summary(scenes=None) -> dict:
    scenes = scenes or SCENES
    out = {"experiment": "P2_fullres_and_renderloss", "scenes": {}}
    for s in scenes:
        rec = {}
        meta_fr = _load(OUT / f"{s}-fr.aura" / "p2_train_meta.json")
        meta_q = _load(OUT / f"{s}-q.aura" / "p2_train_meta.json")
        if meta_fr:
            rec["train_fullres"] = meta_fr
        if meta_q:
            rec["train_quarter"] = meta_q
        for cond, (calib_pat, rel_pat) in CONDITIONS.items():
            calib = _load(OUT / calib_pat.format(s=s))
            rel = _load(OUT / rel_pat.format(s=s))
            if calib is None and rel is None:
                continue
            entry = {}
            if rel is not None:
                entry["reliability_summary"] = rel
            if calib is not None:
                entry["calibration"] = calib.get("calibration")
                entry["pruning_certificate"] = calib.get("pruning_certificate")
                entry["selection_auc"] = calib.get("selection_auc_retained_reliability")
                entry["carriers_labeled"] = calib.get("carriers_labeled")
                entry["label"] = calib.get("label")
            rec[cond] = entry
        if rec:
            out["scenes"][s] = rec
    return out


def main() -> None:
    summary = build_p2_summary()
    (OUT / "p2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # compact console view
    for s, rec in summary["scenes"].items():
        print(f"\n== {s} ==")
        for cond in CONDITIONS:
            e = rec.get(cond)
            if not e:
                continue
            auc = e.get("selection_auc", {})
            cal = e.get("calibration", {})
            cert = e.get("pruning_certificate", {})
            rs = e.get("reliability_summary", {})
            corr = rs.get("corr_trainagree_reliability")
            print(f"  {cond:20s} corr={corr} "
                  f"ece {cal.get('ece_raw_heuristic')}->{cal.get('ece_calibrated')} "
                  f"AUC cal={auc.get('calibrated_confidence')} "
                  f"orac={auc.get('oracle_ceiling')} opac={auc.get('opacity')} "
                  f"cert kept={cert.get('kept_fraction')}")
    print(f"\n-> {OUT / 'p2_summary.json'}")


if __name__ == "__main__":
    main()
