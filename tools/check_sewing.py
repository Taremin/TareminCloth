#!/usr/bin/env python3
"""縫合辺長の整合検査: python tools/check_sewing.py tests/fixtures/garments/tshirt/tshirt_scene.json"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python"))

from taremin_cloth.utils.fixture_io import audit_seams, load_scene_manifest


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_sewing.py <scene.json> [--tolerance-mm 5]")
        return 2
    tol = 0.005
    if "--tolerance-mm" in sys.argv:
        tol = float(sys.argv[sys.argv.index("--tolerance-mm") + 1]) / 1000.0
    m = load_scene_manifest(sys.argv[1])
    rep = audit_seams(m["positions"], m["faces"], m["sewing_springs"])
    print(f"pairs={rep['num_pairs']} chains={rep['num_chains']} "
          f"gap_max={rep['gap_max_m']*1000:.1f}mm gap_mean={rep['gap_mean_m']*1000:.1f}mm "
          f"orphans={len(rep['orphan_pairs'])}")
    bad = 0
    for i, c in enumerate(rep["chains"]):
        ok = "OK " if c["diff_m"] <= tol else "NG "
        if c["diff_m"] > tol:
            bad += 1
        print(f"  [{ok}] chain{i}: n={c['length']} A={c['side_a_m']*1000:.1f}mm "
              f"B={c['side_b_m']*1000:.1f}mm diff={c['diff_m']*1000:.1f}mm")
    # 共有頂点ペア (袖キャップの最近傍割付で意図的に発生) は警告に留める
    if rep["orphan_pairs"]:
        print(f"  [WARN] shared-vertex pairs={len(rep['orphan_pairs'])} (cap nearest-match by design)")
    print("NG" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
