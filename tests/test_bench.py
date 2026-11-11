"""Der Messprüfstand (bench/edit_bench.py) läuft als Test mit: synthetische Wahrheit, eingefrorene Schwellen."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import edit_bench as eb  # noqa: E402


@pytest.mark.parametrize("typ,kind,form", [("loch", "plate", "rect"), ("loch", "plate", "c"), ("loch", "sphere", "rect"), ("loch", "sphere", "c"),
                                           ("swap", "box", ""), ("swap", "sphere", ""), ("swap", "cylinder", "")])
@pytest.mark.parametrize("scale", [1e-9, 1e-3, 1e3, 1e9])
def test_bench_case_is_inside_the_frozen_thresholds(tmp_path, typ, kind, form, scale):
    m = eb.case_hole(kind, scale, tmp_path, form) if typ == "loch" else eb.case_swap(kind, scale, tmp_path)
    assert eb.judge(typ + "_" + kind, m, eb._thresholds(typ, kind, form)) == [], m


def test_flat_hole_stays_flat_and_curved_hole_follows_the_surface(tmp_path):
    flat = eb.case_hole("plate", 1.0, tmp_path)
    curved = eb.case_hole("sphere", 1.0, tmp_path)
    assert flat["fuell_abweichung"] < 1e-9 and flat["normale_grad"] < 1e-6        # Ebene bleibt exakt eben
    assert curved["fuell_abweichung"] < 0.015                                     # vorher (flacher Deckel) 0,042


def test_flat_hole_uses_the_plain_cap_not_the_rings(tmp_path):
    m = eb.case_hole("plate", 1.0, tmp_path)
    assert m["fuell_dreiecke"] <= 16                                              # Rand mit 16 Punkten: Ohrenschnitt gibt 14, die Ringe gaeben 80
