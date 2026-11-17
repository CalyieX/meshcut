"""Das Urteil von remove/swap/edit (verdict: ok | unsicher | nicht ok), der Knick am Fuellrand und der Ein-Satz-Befehl.

Schwellen: bench/OK_SCHWELLEN.md (eingefroren vor der ersten Messung)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import kkmeshcut as mc
import kkmeshcut_edit as ed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import edit_bench as eb  # noqa: E402


def write_case(tmp_path, kind, form="rect", scale=1.0, name="case.obj"):
    """Platte oder Kugel mit Hoecker (Objekt 'bump', rot) auf blauem Grund, als OBJ mit Material, UV und Normalen."""
    V, F, objs, V0 = (eb.plate if kind == "plate" else eb.sphere)(shape=form)
    mats = ["red" if o == "bump" else "blue" for o in objs]
    p = tmp_path / name
    p.write_bytes(eb.obj_bytes(V, F, objs, mats, scale, (V0.min(axis=0), V0.max(axis=0))))
    (tmp_path / "m.mtl").write_text("newmtl blue\nKd 0 0 1\nnewmtl red\nKd 1 0 0\n")
    return p


def last_line(text):
    return [ln for ln in text.splitlines() if ln.strip()][-1]


# ----------------------------------------------------------------------------- Stufen
@pytest.mark.parametrize("kind", ["plate", "sphere"])
def test_good_hole_gets_verdict_ok_and_the_last_line_says_so(tmp_path, kind, capsys):
    p = write_case(tmp_path, kind)
    assert mc.main(["remove", str(p), "--name", "bump", "--out", str(tmp_path / "o.obj")]) in (0, None)
    out = capsys.readouterr().out
    assert last_line(out) == "verdict: ok"
    assert "Knick am Fuellrand" in out


@pytest.mark.parametrize("shape", ["box", "sphere", "cylinder"])
def test_good_swap_gets_verdict_ok(tmp_path, shape):
    Vb, Fb, _, _ = eb.sphere(block=(0, 0, 0))
    lo, hi = (1.4, -0.3, -0.3), (2.0, 0.3, 0.3)
    Vp, Fp = eb.prim(shape, lo, hi)
    V, F = np.vstack([Vb, Vp]), np.vstack([Fb, Fp + len(Vb)])
    objs = ["base"] * len(Fb) + ["part"] * len(Fp)
    src = tmp_path / "s.obj"
    src.write_bytes(eb.obj_bytes(V, F, objs, ["blue"] * len(Fb) + ["green"] * len(Fp), 1.0))
    (tmp_path / "m.mtl").write_text("newmtl blue\nKd 0 0 1\nnewmtl green\nKd 0 1 0\n")
    r = mc.swap(src, name="part", out=tmp_path / "o.obj")
    assert r["verdict"] == "ok" and r["reason"] is None


def test_flat_fill_on_curved_surface_is_unsure_with_a_plain_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "FAIR_MIN_RIM", 10 ** 9)                          # der alte flache Fueller (Loecher unter FAIR_MIN_RIM bleiben bei ihm)
    p = write_case(tmp_path, "sphere", "c")
    r = mc.remove(p, name="bump", out=tmp_path / "o.obj")
    assert r["verdict"] == "unsicher"
    assert "flach" in r["reason"] and "gewoelbt" in r["reason"]
    assert last_line(r["text"]).startswith("verdict: unsicher - ")


def test_removing_most_of_the_model_is_not_ok_exit_code_2_and_the_file_is_still_written(tmp_path, capsys):
    p = write_case(tmp_path, "sphere")
    out = tmp_path / "o.obj"
    with pytest.raises(SystemExit) as e:
        mc.main(["edit", str(p), "entferne das groesste Teil", "--out", str(out)])
    assert e.value.code == 2
    assert out.exists()                                                       # ausgeschrieben, aber der Aufrufer merkt es
    text = capsys.readouterr().out
    assert last_line(text).startswith("verdict: nicht ok - ")
    assert "knickt" in last_line(text)


def test_api_returns_not_ok_as_data_and_does_not_raise(tmp_path):
    p = write_case(tmp_path, "sphere")
    r = mc.edit(p, "entferne das groesste Teil", out=tmp_path / "o.obj")
    assert r["verdict"] == "nicht ok" and "knickt" in r["reason"] and Path(r["output"]).exists()


def test_json_output_is_one_object_with_verdict_and_numbers(tmp_path, capsys):
    p = write_case(tmp_path, "plate")
    assert mc.main(["remove", str(p), "--name", "bump", "--json", "--out", str(tmp_path / "o.obj")]) in (0, None)
    js = json.loads(capsys.readouterr().out)
    assert js["verdict"] == "ok" and js["grund"] == [] and js["knick_mittel_grad"] < 1e-6 and js["offene_kanten_nachher"] == js["offene_kanten_vorher"]
    r = mc.remove(p, name="bump", json=True, out=tmp_path / "o2.obj")
    assert r["verdict"] == "ok" and Path(r["output"]).name == "o2.obj"


def test_json_with_not_ok_exits_2_too(tmp_path, capsys):
    p = write_case(tmp_path, "sphere")
    with pytest.raises(SystemExit) as e:
        mc.main(["edit", str(p), "remove the largest part", "--json", "--out", str(tmp_path / "o.obj")])
    assert e.value.code == 2
    assert json.loads(capsys.readouterr().out)["verdict"] == "nicht ok"


# ----------------------------------------------------------------------------- Knick am Fuellrand (Schritt 2)
def two_triangles(angle_deg):
    """Behaltenes Dreieck in der xy-Ebene, daran ein Fuelldreieck, um angle_deg aus der Ebene geklappt (gleiche Windung)."""
    a = np.radians(angle_deg)
    sc = ed.Scene()
    sc.V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, -np.cos(a), -np.sin(a)]], float)
    sc.faces = [ed.Face([(0, -1, -1), (1, -1, -1), (2, -1, -1)]), ed.Face([(1, -1, -1), (0, -1, -1), (3, -1, -1)])]
    return sc


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 150.0])
def test_fill_jump_measures_the_known_angle(angle):
    mean, mx, n = ed._fill_jump(two_triangles(angle), 1, 1)
    assert n == 1 and mean == pytest.approx(angle, abs=1e-6) and mx == pytest.approx(angle, abs=1e-6)


def test_fill_jump_is_none_without_a_kept_neighbour():
    sc = two_triangles(10.0)
    assert ed._fill_jump(sc, 0, 2) is None and ed._fill_jump(sc, 2, 0) is None


@pytest.mark.parametrize("jump,level", [(10.0, "ok"), (ed.JUMP_UNSURE + 1, "unsicher"), (ed.JUMP_BAD + 1, "nicht ok")])
def test_the_frozen_jump_thresholds_decide_the_level(jump, level):
    rep = dict(offene_kanten_vorher=0, offene_kanten_nachher=0, loecher_geschlossen=1, knick=(jump, jump, 4))
    assert ed._verdict(ed.Scene(), rep, "obj")[0] == level


def test_the_jump_is_in_the_report_of_the_real_result(tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "FAIR_MIN_RIM", 10 ** 9)
    p = write_case(tmp_path, "sphere", "c")
    text = mc.remove(p, name="bump", out=tmp_path / "o.obj")["text"]
    line = next(ln for ln in text.splitlines() if ln.startswith("Knick am Fuellrand"))
    assert "im Mittel 3" in line                                              # die flache Fuellung der C-Form knickt etwa 31 Grad ab


# ----------------------------------------------------------------------------- Grenzenliste: je Grenze ein Test (Schritt 6)
def rep(**kw):
    base = dict(offene_kanten_vorher=0, offene_kanten_nachher=0, loecher_geschlossen=1, knick=None)
    base.update(kw)
    return base


def test_limit_new_open_edges_is_not_ok():
    level, why = ed._verdict(ed.Scene(), rep(offene_kanten_nachher=6), "obj")
    assert level == "nicht ok" and "6 offene Kanten" in why[0]


def test_limit_volume_change_over_twenty_percent_is_unsure():
    assert ed._verdict(ed.Scene(), rep(volumen_aenderung=0.1), "obj")[0] == "ok"
    level, why = ed._verdict(ed.Scene(), rep(volumen_aenderung=-0.35), "obj")
    assert level == "unsicher" and "-35 %" in why[0]


def test_limit_very_large_hole_is_unsure_end_to_end(tmp_path):
    p = write_case(tmp_path, "plate", "c")                                     # die C-Form nimmt 43 % der Platte weg
    r = mc.remove(p, name="bump", out=tmp_path / "o.obj")
    assert r["verdict"] == "unsicher" and "gross" in r["reason"] and "erfunden" in r["reason"]


def test_limit_several_separate_holes_is_unsure_end_to_end(tmp_path):
    n = 12
    idx = {(i, j): j * (n + 1) + i for j in range(n + 1) for i in range(n + 1)}
    V = np.array([(i / n, j / n, 0.0) for j in range(n + 1) for i in range(n + 1)])
    F, objs = [], []
    blocks = [{(2, 2), (3, 2), (2, 3), (3, 3)}, {(8, 8), (9, 8), (8, 9), (9, 9)}]
    for j in range(n):
        for i in range(n):
            bump = any((i, j) in b for b in blocks)
            q = [idx[(i, j)], idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]]
            for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                F.append(t)
                objs.append("bump" if bump else "base")
    for b in blocks:                                                          # beide Hoecker anheben, der Rand bleibt auf der Platte
        for (i, j) in b:
            for di, dj in ((1, 1),):
                if all((i + a, j + c) in b for a in (-1, 0) for c in (-1, 0)):
                    V[idx[(i, j)], 2] = 0.2
    p = tmp_path / "two.obj"
    p.write_bytes(eb.obj_bytes(V, np.array(F), objs, ["m"] * len(F), 1.0))
    (tmp_path / "m.mtl").write_text("newmtl m\nKd 1 1 1\n")
    r = mc.remove(p, name="bump", out=tmp_path / "o.obj")
    assert r["verdict"] == "unsicher" and "getrennte Loecher" in r["reason"]


def test_limit_material_lost_outside_obj_is_unsure_end_to_end(tmp_path):
    p = write_case(tmp_path, "plate")
    r = mc.remove(p, name="bump", out=tmp_path / "o.stl")
    assert r["verdict"] == "unsicher" and "gehen in diesem Dateiformat verloren" in r["reason"]


def test_limit_material_typo_is_unsure(tmp_path):
    p = write_case(tmp_path, "plate")
    r = mc.swap(p, name="bump", material="rott", out=tmp_path / "o.obj")
    assert r["verdict"] == "unsicher" and "Material kommt in der Datei sonst nicht vor" in r["reason"]


def test_limit_no_basic_shape_fits_is_unsure():
    level, why = ed._verdict(ed.Scene(), rep(keine_form=True), "obj")
    assert level == "unsicher" and "Grundformen" in why[0]


def test_limit_non_manifold_mesh_is_unsure_end_to_end(tmp_path):
    n = 12
    V = [(i / n, j / n, 0.0) for j in range(n + 1) for i in range(n + 1)]
    V.append((0.2, 0.2, 0.5))                                                 # Flosse: ein dritter Dreiecksfluegel an einer inneren Kante
    top = len(V) - 1
    idx = lambda i, j: j * (n + 1) + i                                        # noqa: E731
    F, objs = [], []
    for j in range(n):
        for i in range(n):
            bump = 4 <= i < 7 and 4 <= j < 7
            q = [idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
            for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                F.append(t)
                objs.append("bump" if bump else "base")
    F.append((idx(2, 2), idx(3, 2), top))
    objs.append("fin")
    p = tmp_path / "fin.obj"
    p.write_bytes(eb.obj_bytes(np.array(V), np.array(F), objs, ["m"] * len(F), 1.0))
    (tmp_path / "m.mtl").write_text("newmtl m\nKd 1 1 1\n")
    r = mc.remove(p, name="bump", out=tmp_path / "o.obj")
    assert r["verdict"] == "unsicher" and "mehr als zwei Flaechen" in r["reason"]


def test_limit_flat_fill_on_curved_surface_even_without_a_jump_is_unsure():
    level, why = ed._verdict(ed.Scene(), rep(bent=2, ruhiges_volumen=False), "obj")
    assert level == "unsicher" and "flach" in why[0]
    assert ed._verdict(ed.Scene(), rep(bent=2, ruhiges_volumen=True), "obj")[0] == "ok"     # bleibt das Volumen ruhig, ist die Fuellung gut genug


def test_the_most_severe_level_wins_and_reasons_come_in_order():
    level, why = ed._verdict(ed.Scene(), rep(offene_kanten_nachher=2, volumen_aenderung=0.5), "obj")
    assert level == "nicht ok" and "offene Kanten" in why[0] and "Volumen" in why[1]


# ----------------------------------------------------------------------------- Pruefstand: Fehlalarme und Treffer (Kriterium Schritt 1)
def test_good_cases_do_not_cry_wolf_and_bad_cases_are_reported(tmp_path):
    good = [("plate", "rect"), ("sphere", "rect")]
    for kind, form in good:
        sub = tmp_path / ("g_" + kind)
        sub.mkdir()
        p = write_case(sub, kind, form)
        assert mc.remove(p, name="bump", out=sub / "o.obj")["verdict"] == "ok", (kind, form)
    sub = tmp_path / "bad"
    sub.mkdir()
    p = write_case(sub, "sphere", "c")
    assert mc.remove(p, name="bump", out=sub / "o.obj")["verdict"] != "ok"


# ----------------------------------------------------------------------------- Ein Satz Absicht (Schritte 3 und 4)
def test_understand_reads_remove_and_swap_sentences():
    assert ed.understand("entferne den Hoecker") == ("remove", "hoecker", None)
    assert ed.understand("Loesche das rote Teil") == ("remove", "rote", None)
    assert ed.understand("ersetze das groesste Teil durch eine Kugel") == ("swap", "groesste", "sphere")
    assert ed.understand("entferne Teil 7") == ("remove", "teil 7", None)
    assert ed.understand("replace the bump with a cylinder") == ("swap", "bump", "cylinder")
    assert ed.understand("Hoecker durch Quader") == ("swap", "hoecker", "box")
    with pytest.raises(mc.MeshcutError, match="Nicht verstanden"):
        ed.understand("mach es huebsch")
    with pytest.raises(mc.MeshcutError, match="Satz leer"):
        ed.understand("   ")


def test_one_sentence_gives_the_same_bytes_as_the_explicit_options(tmp_path):
    p = write_case(tmp_path, "sphere")
    a, b = tmp_path / "a.obj", tmp_path / "b.obj"
    mc.remove(p, name="bump", out=a)
    mc.edit(p, "entferne den bump", out=b)
    assert a.read_bytes() == b.read_bytes()
    mc.swap(p, name="bump", shape="sphere", out=a)
    mc.edit(p, "ersetze den bump durch eine Kugel", out=b)
    assert a.read_bytes() == b.read_bytes()


def test_the_sentence_is_answered_with_what_was_understood(tmp_path):
    p = write_case(tmp_path, "plate")
    text = mc.edit(p, "ersetze den bump durch einen Zylinder", out=tmp_path / "o.obj")["text"]
    assert "Verstanden: ersetzen durch Zylinder" in text and "Zylinder" in text


@pytest.mark.parametrize("phrase,name", [("der bump", "bump"), ("das rote Teil", "bump"), ("der Hoecker", "bump"), ("Teil 2", None), ("#1", None)])
def test_describe_region_by_name_colour_german_word_or_number(tmp_path, phrase, name):
    p = write_case(tmp_path, "plate")
    sc, regs = ed.read_scene(p), None
    regs = ed.find_regions(sc)
    r = ed.describe_region(regs, phrase)
    if name:
        assert r["name"] == name or any(a["name"] == name for a in r["also"]) or name in r["materials"] or r["name"] in ("red",)
    else:
        assert r["id"] == int(phrase.lstrip("#").split()[-1])


def test_describe_region_biggest_and_smallest(tmp_path):
    p = write_case(tmp_path, "plate")
    regs = ed.find_regions(ed.read_scene(p))
    big, small = ed.describe_region(regs, "das groesste Teil"), ed.describe_region(regs, "das kleinste Teil")
    assert big["triangles"] == max(r["triangles"] for r in regs) and small["triangles"] == min(r["triangles"] for r in regs)


def test_describe_region_failures_name_what_exists(tmp_path):
    p = write_case(tmp_path, "plate")
    regs = ed.find_regions(ed.read_scene(p))
    with pytest.raises(mc.MeshcutError, match="Nichts passt zu .*vorhanden: .*bump"):
        ed.describe_region(regs, "das gruene Haus")
    with pytest.raises(mc.MeshcutError, match="Es gibt keine Region 99"):
        ed.describe_region(regs, "Teil 99")
    with pytest.raises(mc.MeshcutError, match="Beschreibung leer"):
        ed.describe_region(regs, "das")


def test_was_is_one_more_way_to_choose_and_only_one_may_be_given(tmp_path):
    p = write_case(tmp_path, "plate")
    r = mc.remove(p, was="der bump", out=tmp_path / "o.obj")
    assert r["verdict"] == "ok"
    with pytest.raises(mc.MeshcutError, match="Genau eine Auswahl"):
        mc.remove(p, was="bump", name="bump", out=tmp_path / "o2.obj")


def test_serve_edit_tool_returns_verdict(tmp_path):
    import kkmeshcut_serve as srv
    p = write_case(tmp_path, "plate")
    tools = srv.Tools(srv.Sandbox(tmp_path))
    res = srv.call_tool(tools, "edit", {"input": p.name, "satz": "entferne den bump", "out": "o.obj"})
    assert res["verdict"] == "ok" and "verdict: ok" in res["report"] and res["output"] == "o.obj"
    with pytest.raises(mc.MeshcutError, match="satz fehlt"):
        srv.call_tool(tools, "edit", {"input": p.name})
    assert "edit" in {s["name"] for s in srv.tool_specs()}


# ----------------------------------------------------------------------------- Glatte Fuellung fuer grosse, nicht ebene Loecher
def test_min_area_triangulation_avoids_a_chord_that_already_exists_in_the_mesh():
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    n = np.array([0, 0, 1.0])
    assert sorted(tuple(sorted(t)) for t in ed._min_area_triangulation(P, n)) in ([(0, 1, 2), (0, 2, 3)], [(0, 1, 3), (1, 2, 3)])
    bad = np.zeros((4, 4), dtype=bool)
    bad[0, 2] = True
    assert sorted(tuple(sorted(t)) for t in ed._min_area_triangulation(P, n, bad)) == [(0, 1, 3), (1, 2, 3)]
    bad[1, 3] = True
    assert ed._min_area_triangulation(P, n, bad) is None                      # beide Diagonalen verboten: keine Fuellung


def test_refine_keeps_the_rim_edges_and_makes_a_closed_disc():
    m = 14
    ang = np.linspace(0, 2 * np.pi, m, endpoint=False)
    P = np.stack([np.cos(ang), np.sin(ang), 0.2 * np.cos(2 * ang)], axis=1)   # leicht gewellter Kreis
    n = np.array([0, 0, 1.0])
    X, T = ed._refine(P, ed._min_area_triangulation(P, n), n, 400)
    assert np.allclose(X[:m], P) and len(X) > m and len(T) > m - 2          # Randpunkte unveraendert, Innenpunkte dazu
    cnt = {}
    for a, b, c in T:
        for u, v in ((a, b), (b, c), (c, a)):
            cnt[(min(u, v), max(u, v))] = cnt.get((min(u, v), max(u, v)), 0) + 1
    rim = {(min(k, (k + 1) % m), max(k, (k + 1) % m)) for k in range(m)}
    assert all(cnt[e] == 1 for e in rim)                                      # jede Randkante genau einmal: die Nachbarflaechen bleiben dicht
    assert all(v == 2 for e, v in cnt.items() if e not in rim)                # alle inneren Kanten genau zweimal: eine Scheibe ohne Loch
    assert all(float(np.cross(X[b] - X[a], X[c] - X[a]) @ n) > 0 for a, b, c in T)       # alle gleich herum


def test_big_hole_on_a_curved_surface_is_closed_smoothly_and_stays_watertight(tmp_path):
    p = write_case(tmp_path, "sphere", "c")
    r = mc.remove(p, name="bump", out=tmp_path / "o.obj")
    out = ed.read_scene(tmp_path / "o.obj")
    assert all(v == 2 for v in ed.edge_counts(out, ed.position_ids(out)).values())          # geschlossene Kugel bleibt geschlossen
    assert "nicht ebenem Rand" not in r["text"] and "gewoelbten Umgebung" not in r["text"]  # kein flacher Deckel mehr
    fill = out.faces[-60:]
    assert all(f.c[0][1] >= 0 and f.c[0][2] >= 0 for f in out.faces[len(out.faces) - 20:])  # UV und Normalen der neuen Flaechen sind geschrieben


def test_without_the_smooth_filler_the_flat_path_still_works_and_is_unsure(tmp_path, monkeypatch):
    p = write_case(tmp_path, "sphere", "c")
    monkeypatch.setattr(ed, "FAIR_MAX_RIM", 3)                                # Loch zu gross fuer die glatte Fuellung -> alter Weg
    r = mc.remove(p, name="bump", out=tmp_path / "o.obj")
    assert r["verdict"] == "unsicher" and "flach" in r["reason"]


@pytest.mark.parametrize("share,bend,level", [(0.01, 0.2, "ok"), (0.03, 0.01, "ok"), (0.03, 0.2, "unsicher"), (0.02, 0.2, "unsicher")])
def test_large_hole_in_a_curved_surrounding_is_unsure_from_two_percent(share, bend, level):
    rp = dict(offene_kanten_vorher=0, offene_kanten_nachher=0, loecher_geschlossen=1, knick=(3.0, 5.0, 10), flaechenanteil=share, woelbung=bend)
    got, why = ed._verdict(ed.Scene(), rp, "obj")
    assert got == level and (level == "ok" or "geschaetzt" in why[0])


def test_fill_jump_reports_mean_and_maximum_over_several_edges():
    """Ein Fuelldreieck (z = 0) mit zwei behaltenen Nachbarn: einer eben (0 Grad), einer um einen bekannten Winkel aufgeklappt."""
    sc = ed.Scene()
    sc.V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, -1, 0], [1.0, 1.0, 1.0]], float)
    sc.faces = [ed.Face([(1, -1, -1), (0, -1, -1), (3, -1, -1)]),         # flach unter der Kante 0-1
                ed.Face([(2, -1, -1), (1, -1, -1), (4, -1, -1)]),         # aufgeklappt an der Kante 1-2
                ed.Face([(0, -1, -1), (1, -1, -1), (2, -1, -1)])]         # die Fuellung
    n1 = np.cross(sc.V[1] - sc.V[2], sc.V[4] - sc.V[2])
    want = float(np.degrees(np.arccos(np.clip(n1[2] / np.linalg.norm(n1), -1, 1))))
    mean, mx, n = ed._fill_jump(sc, 2, 1)
    assert n == 2 and mx == pytest.approx(want, abs=1e-6) and mean == pytest.approx(want / 2, abs=1e-6) and 5 < want < 175


def test_min_area_triangulation_prefers_triangles_that_face_the_mean_normal():
    """Windschiefer Rand (Sattel): ohne die Strafe fuer Dreiecke gegen die mittlere Normale gewaenne eine Zerlegung mit einem umgeklappten Dreieck."""
    P = np.array([[-0.3363, 0.6121, -0.3511], [-0.3547, 0.5344, 0.4198], [-0.8708, 0.9383, -0.081], [-0.4279, 0.4307, 0.153],
                  [-0.9291, -1.0244, -0.4533], [-0.5775, -1.1217, -0.3764], [-0.5235, -1.111, -0.0033], [0.6365, -0.0503, 0.3114]])
    cen = P.mean(axis=0)
    n = np.cross(P - cen, np.roll(P, -1, axis=0) - cen).sum(axis=0)
    n /= np.linalg.norm(n)
    T = ed._min_area_triangulation(P, n)
    assert len(T) == len(P) - 2 and all(float(np.cross(P[b] - P[a], P[c] - P[a]) @ n) > 0 for a, b, c in T)


def test_flipped_sees_a_turned_over_or_collapsed_triangle():
    X0 = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert not ed._flipped([(0, 1, 2)], X0, X0 + 0.1)
    assert ed._flipped([(0, 1, 2)], X0, X0[[0, 2, 1]])                         # umgeklappt
    assert ed._flipped([(0, 1, 2)], X0, np.zeros((3, 3)))                      # zusammengefallen


def _sphere_fill_radius(tmp_path, flipped):
    p = write_case(tmp_path, "sphere", "c", name="f%d.obj" % flipped)
    out_path = tmp_path / ("o%d.obj" % flipped)
    mc.remove(p, name="bump", out=out_path)
    out = ed.read_scene(out_path)
    T, _ = out.triangles()
    used = np.unique(T)
    return out.V[used]


def _sphere_deviation(P):
    A = np.hstack([2 * P, np.ones((len(P), 1))])
    sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(axis=1), rcond=None)
    c = sol[:3]
    R = float(np.sqrt(sol[3] + c @ c))
    return float(np.abs(np.linalg.norm(P - c, axis=1) - R).max() / R)


def test_smoothing_is_kept_unless_triangles_flip_and_stays_on_the_sphere(tmp_path, monkeypatch):
    smooth = _sphere_fill_radius(tmp_path, 0)
    assert _sphere_deviation(smooth) < 0.008                                   # glatt an Rand und Nachbarring angeschlossen: Kugel bis auf 0,8 %
    monkeypatch.setattr(ed, "_flipped", lambda *a: True)                       # Umklappen erzwungen: es bleibt bei der ungeglaetteten Fuellung
    raw = _sphere_fill_radius(tmp_path, 1)
    assert raw.shape != smooth.shape or float(np.abs(raw - smooth).max()) > 1e-6
