"""regions / remove / swap / tools: Elemente waehlen, entfernen, ersetzen - mit Material, UV und Normalen, und die Pruefung am Ende."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

import kkmeshcut as mc
import kkmeshcut_edit as ed
import shapes


def cube_lines(lo, hi, v0, vt0, vn0, mats, obj, with_vt=True, with_vn=True):
    """Quader als sechs Vierecke mit UV und Normalen; mats: Material je Seite (unten, oben, vorn, hinten, rechts, links)."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    V = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    quads = [((0, 3, 2, 1), (0, 0, -1)), ((4, 5, 6, 7), (0, 0, 1)), ((0, 1, 5, 4), (0, -1, 0)), ((2, 3, 7, 6), (0, 1, 0)),
             ((1, 2, 6, 5), (1, 0, 0)), ((3, 0, 4, 7), (-1, 0, 0))]
    head = ["v %g %g %g" % v for v in V]
    head += ["vn %g %g %g" % n for _, n in quads] if with_vn else []
    body = ["o " + obj]
    for i, (q, _) in enumerate(quads):
        body.append("usemtl " + mats[i])
        if with_vt:
            head += ["vt %g %g" % u for u in ((0, 0), (1, 0), (1, 1), (0, 1))]
        toks = []
        for j, a in enumerate(q):
            t = "%d" % (v0 + a + 1)
            if with_vt:
                t += "/%d" % (vt0 + 4 * i + j + 1)
            if with_vn:
                t += ("/" if with_vt else "//") + "%d" % (vn0 + i + 1)
            toks.append(t)
        body.append("f " + " ".join(toks))
    return head, body


def scene_obj(tmp_path, with_vt=True, with_vn=True, name="scene.obj"):
    """Links ein 2er-Quader (Oberseite rot, Rest blau), rechts ein freistehender gruener 1er-Quader."""
    h1, b1 = cube_lines((0, 0, 0), (2, 2, 2), 0, 0, 0, ["blue", "red", "blue", "blue", "blue", "blue"], "part1", with_vt, with_vn)
    h2, b2 = cube_lines((5, 0, 0), (6, 1, 1), 8, 24, 6, ["green"] * 6, "part2", with_vt, with_vn)
    p = tmp_path / name
    p.write_text("\n".join(["mtllib m.mtl"] + h1 + h2 + b1 + b2) + "\n")
    (tmp_path / "m.mtl").write_text("newmtl blue\nKd 0 0 1\nnewmtl red\nKd 1 0 0\nnewmtl green\nKd 0 1 0\n")
    return p


def regions_of(p):
    sc = ed.read_scene(p)
    return sc, ed.find_regions(sc)


def by_name(regs, name):
    return next(r for r in regs if r["name"] == name)


def mesh_volume(sc, face_ids):
    T, _ = sc.triangles(face_ids)
    A, B, C = sc.V[T[:, 0]], sc.V[T[:, 1]], sc.V[T[:, 2]]
    return float(np.einsum("ij,ij->i", A, np.cross(B, C)).sum()) / 6


# ----------------------------------------------------------------------------- Lesen und Schreiben
def test_obj_scene_roundtrip_keeps_every_attribute(tmp_path):
    p = scene_obj(tmp_path)
    sc = ed.read_scene(p)
    back = ed.parse_obj(ed.obj_text(sc))
    assert [sc.signature(f) for f in sc.faces] == [back.signature(f) for f in back.faces]
    assert back.mtllib == ["m.mtl"] and len(back.faces) == 12


def test_negative_indices_and_mixed_corner_forms(tmp_path):
    p = tmp_path / "n.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvn 0 0 1\nf -4/-3/-1 -3/-2/-1 -2/-1/-1 -1//1\nf 1 2 3\n")
    sc = ed.read_scene(p)
    assert sc.faces[0].c == [(0, 0, 0), (1, 1, 0), (2, 2, 0), (3, -1, 0)] and sc.faces[1].c == [(0, -1, -1), (1, -1, -1), (2, -1, -1)]


@pytest.mark.parametrize("text,match", [("v 0 0 0\nf 1 2 3\n", "nicht gibt"), ("v 0 0 0\nv 1 0 0\nf 1 2\n", "weniger als drei"), ("v 0 0\n", "unlesbare"),
                                        ("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/x 2 3\n", "unlesbare"), ("v 0 0 0\n", "keine Flaechen"),
                                        ("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 2 3\n", "ohne Eckpunkt")])
def test_obj_scene_rejects_broken_files(tmp_path, text, match):
    p = tmp_path / "b.obj"
    p.write_text(text)
    with pytest.raises(mc.MeshcutError, match=match):
        ed.read_scene(p)


def test_unused_vertices_uv_and_normals_are_dropped_in_the_output(tmp_path):
    p = scene_obj(tmp_path)
    sc = ed.read_scene(p)
    keep = [i for i, f in enumerate(sc.faces) if f.obj == "part2"]
    sc2 = sc.copy()
    sc2.faces = [sc.faces[i] for i in keep]
    txt = ed.obj_text(sc2).decode().splitlines()
    assert sum(1 for t in txt if t.startswith("v ")) == 8 and sum(1 for t in txt if t.startswith("vt ")) == 24 and sum(1 for t in txt if t.startswith("vn ")) == 6


# ----------------------------------------------------------------------------- regions
def test_regions_number_parts_and_materials_without_duplicates(tmp_path):
    sc, regs = regions_of(scene_obj(tmp_path))
    assert [(r["id"], r["kind"], r["name"], r["triangles"]) for r in regs] == [
        (1, "teil", "Teil 1", 12), (2, "teil", "Teil 2", 12), (3, "material", "blue", 10), (4, "material", "red", 2)]
    # Objekt part1/part2 und Material green waeren dieselben Flaechen wie Teil 1/2: kommen nicht doppelt vor
    assert not any(r["name"] in ("part1", "part2", "green") for r in regs)


def test_region_statistics(tmp_path):
    sc, regs = regions_of(scene_obj(tmp_path))
    t1, t2 = regs[0], regs[1]
    assert t1["closed"] and abs(t1["volume"] - 8) < 1e-12 and abs(t1["area"] - 24) < 1e-12 and t1["bbox"] == {"min": [0, 0, 0], "max": [2, 2, 2]}
    assert t2["closed"] and abs(t2["volume"] - 1) < 1e-12 and t2["materials"] == ["green"] and t2["center"] == [5.5, 0.5, 0.5]
    assert regs[3]["closed"] is False and regs[3]["materials"] == ["red"]


def test_uv_and_normal_seams_do_not_split_a_part(tmp_path):
    """Jede Seite hat eigene UV und Normalen (Naehte), das Teil bleibt trotzdem eins."""
    sc, regs = regions_of(scene_obj(tmp_path))
    assert sum(1 for r in regs if r["kind"] == "teil") == 2


def test_single_piece_mesh_has_nothing_to_choose(tmp_path):
    p = tmp_path / "c.stl"
    p.write_bytes(mc.mesh_bytes("stl", *(lambda m: (m.V, m.F))(shapes.cube())))
    sc, regs = regions_of(p)
    assert regs == []
    with pytest.raises(mc.MeshcutError, match="nichts auszuwaehlen"):
        ed.choose_region(sc, regs, region=1)


def test_regions_command_writes_png_and_json(tmp_path, capsys):
    p = scene_obj(tmp_path)
    js = mc.regions(p, tmp_path / "r")
    png = (tmp_path / "r" / "regions.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and png[16:24] == (720).to_bytes(4, "big") + (720).to_bytes(4, "big")
    assert js["triangles"] == 24 and js["has_uv"] and js["has_normals"] and [r["id"] for r in js["regions"]] == [1, 2, 3, 4]
    assert all("face_ids" not in r and r["visible"] in (True, False) for r in js["regions"])
    assert json.loads((tmp_path / "r" / "regions.json").read_text())["regions"][0]["name"] == "Teil 1"


def test_overview_labels_are_on_visible_pixels(tmp_path):
    sc, regs = regions_of(scene_obj(tmp_path))
    png, visible = ed.render_overview(sc, regs, 360)
    assert visible[2] and visible[4] and visible[3]                # Teil 2 und die sichtbaren Materialien
    img_before = ed.render_overview(sc, [], 360)[0]
    assert png != img_before                                        # mit Nummern anders als ohne


def test_png_bytes_is_a_valid_image():
    import zlib
    img = np.zeros((3, 4, 3), dtype=np.uint8)
    img[1, 2] = (9, 8, 7)
    b = ed.png_bytes(img)
    idat = b[b.index(b"IDAT") + 4:b.index(b"IEND") - 8]
    raw = zlib.decompress(idat)
    assert len(raw) == 3 * (1 + 12) and raw[13 + 1 + 6:13 + 1 + 9] == bytes((9, 8, 7))


def test_at_point_of_a_region_selects_that_region(tmp_path):
    sc, regs = regions_of(scene_obj(tmp_path))
    for r in regs:
        if r["at_selects_this"]:
            assert ed.choose_region(sc, regs, at=",".join(repr(x) for x in r["point"]))["id"] == r["id"]
    assert [r["at_selects_this"] for r in regs] == [False, True, True, True]       # Teil 1 enthaelt nur groessere Elemente (blue, red)


# ----------------------------------------------------------------------------- Auswahl
def test_choose_exactly_one_selector(tmp_path):
    sc, regs = regions_of(scene_obj(tmp_path))
    for kw in ({}, {"region": 1, "name": "red"}):
        with pytest.raises(mc.MeshcutError, match="Genau eine Auswahl"):
            ed.choose_region(sc, regs, **kw)


@pytest.mark.parametrize("region,match", [(0, "keine Region"), (99, "keine Region"), ("x", "keine ganze Zahl")])
def test_choose_region_number_is_checked(tmp_path, region, match):
    sc, regs = regions_of(scene_obj(tmp_path))
    with pytest.raises(mc.MeshcutError, match=match):
        ed.choose_region(sc, regs, region=region)


def test_choose_by_name_is_case_insensitive_and_checked(tmp_path):
    sc, regs = regions_of(scene_obj(tmp_path))
    assert ed.choose_region(sc, regs, name="RED")["id"] == 4
    with pytest.raises(mc.MeshcutError, match="Kein Element heisst"):
        ed.choose_region(sc, regs, name="gold")


@pytest.mark.parametrize("at", ["1,2", "a,b,c", "1,2,nan", "1,2,3,4"])
def test_choose_at_checks_the_point(tmp_path, at):
    sc, regs = regions_of(scene_obj(tmp_path))
    with pytest.raises(mc.MeshcutError, match="nicht verstanden"):
        ed.choose_region(sc, regs, at=at)


def test_at_picks_the_smallest_element_there(tmp_path):
    sc, regs = regions_of(scene_obj(tmp_path))
    assert ed.choose_region(sc, regs, at="1,1,2")["name"] == "red"          # Oberseite: rot (2) liegt in Teil 1 (12)
    assert ed.choose_region(sc, regs, at="1,1,0")["name"] == "blue"


# ----------------------------------------------------------------------------- remove
def test_remove_a_free_standing_part(tmp_path):
    p = scene_obj(tmp_path)
    r = mc.remove(p, region=2, out=tmp_path / "o.obj")
    assert "unveraendert" in r["text"] and "von 24 Dreiecken 12 entfernt, 0 neu" in r["text"] and "Loch" not in r["text"]
    sc = ed.read_scene(tmp_path / "o.obj")
    assert len(sc.faces) == 6 and {f.obj for f in sc.faces} == {"part1"}
    assert len(sc.V) == 8 and len(sc.vt) == 24 and len(sc.vn) == 6


def test_remove_closes_the_hole_with_neighbour_material_uv_and_normal(tmp_path):
    p = scene_obj(tmp_path)
    out = tmp_path / "o.obj"
    r = mc.remove(p, region=4, out=out)
    assert "Loch geschlossen" in r["text"]
    new = ed.read_scene(out)
    pid = ed.position_ids(new)
    assert all(v == 2 for v in ed.edge_counts(new, pid, [i for i, f in enumerate(new.faces) if f.obj == "part1"]).values())     # Teil 1 wieder dicht
    patch = new.faces[-2:]
    assert all(f.mat == "blue" and f.obj == "part1" for f in patch)             # Umgebung ist blau, nicht rot
    for f in patch:
        assert all(c[1] >= 0 and c[2] >= 0 for c in f.c)                         # UV und Normale vorhanden
        n = np.array([float(x) for x in new.vn[f.c[0][2]].split()])
        assert np.allclose(n, (0, 0, 1))                                         # Deckel zeigt nach aussen
        P = new.V[[c[0] for c in f.c]]
        assert np.cross(P[1] - P[0], P[2] - P[0])[2] > 0                         # und die Windung stimmt dazu


def test_remove_rest_is_identical_independent_check(tmp_path):
    p = scene_obj(tmp_path)
    src = ed.read_scene(p)
    mc.remove(p, region=4, out=tmp_path / "o.obj")
    new = ed.read_scene(tmp_path / "o.obj")
    red = {i for i, f in enumerate(src.faces) if f.mat == "red"}
    want = [src.signature(f) for i, f in enumerate(src.faces) if i not in red]
    assert [new.signature(f) for f in new.faces[:len(want)]] == want and len(new.faces) == len(want) + 2


def _skew_hexagon_cube(tmp_path):
    h, b = cube_lines((0, 0, 0), (2, 2, 2), 0, 0, 0, ["a", "b", "c", "a", "b", "c"], "w")
    out, fi = [], 0
    for ln in h + b:
        if ln.startswith("f "):
            out.append("g " + ("weg" if fi in (1, 3, 4) else "rest"))
            fi += 1
        out.append(ln)
    p = tmp_path / "w.obj"
    p.write_text(chr(10).join(out) + chr(10))
    return p


def test_remove_non_planar_hole_is_closed_by_projected_ear_clip_with_own_normals(tmp_path):
    """Windschiefes Sechseck: der Ohrenschnitt in der Randebene genuegt, ohne neuen Punkt; jedes Dreieck hat seine eigene Normale."""
    p = _skew_hexagon_cube(tmp_path)
    out_p = tmp_path / "o.obj"
    r = mc.remove(p, name="weg", out=out_p)
    new = ed.read_scene(out_p)
    assert "Loch geschlossen" in r["text"] and "offene Kanten" not in r["text"]
    assert all(v == 2 for v in ed.edge_counts(new, ed.position_ids(new)).values())
    assert len(new.V) == 7 and len(new.faces) == 3 + 4 and all(len(f.c) == 3 for f in new.faces[3:])
    assert len(new.vn) == 3 + 4


def test_remove_non_planar_hole_gets_a_centre_fan_and_is_watertight(tmp_path, monkeypatch):
    """Scheitert der Ohrenschnitt, bleibt der Faecher vom Mittelpunkt (nur wenn der Rand sternfoermig ist)."""
    monkeypatch.setattr(ed, "_projected_ear_clip", lambda *a: None)
    p = _skew_hexagon_cube(tmp_path)
    out_p = tmp_path / "o.obj"
    r = mc.remove(p, name="weg", out=out_p)
    new = ed.read_scene(out_p)
    assert "Loch geschlossen" in r["text"] and "offene Kanten" not in r["text"]
    assert all(v == 2 for v in ed.edge_counts(new, ed.position_ids(new)).values())
    assert len(new.V) == 7 + 1                                                 # 7 Ecken bleiben, der Mittelpunkt kommt dazu
    assert all(len(f.c) == 3 for f in new.faces[3:]) and len(new.faces) == 3 + 6
    assert len(new.vn) == 3 + 6                                                # drei alte Normalen der Restseiten, je Dreieck eine neue


def test_remove_without_uv_and_normals_writes_none(tmp_path):
    p = scene_obj(tmp_path, with_vt=False, with_vn=False)
    mc.remove(p, region=4, out=tmp_path / "o.obj")
    txt = (tmp_path / "o.obj").read_text()
    assert "vt " not in txt and "vn " not in txt and "//" not in txt


def test_remove_refuses_to_overwrite_the_input_and_checks_the_extension(tmp_path):
    p = scene_obj(tmp_path)
    with pytest.raises(mc.MeshcutError, match="nicht die Eingabedatei"):
        mc.remove(p, region=2, out=p)
    with pytest.raises(mc.MeshcutError, match="Endung"):
        mc.remove(p, region=2, out=tmp_path / "o.txt")
    assert "part2" in p.read_text()


def test_remove_default_output_name_and_mtl_copy(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    p = scene_obj(d)
    r = mc.remove(p, region=2)
    assert Path(r["output"]).name == "scene_edit.obj" and (d / "scene_edit.obj").is_file()
    e = tmp_path / "elsewhere"
    e.mkdir()
    r = mc.remove(p, region=2, out=e / "x.obj")
    assert (e / "m.mtl").is_file() and "Materialdatei mitkopiert" in r["text"]


def test_remove_everything_is_refused(tmp_path):
    sc, regs = regions_of(scene_obj(tmp_path))
    whole = dict(regs[0], face_ids=list(range(len(sc.faces))))
    with pytest.raises(mc.MeshcutError, match="nichts uebrig"):
        ed.edit_scene(sc, whole, "remove")


def test_remove_reports_changed_open_edges(tmp_path):
    p = scene_obj(tmp_path)
    r = mc.remove(p, region=3, out=tmp_path / "o.obj")                        # Material blue: Rest = offene Oberseite, wird geschlossen
    assert "offene Kanten" not in r["text"] or "nachher 0" in r["text"]


def test_stl_is_edited_geometry_only(tmp_path):
    a, b = shapes.cube(), shapes.cube(lo=(5, 0, 0))
    V = np.vstack([a.V, b.V])
    F = np.vstack([a.F, b.F + len(a.V)])
    p = tmp_path / "two.stl"
    p.write_bytes(mc.mesh_bytes("stl", V, F))
    r = mc.remove(p, region=2, out=tmp_path / "o.stl")
    assert "nur mit Geometrie" in r["text"]
    assert len(mc.load_mesh(tmp_path / "o.stl").F) == len(a.F)


# ----------------------------------------------------------------------------- swap
def test_swap_sphere_has_the_right_size_material_normals_and_uv(tmp_path):
    p = scene_obj(tmp_path)
    out = tmp_path / "o.obj"
    r = mc.swap(p, region=2, shape="sphere", out=out)
    assert "Kugel" in r["text"] and "unveraendert" in r["text"]
    sc = ed.read_scene(out)
    ball = [i for i, f in enumerate(sc.faces) if f.obj == "part2"]
    assert all(sc.faces[i].mat == "green" and sc.faces[i].smooth == "1" for i in ball)
    vol = mesh_volume(sc, ball)
    assert 0.9 * (math.pi / 6) < vol <= math.pi / 6 * 1.001                      # Kugel mit Durchmesser 1 in der Kiste 1x1x1 (innen eingeschrieben)
    P = sc.V[[c[0] for i in ball for c in sc.faces[i].c]]
    assert np.allclose(P.min(axis=0), (5, 0, 0)) and np.allclose(P.max(axis=0), (6, 1, 1))
    for i in ball:
        for v, t, n in sc.faces[i].c:
            nv = np.array([float(x) for x in sc.vn[n].split()])
            assert abs(np.linalg.norm(nv) - 1) < 1e-9
            u, w = [float(x) for x in sc.vt[t].split()]
            assert -1e-9 <= u <= 1 + 1e-9 and -1e-9 <= w <= 1 + 1e-9
    closed = ed.edge_counts(sc, ed.position_ids(sc), ball)
    assert all(v == 2 for v in closed.values())


def test_swap_box_fills_the_bounding_box_exactly_with_flat_normals(tmp_path):
    p = scene_obj(tmp_path)
    mc.swap(p, region=2, shape="box", scale=2.0, out=tmp_path / "o.obj")
    sc = ed.read_scene(tmp_path / "o.obj")
    box = [i for i, f in enumerate(sc.faces) if f.obj == "part2"]
    assert len(box) == 12 and abs(mesh_volume(sc, box) - 8.0) < 1e-9             # Kante 1 mal 2
    assert all(sc.faces[i].smooth == "" for i in box)
    ns = {tuple(round(float(x), 6) for x in sc.vn[c[2]].split()) for i in box for c in sc.faces[i].c}
    assert ns == {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_swap_cylinder_axis_and_volume(tmp_path, axis):
    h, b = cube_lines((0, 0, 0), (1, 1, 1), 0, 0, 0, ["m"] * 6, "a")
    h2, b2 = cube_lines((5, 0, 0), (7, 1, 1), 8, 24, 6, ["m"] * 6, "b")
    p = tmp_path / "s.obj"
    p.write_text("\n".join(h + h2 + b + b2) + "\n")
    mc.swap(p, name="Teil 2", shape="cylinder", axis=axis, out=tmp_path / "o.obj")
    sc = ed.read_scene(tmp_path / "o.obj")
    cyl = [i for i, f in enumerate(sc.faces) if f.obj == "b"]
    ext = np.array([2.0, 1.0, 1.0])
    a1, a2 = [k for k in range(3) if k != "xyz".index(axis)]
    expect = math.pi * (ext[a1] / 2) * (ext[a2] / 2) * ext["xyz".index(axis)]
    vol = mesh_volume(sc, cyl)
    assert 0.97 * expect < vol <= expect * 1.001
    assert all(v == 2 for v in ed.edge_counts(sc, ed.position_ids(sc), cyl).values())


def test_swap_auto_shape_follows_the_original(tmp_path):
    """Kugel bleibt Kugel, Quader bleibt Quader."""
    sph = shapes.sphere(n=16, m=10, r=1.0, c=(5, 0, 0))
    box = shapes.cube(2.0, lo=(-1, -1, -1))
    V = np.vstack([box.V, sph.V])
    F = np.vstack([box.F, sph.F + len(box.V)])
    p = tmp_path / "m.stl"
    p.write_bytes(mc.mesh_bytes("stl", V, F))
    sc, regs = regions_of(p)
    assert ed.choose_shape(by_name(regs, "Teil 2"))[0] == "sphere"
    assert ed.choose_shape(by_name(regs, "Teil 1"))[0] == "box"


def test_swap_material_override_and_default(tmp_path):
    p = scene_obj(tmp_path)
    mc.swap(p, region=2, shape="box", material="red", out=tmp_path / "o.obj")
    sc = ed.read_scene(tmp_path / "o.obj")
    assert {f.mat for f in sc.faces if f.obj == "part2"} == {"red"}


def test_swap_of_an_attached_element_closes_the_hole_first(tmp_path):
    p = scene_obj(tmp_path)
    r = mc.swap(p, region=4, shape="box", out=tmp_path / "o.obj")
    assert "Loch geschlossen" in r["text"]
    sc = ed.read_scene(tmp_path / "o.obj")
    assert all(v == 2 for v in ed.edge_counts(sc, ed.position_ids(sc)).values())


def test_swap_flat_element_gets_a_thickness(tmp_path):
    p = scene_obj(tmp_path)
    mc.swap(p, region=4, shape="box", out=tmp_path / "o.obj")                  # Oberseite ist flach (z-Ausdehnung 0)
    sc = ed.read_scene(tmp_path / "o.obj")
    box = [i for i, f in enumerate(sc.faces) if f.mat == "red"]
    assert mesh_volume(sc, box) > 0.05


@pytest.mark.parametrize("kw,match", [({"shape": "pyramide"}, "Form"), ({"axis": "w"}, "Achse")])
def test_swap_rejects_unknown_shape_and_axis(tmp_path, kw, match):
    with pytest.raises(mc.MeshcutError, match=match):
        ed.choose_shape({"bbox": {"min": [0, 0, 0], "max": [1, 1, 1]}, "area": 6.0}, **kw)


def test_swap_uv_follows_the_uv_box_of_the_original(tmp_path):
    h, b = cube_lines((0, 0, 0), (1, 1, 1), 0, 0, 0, ["m"] * 6, "a")
    h2, b2 = cube_lines((5, 0, 0), (6, 1, 1), 8, 24, 6, ["m"] * 6, "b")
    h2 = [("vt %g %g" % (0.25 + 0.5 * float(ln.split()[1]), 0.5 + 0.5 * float(ln.split()[2]))) if ln.startswith("vt ") else ln for ln in h2]
    p = tmp_path / "u.obj"
    p.write_text("\n".join(h + h2 + b + b2) + "\n")
    mc.swap(p, name="Teil 2", shape="sphere", out=tmp_path / "o.obj")
    sc = ed.read_scene(tmp_path / "o.obj")
    uvs = np.array([[float(x) for x in sc.vt[c[1]].split()] for f in sc.faces if f.obj == "b" for c in f.c])
    assert uvs[:, 0].min() >= 0.25 - 1e-9 and uvs[:, 0].max() <= 0.75 + 1e-9 and uvs[:, 1].min() >= 0.5 - 1e-9 and uvs[:, 1].max() <= 1.0 + 1e-9


# ----------------------------------------------------------------------------- Selbstpruefung
def test_self_check_fails_when_the_rest_changed(tmp_path, monkeypatch):
    p = scene_obj(tmp_path)
    real = ed.scene_bytes

    def tamper(sc, fmt):
        data = real(sc, fmt)
        return data.replace(b"v 0 0 0", b"v 0 0 0.5", 1)
    monkeypatch.setattr(ed, "scene_bytes", tamper)
    with pytest.raises(mc.MeshcutError, match="Selbstpruefung"):
        mc.remove(p, region=2, out=tmp_path / "o.obj")


# ----------------------------------------------------------------------------- tools
def test_tools_schema_comes_from_the_parser(capsys):
    assert mc.main(["tools"]) in (0, None)
    d = json.loads(capsys.readouterr().out)
    names = [t["name"] for t in d["tools"]]
    assert {"meshcut_split", "meshcut_join", "meshcut_verify", "meshcut_regions", "meshcut_remove", "meshcut_swap", "meshcut_info"} <= set(names)
    assert "meshcut_serve" not in names and "meshcut_tools" not in names
    for t in d["tools"]:
        sch = t["input_schema"]
        assert sch["type"] == "object" and set(sch["required"]) <= set(sch["properties"]) and t["description"]
    swap = next(t for t in d["tools"] if t["name"] == "meshcut_swap")
    assert swap["input_schema"]["properties"]["shape"]["enum"] == ["auto", "box", "sphere", "cylinder"]
    assert swap["input_schema"]["required"] == ["input"]
    split = next(t for t in d["tools"] if t["name"] == "meshcut_split")
    assert split["input_schema"]["properties"]["plane"]["type"] == "array" and split["input_schema"]["properties"]["wedges"]["type"] == "integer"
    assert split["input_schema"]["properties"]["no_cap"]["type"] == "boolean"
    join = next(t for t in d["tools"] if t["name"] == "meshcut_join")
    assert join["input_schema"]["properties"]["parts"]["type"] == "array"


def test_tools_api_matches_the_command():
    assert [t["name"] for t in mc.tools()] == [t["name"] for t in mc.tool_definitions()]


def test_every_parser_command_is_a_tool_except_serve_and_tools():
    parser = mc.build_parser()
    import argparse
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert {"meshcut_" + n for n in sub.choices} - {"meshcut_serve", "meshcut_tools"} == {t["name"] for t in mc.tools()}


def test_command_line_runs_the_new_commands(tmp_path, capsys):
    p = scene_obj(tmp_path)
    assert mc.main(["regions", str(p), str(tmp_path / "r")]) in (0, None)
    assert mc.main(["swap", str(p), "--region", "2", "--shape", "box", "--out", str(tmp_path / "o.obj")]) in (0, None)
    out = capsys.readouterr().out
    assert "Teil 2" in out and "Ersetzt" in out and "unveraendert" in out


# ----------------------------------------------------------------------------- Dienst (MCP/HTTP)
def test_serve_tools_regions_remove_swap(tmp_path):
    import kkmeshcut_serve as serve
    scene_obj(tmp_path)
    t = serve.Tools(serve.Sandbox(tmp_path))
    js = serve.call_tool(t, "regions", {"input": "scene.obj", "outdir": "r"})
    assert [r["id"] for r in js["regions"]] == [1, 2, 3, 4]
    r = serve.call_tool(t, "remove", {"input": "scene.obj", "region": 2, "out": "a.obj"})
    assert r["output"] == "a.obj" and "unveraendert" in r["report"]
    r = serve.call_tool(t, "swap", {"input": "scene.obj", "name": "red", "shape": "box", "scale": 1.5})
    assert r["output"] == "scene_edit.obj" and (tmp_path / "scene_edit.obj").is_file()
    names = {s["name"] for s in serve.tool_specs()}
    assert {"regions", "remove", "swap"} <= names


@pytest.mark.parametrize("tool,args,match", [("remove", {"input": "scene.obj", "region": "2"}, "ganze Zahl"), ("swap", {"input": "scene.obj", "region": 2, "scale": "2"}, "Zahl"),
                                              ("swap", {"input": "scene.obj", "region": 2, "scale": True}, "Zahl"), ("remove", {"input": "../x.obj", "region": 1}, "ausserhalb"),
                                              ("remove", {"input": "scene.obj", "region": 2, "out": "../o.obj"}, "ausserhalb"),
                                              ("remove", {"input": "scene.obj"}, "Genau eine Auswahl")])
def test_serve_edit_tools_validate_arguments(tmp_path, tool, args, match):
    import kkmeshcut_serve as serve
    scene_obj(tmp_path)
    with pytest.raises(mc.MeshcutError, match=match):
        serve.call_tool(serve.Tools(serve.Sandbox(tmp_path)), tool, args)


# ----------------------------------------------------------------------------- Schutz vor boesen Dateien und Eingaben
@pytest.mark.parametrize("mtl", ["../evil.mtl", "sub/x.mtl", "/abs.mtl", "..", "C:" + chr(92) + "evil.mtl"])
def test_mtllib_cannot_make_meshcut_write_outside(tmp_path, mtl):
    d = tmp_path / "in"
    d.mkdir()
    p = scene_obj(d)
    p.write_text(p.read_text().replace("mtllib m.mtl", "mtllib " + mtl))
    (tmp_path / "evil.mtl").write_text("x")
    (d / "sub").mkdir()
    (d / "sub" / "x.mtl").write_text("x")
    e = tmp_path / "out" / "deep"
    e.mkdir(parents=True)
    r = mc.remove(p, region=2, out=e / "o.obj")
    assert sorted(f.name for f in e.iterdir()) == ["o.obj"] and "mitkopiert" not in r["text"]
    assert not (tmp_path / "out" / "evil.mtl").exists() and not (tmp_path / "out" / "x.mtl").exists()


@pytest.mark.parametrize("name", ["rot\nusemtl boese", "a\rb", "", "   ", "x" * 129, "t\x00t"])
def test_swap_material_name_is_validated(tmp_path, name):
    p = scene_obj(tmp_path)
    with pytest.raises(mc.MeshcutError, match="Material"):
        mc.swap(p, region=2, material=name, out=tmp_path / "o.obj")
    assert not (tmp_path / "o.obj").exists()


def test_region_number_true_is_not_one(tmp_path):
    sc, regs = regions_of(scene_obj(tmp_path))
    with pytest.raises(mc.MeshcutError, match="keine ganze Zahl"):
        ed.choose_region(sc, regs, region=True)


def test_too_many_faces_are_refused(tmp_path, monkeypatch):
    p = scene_obj(tmp_path)
    monkeypatch.setattr(ed, "MAX_EDIT_FACES", 10)
    with pytest.raises(mc.MeshcutError, match="nicht bearbeitet"):
        ed.read_scene(p)
    s = tmp_path / "s.stl"
    s.write_bytes(mc.mesh_bytes("stl", shapes.cube().V, shapes.cube().F))
    with pytest.raises(mc.MeshcutError, match="nicht bearbeitet"):
        ed.read_scene(s)
    monkeypatch.setattr(ed, "MAX_EDIT_FACES", 12)
    assert len(ed.read_scene(s).faces) == 12


def test_many_parts_are_capped_and_the_largest_stay(tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "MAX_REGIONS", 3)
    parts = []
    for i in range(6):
        c = shapes.cube(1.0 + (i % 3), lo=(10 * i, 0, 0))
        parts.append(c)
    V = np.vstack([c.V for c in parts])
    F = np.vstack([c.F + sum(len(q.V) for q in parts[:i]) for i, c in enumerate(parts)])
    p = tmp_path / "many.stl"
    p.write_bytes(mc.mesh_bytes("stl", V, F))
    sc, regs = regions_of(p)
    assert len(regs) == 3 and ed.find_regions.omitted == 3
    assert [r["id"] for r in regs] == [1, 2, 3]
    js = mc.regions(p, tmp_path / "r")
    assert js["omitted"] == 3 and len(js["regions"]) == 3


def test_regions_refuses_links_as_target(tmp_path):
    import os
    p = scene_obj(tmp_path)
    (tmp_path / "t").mkdir()
    link = tmp_path / "l"
    try:
        os.symlink(tmp_path / "t", link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("keine Links moeglich")
    with pytest.raises(mc.MeshcutError, match="Link"):
        mc.regions(p, link)


# ----------------------------------------------------------------------------- gezielte Tests (Mutanten, Runde 6)
def test_region_cap_keeps_the_biggest_parts_in_file_order(tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "MAX_REGIONS", 3)
    parts = [shapes.cube(1.0, lo=(10 * i, 0, 0), sub=s) for i, s in enumerate([1, 3, 1, 3, 2, 1])]
    off = np.cumsum([0] + [len(c.V) for c in parts])
    V = np.vstack([c.V for c in parts])
    F = np.vstack([c.F + o for c, o in zip(parts, off)])
    p = tmp_path / "many.stl"
    p.write_bytes(mc.mesh_bytes("stl", V, F))
    sc, regs = regions_of(p)
    assert [r["triangles"] for r in regs] == [len(parts[1].F), len(parts[3].F), len(parts[4].F)]
    assert [r["bbox"]["min"][0] for r in regs] == [10.0, 30.0, 40.0]


@pytest.mark.parametrize("ext,axis", [((2, 1, 1), 0), ((1, 3, 1), 1), ((1, 1, 4), 2)])
def test_cylinder_axis_is_the_odd_one_out(ext, axis):
    reg = {"bbox": {"min": [0, 0, 0], "max": list(ext)}, "area": 10.0}
    assert ed.choose_shape(reg, "cylinder", "auto") == ("cylinder", axis)
    assert ed.choose_shape(reg, "cylinder", "y")[1] == 1


def test_hidden_part_is_marked_not_visible(tmp_path):
    outer = shapes.cube(4.0, lo=(0, 0, 0))
    inner = shapes.cube(1.0, lo=(1.5, 1.5, 1.5))
    V = np.vstack([outer.V, inner.V])
    F = np.vstack([outer.F, inner.F + len(outer.V)])
    p = tmp_path / "i.stl"
    p.write_bytes(mc.mesh_bytes("stl", V, F))
    js = mc.regions(p, tmp_path / "r", size=300)
    assert [r["visible"] for r in js["regions"]] == [True, False]


def test_vertices_at_the_same_place_belong_to_one_part_even_when_not_shared(tmp_path):
    """OBJ aus Dreiecksuppe (jede Flaeche eigene Eckpunkte): ein Quader ist trotzdem ein Teil."""
    lines = []
    for ox in (0, 5):
        c = shapes.cube(1.0, lo=(ox, 0, 0))
        for f in c.F:
            for k in f:
                lines.append("v %r %r %r" % tuple(float(x) for x in c.V[k]))
    n = 0
    for _ in range(24):
        lines.append("f %d %d %d" % (n + 1, n + 2, n + 3))
        n += 3
    p = tmp_path / "soup.obj"
    p.write_text("\n".join(lines) + "\n")
    sc, regs = regions_of(p)
    assert [(r["kind"], r["triangles"]) for r in regs] == [("teil", 12), ("teil", 12)]


def test_centre_fan_gets_a_uv_in_the_middle_of_the_rim(tmp_path):
    h, b = cube_lines((0, 0, 0), (2, 2, 2), 0, 0, 0, ["a", "b", "c", "a", "b", "c"], "w")
    out, fi = [], 0
    for ln in h + b:
        if ln.startswith("f "):
            out.append("g " + ("weg" if fi in (1, 3, 4) else "rest"))
            fi += 1
        out.append(ln)
    p = tmp_path / "w.obj"
    p.write_text("\n".join(out) + "\n")
    mc.remove(p, name="weg", out=tmp_path / "o.obj")
    new = ed.read_scene(tmp_path / "o.obj")
    fan = new.faces[3:]
    centre = {c[1] for f in fan for c in f.c if c[0] == max(cc[0] for ff in fan for cc in ff.c)}
    assert len(centre) == 1 and next(iter(centre)) >= 0
    uv = [float(x) for x in new.vt[next(iter(centre))].split()]
    rim = np.array([[float(x) for x in new.vt[c[1]].split()] for f in fan for c in f.c if c[1] >= 0])
    assert rim.min(axis=0)[0] - 1e-9 <= uv[0] <= rim.max(axis=0)[0] + 1e-9 and all(c[1] >= 0 for f in fan for c in f.c)


# ----------------------------------------------------------------------------- Pruefer: Kandidaten bewerten
@pytest.mark.parametrize("make,kind", [(lambda: shapes.cube(2.0), "box"), (lambda: shapes.sphere(n=24, m=14, r=1.0), "sphere")])
def test_swap_ranks_candidates_and_the_matching_one_wins(tmp_path, make, kind):
    a, b = make(), shapes.cube(1.0, lo=(9, 9, 9))
    V = np.vstack([a.V, b.V])
    F = np.vstack([a.F, b.F + len(a.V)])
    p = tmp_path / "m.stl"
    p.write_bytes(mc.mesh_bytes("stl", V, F))
    sc, regs = regions_of(p)
    rank = ed.rank_shapes(sc, regs[0])
    assert rank[0][1] == kind and rank[0][0] < 0.15 and [r[0] for r in rank] == sorted(r[0] for r in rank)
    assert {(k, a) for _, k, a in rank} == {("box", 0), ("sphere", 0), ("cylinder", 0), ("cylinder", 1), ("cylinder", 2)}
    r = mc.swap(p, region=1, out=tmp_path / "o.stl")
    assert "Kandidaten verglichen" in r["text"]


def test_swap_axis_limits_the_cylinder_candidates(tmp_path):
    sc, regs = regions_of(scene_obj(tmp_path))
    rank = ed.rank_shapes(sc, regs[1], axis="y")
    assert [(k, a) for _, k, a in rank if k == "cylinder"] == [("cylinder", 1)]


def test_explicit_shape_skips_the_ranking(tmp_path):
    r = mc.swap(scene_obj(tmp_path), region=2, shape="sphere", out=tmp_path / "o.obj")
    assert "Kandidaten" not in r["text"]


def test_triangle_quality_scale():
    eq = np.array([[0, 0, 0], [1, 0, 0], [0.5, math.sqrt(3) / 2, 0]])
    assert abs(ed._quality(eq) - 1.0) < 1e-12
    assert ed._quality(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])) == 0.0
    assert ed._quality(np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.01, 0]])) < 0.05


def test_remove_has_no_candidate_ranking(tmp_path):
    r = mc.remove(scene_obj(tmp_path), region=2, out=tmp_path / "o.obj")
    assert "Kandidaten" not in r["text"]


def test_sliver_ear_clip_falls_back_to_a_fan(tmp_path, monkeypatch):
    """Sind die Ohrenschnitt-Dreiecke Splitter und der Faecher vom Mittelpunkt gut, gilt der Faecher; das Loch bleibt trotzdem zu."""
    real = ed._quality
    calls = []

    def fake(P):
        calls.append(1)
        return 0.001 if len(calls) <= 2 else real(P)             # die zwei Ohrenschnitt-Dreiecke des Vierecks sind "Splitter"
    monkeypatch.setattr(ed, "_quality", fake)
    out = tmp_path / "o.obj"
    mc.remove(scene_obj(tmp_path), region=4, out=out)
    monkeypatch.undo()
    new = ed.read_scene(out)
    pid = ed.position_ids(new)
    assert len(new.faces[-4:]) == 4 and all(len(f.c) == 3 for f in new.faces[-4:])      # Faecher aus vier statt zwei Dreiecken
    assert all(v == 2 for v in ed.edge_counts(new, pid, [i for i, f in enumerate(new.faces) if f.obj == "part1"]).values())


def test_good_ear_clip_is_kept(tmp_path):
    out = tmp_path / "o.obj"
    mc.remove(scene_obj(tmp_path), region=4, out=out)
    new = ed.read_scene(out)
    assert len([f for f in new.faces if f.obj == "part1" and f.mat == "blue"]) >= 1 and len(new.faces[-2:]) == 2


# ----------------------------------------------------------------------------- Pruefrunde 6: Befunde von Spock
def test_object_name_finds_a_single_part_object(tmp_path):
    """Ein Objekt, das zugleich ein eigenes Teil ist, steht als 'Teil N' im Verzeichnis, bleibt aber unter seinem Objektnamen waehlbar."""
    p = scene_obj(tmp_path)
    r = mc.remove(p, name="part2", out=tmp_path / "o.obj")
    assert "Region 2" in r["text"]
    sc = ed.read_scene(p)
    regs = ed.find_regions(sc)
    assert any(a["name"] == "part2" for r_ in regs for a in r_["also"])
    with pytest.raises(mc.MeshcutError, match="part2"):
        mc.remove(p, name="gibtsnicht", out=tmp_path / "o2.obj")
    assert ed.choose_region(sc, regs, name="PART2")["id"] == ed.choose_region(sc, regs, name="part2")["id"]


def test_ambiguous_name_is_refused(tmp_path):
    p = tmp_path / "a.obj"
    h1, b1 = cube_lines((0, 0, 0), (1, 1, 1), 0, 0, 0, ["x"] * 6, "A")
    h2, b2 = cube_lines((3, 0, 0), (4, 1, 1), 8, 24, 6, ["y"] * 6, "x")        # Objekt heisst wie das Material des ersten
    p.write_text("\n".join(h1 + h2 + b1 + b2) + "\n")
    with pytest.raises(mc.MeshcutError, match="mehrere"):
        mc.remove(p, name="x", out=tmp_path / "o.obj")


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), float("inf"), 1000.0, True])
def test_scale_must_be_sane(tmp_path, bad):
    with pytest.raises(mc.MeshcutError, match="--scale"):
        mc.swap(scene_obj(tmp_path), region=1, scale=bad, out=tmp_path / "o.obj")
    assert not (tmp_path / "o.obj").exists()


@pytest.mark.parametrize("bad", [-5, 0, 10, 40000])
def test_size_must_be_sane(tmp_path, bad):
    with pytest.raises(mc.MeshcutError, match="--size"):
        mc.regions(scene_obj(tmp_path), tmp_path / "r", size=bad)


def test_swap_without_any_material_does_not_inherit_the_previous_one(tmp_path):
    """Ein Ersatz ohne Material darf nicht still das Material der Flaeche davor erben."""
    h, b = cube_lines((0, 0, 0), (1, 1, 1), 0, 0, 0, ["blue"] * 6, "A")
    h2, b2 = cube_lines((3, 0, 0), (4, 1, 1), 8, 24, 6, ["blue"] * 6, "B")
    p = tmp_path / "n.obj"
    txt = "\n".join(h + h2 + b + b2) + "\n"
    p.write_text(txt)
    sc = ed.read_scene(p)
    for f in sc.faces[6:]:
        f.mat = None                                                # zweiter Wuerfel: gar kein Material
    q = tmp_path / "n2.obj"
    q.write_bytes(ed.obj_text(sc))
    assert b"usemtl off" in q.read_bytes()
    back = ed.read_scene(q)
    assert [f.mat for f in back.faces[:6]] == ["blue"] * 6 and [f.mat for f in back.faces[6:]] == [None] * 6
    out = tmp_path / "o.obj"
    mc.swap(q, region=2, shape="box", out=out)
    new = ed.read_scene(out)
    assert new.faces[-1].mat is None and b"usemtl off" in out.read_bytes()                 # der Ersatz steht ohne Material da, nicht mit dem davor


def test_equal_size_regions_pick_the_lower_number_everywhere(tmp_path):
    """--at und das Flag at_selects_this meinen bei gleich grossen, sich deckenden Elementen dasselbe."""
    h, b = cube_lines((0, 0, 0), (1, 1, 1), 0, 0, 0, ["m1", "m1", "m1", "m1", "m2", "m2"], "O")
    p = tmp_path / "t.obj"
    p.write_text("\n".join(h + b) + "\n")
    sc = ed.read_scene(p)
    regs = ed.find_regions(sc)
    for r in regs:
        got = ed.choose_region(sc, regs, at=",".join("%.6g" % x for x in r["point"]))
        assert (got["id"] == r["id"]) == r["at_selects_this"]


def test_self_check_covers_new_faces(tmp_path, monkeypatch):
    """Veraendert sich nur der Ersatz (Material) im Geschriebenen, schlaegt die Pruefung an und die Ausgabe wird entfernt."""
    p = scene_obj(tmp_path)
    real = ed.scene_bytes

    def tamper(sc, fmt):
        data = real(sc, fmt)
        i = data.rindex(b"usemtl ")
        return data[:i] + b"usemtl ZZZ\n" + data[data.index(b"\n", i) + 1:]
    monkeypatch.setattr(ed, "scene_bytes", tamper)
    out = tmp_path / "o.obj"
    with pytest.raises(mc.MeshcutError, match="Selbstpruefung"):
        mc.swap(p, region=2, shape="box", out=out)
    assert not out.exists()


def test_report_numbers_come_from_the_written_file(tmp_path):
    p = scene_obj(tmp_path)
    r = mc.remove(p, region=2, out=tmp_path / "o.obj")
    assert "aus der geschriebenen Datei" in r["text"]
    back = ed.read_scene(tmp_path / "o.obj")
    assert str(ed.read_scene(p).n_triangles()) in r["text"] and back.n_triangles() < ed.read_scene(p).n_triangles()


def test_other_formats_are_checked_too(tmp_path):
    src = tmp_path / "a.stl"
    sc = ed.read_scene(scene_obj(tmp_path))
    src.write_bytes(ed.scene_bytes(sc, "stl"))
    r = mc.remove(src, region=2, out=tmp_path / "o.stl")
    assert "Farben, Materialien und UV" in r["text"]


# ----------------------------------------------------------------------------- Pruefrunde 6, zweite Welle: Gleichstand, Formate, Woelbung
def _tie_scene(tmp_path):
    """Wuerfel aus 6 Vierecken: Objekt A = Flaechen 0-3, Objekt B = 4-5, Material m1 = Flaechen 2-5 (gleich gross wie A, ueberlappt es)."""
    quads = [(1, 2, 3, 4), (5, 8, 7, 6), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 8, 4), (5, 1, 4, 8)]
    mats = ["m2", "m2", "m1", "m1", "m1", "m1"]
    objs = ["A", "A", "A", "A", "B", "B"]
    lines = ["v 0 0 0", "v 1 0 0", "v 1 1 0", "v 0 1 0", "v 0 0 1", "v 1 0 1", "v 1 1 1", "v 0 1 1"]
    for q, m, o in zip(quads, mats, objs):
        lines += ["o " + o, "usemtl " + m, "f " + " ".join(map(str, q))]
    p = tmp_path / "tie.obj"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_overlapping_equal_regions_choose_the_lower_number_in_both_places(tmp_path):
    sc = ed.read_scene(_tie_scene(tmp_path))
    regs = ed.find_regions(sc)
    by = {r["name"]: r for r in regs}
    a, m1 = by["A"], by["m1"]
    assert len(a["face_ids"]) == len(m1["face_ids"]) and a["id"] < m1["id"]
    tri = sc.V[[c[0] for c in sc.faces[2].c]].mean(axis=0)                    # Flaeche 2 liegt in beiden
    got = ed.choose_region(sc, regs, at=",".join("%.9g" % x for x in tri))
    assert got["id"] == a["id"]
    assert a["at_selects_this"] and not m1["at_selects_this"]


def test_other_format_with_missing_triangle_fails_the_self_check(tmp_path, monkeypatch):
    src = tmp_path / "a.stl"
    sc = ed.read_scene(scene_obj(tmp_path))
    src.write_bytes(ed.scene_bytes(sc, "stl"))
    real = ed.scene_bytes

    def tamper(s, fmt):
        t = s.copy()
        t.faces = t.faces[:-1]
        return real(t, fmt)
    monkeypatch.setattr(ed, "scene_bytes", tamper)
    out = tmp_path / "o.stl"
    with pytest.raises(mc.MeshcutError, match="Selbstpruefung"):
        mc.remove(src, region=2, out=out)
    assert not out.exists()


def _paraboloid_loop(order):
    ang = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    f = lambda x, y: 0.3 * (x * x + y * y)
    Pr = np.array([[math.cos(a), math.sin(a), f(math.cos(a), math.sin(a))] for a in ang])
    sc = ed.Scene()
    sc.V = np.vstack([Pr, [[1.6, 0, f(1.6, 0)], [1.5, 0.4, f(1.5, 0.4)]]])
    sc.faces = [ed.Face([(0, -1, -1), (12, -1, -1), (13, -1, -1)])]
    rloop = [(k, k, -1, -1, 0) for k in range(12)]
    Pr = Pr[order]
    return sc, [rloop[k] for k in order], Pr


def test_curved_fill_rings_need_a_star_shaped_loop_in_turning_order():
    ok = _paraboloid_loop(list(range(12)))
    assert ed._curved_fill(ok[0], ok[1], ok[2], ok[2].mean(axis=0), np.array([0.0, 0.0, 1.0])) is not None
    assert ed._curved_fill(ok[0], ok[1], ok[2], ok[2].mean(axis=0), np.array([0.0, 0.0, 1.0]))["star"]
    cw = _paraboloid_loop(list(range(12))[::-1])                              # falsch herum: kein Stern, die Flaeche wird trotzdem gefittet
    got = ed._curved_fill(cw[0], cw[1], cw[2], cw[2].mean(axis=0), np.array([0.0, 0.0, 1.0]))
    assert got is not None and not got["star"]
    mix = _paraboloid_loop([0, 5, 2, 7, 4, 1, 6, 3, 8, 11, 10, 9])           # verdreht
    got = ed._curved_fill(mix[0], mix[1], mix[2], mix[2].mean(axis=0), np.array([0.0, 0.0, 1.0]))
    assert got is not None and not got["star"]


def test_self_check_catches_a_reversed_fill(tmp_path, monkeypatch):
    """Liegt die Fuellung falsch herum, ist das Ergebnis nicht gruen: die Prueflinie meldet es und die Ausgabe verschwindet."""
    real = ed.close_holes

    def reversed_fill(sc, pid, removed):
        faces, n, u = real(sc, pid, removed)
        for f in faces:
            f.c = f.c[::-1]
        return faces, n, u
    monkeypatch.setattr(ed, "close_holes", reversed_fill)
    out = tmp_path / "o.obj"
    with pytest.raises(mc.MeshcutError, match="falsch herum"):
        mc.remove(scene_obj(tmp_path), region=4, out=out)
    assert not out.exists()


def test_clean_results_have_no_duplicate_directed_edges(tmp_path):
    p = scene_obj(tmp_path)
    out = tmp_path / "o.obj"
    mc.remove(p, region=4, out=out)
    assert ed.directed_duplicates(ed.read_scene(out)) == 0 == ed.directed_duplicates(ed.read_scene(p))


def test_curved_fill_refuses_surroundings_that_are_no_smooth_surface():
    """Ist die Umgebung keine glatte Flaeche (zufaellige Hoehen), bleibt es beim flachen Deckel - der Fit wird nicht erzwungen."""
    rng = np.random.default_rng(3)
    sc, rloop, Pr = _paraboloid_loop(list(range(12)))
    sc.V[:12, 2] += rng.normal(0, 0.5, 12)
    sc.V[12:, 2] += rng.normal(0, 0.5, 2)
    Pr = sc.V[:12].copy()
    assert ed._curved_fill(sc, rloop, Pr, Pr.mean(axis=0), np.array([0.0, 0.0, 1.0])) is None


# ----------------------------------------------------------------------------- Terra-Befunde
def test_mtllib_symlink_is_not_followed_out_of_the_folder(tmp_path):
    """Ein Link im Eingabeordner auf eine Datei ausserhalb wird nicht in die Ausgabe kopiert."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.mtl").write_text("newmtl blue\nKd 0 0 1\n")
    work = tmp_path / "in"
    work.mkdir()
    p = scene_obj(work)
    (work / "m.mtl").unlink()
    try:
        (work / "m.mtl").symlink_to(outside / "secret.mtl")
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks nicht erlaubt")
    dst = tmp_path / "out"
    dst.mkdir()
    r = mc.remove(p, region=2, out=dst / "o.obj")
    assert not (dst / "m.mtl").exists() and "mitkopiert" not in r["text"]


def test_mtl_textures_are_copied_along_but_only_plain_names(tmp_path):
    work = tmp_path / "in"
    work.mkdir()
    p = scene_obj(work)
    (work / "m.mtl").write_text("newmtl blue\nmap_Kd wood.png\nbump -bm 1 normal.png\nmap_Ks ../evil.png\nmap_Kd sub/x.png\n")
    (work / "wood.png").write_bytes(b"W")
    (work / "normal.png").write_bytes(b"N")
    (tmp_path / "evil.png").write_bytes(b"E")
    dst = tmp_path / "out"
    dst.mkdir()
    r = mc.remove(p, region=2, out=dst / "o.obj")
    assert (dst / "wood.png").read_bytes() == b"W" and (dst / "normal.png").read_bytes() == b"N"
    assert not (dst / "evil.png").exists() and not (dst / "sub").exists()
    assert "wood.png" in r["text"]


def test_mtl_texture_symlink_is_not_copied(tmp_path):
    work = tmp_path / "in"
    work.mkdir()
    p = scene_obj(work)
    (work / "m.mtl").write_text("newmtl blue\nmap_Kd t.png\n")
    (tmp_path / "secret.png").write_bytes(b"S")
    try:
        (work / "t.png").symlink_to(tmp_path / "secret.png")
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks nicht erlaubt")
    dst = tmp_path / "out"
    dst.mkdir()
    mc.remove(p, region=2, out=dst / "o.obj")
    assert not (dst / "t.png").exists()


def test_concave_polygon_is_not_triangulated_as_a_fan(tmp_path):
    """Ein konkaves Fuenfeck (Pfeil) behaelt seine Flaeche, auch wenn die Ausgabe ein Dreiecksformat ist."""
    lines = ["v 0 0 0", "v 4 0 0", "v 4 4 0", "v 2 1 0", "v 0 4 0", "v 9 9 0", "v 10 9 0", "v 9 10 0",
             "o A", "f 1 2 3 4 5", "o B", "f 6 7 8"]
    p = tmp_path / "c.obj"
    p.write_text("\n".join(lines) + "\n")
    out = tmp_path / "o.stl"
    mc.remove(p, name="B", out=out)
    m = mc.load_mesh(out)
    assert abs(mc._face_areas(m.V, m.F).sum() - 10.0) < 1e-4                  # Pfeil: 4*4 minus Dreieck 4*3/2 = 10
    sc = ed.read_scene(p)
    T, src = sc.triangles([0])
    assert len(T) == 3 and abs(sum(abs(np.cross(sc.V[t[1]] - sc.V[t[0]], sc.V[t[2]] - sc.V[t[0]])[2]) / 2 for t in T) - 10.0) < 1e-9


def test_non_obj_self_check_catches_a_shifted_output(tmp_path, monkeypatch):
    src = tmp_path / "a.stl"
    src.write_bytes(ed.scene_bytes(ed.read_scene(scene_obj(tmp_path)), "stl"))
    real = ed.scene_bytes

    def shift(sc, fmt):
        t = sc.copy()
        t.V = t.V + 100.0
        t.vtext = ["%.9g %.9g %.9g" % tuple(v) for v in t.V]
        return real(t, fmt)
    monkeypatch.setattr(ed, "scene_bytes", shift)
    out = tmp_path / "o.stl"
    with pytest.raises(mc.MeshcutError, match="Ausdehnung"):
        mc.remove(src, region=2, out=out)
    assert not out.exists()


def test_verify_says_open_parts_are_not_checked_without_caps(tmp_path):
    cube = shapes.cube(2.0, lo=(-1, -1, -1))
    c = tmp_path / "c.obj"
    c.write_bytes(mc.mesh_bytes("obj", cube.V, cube.F))
    mc.split(c, tmp_path / "o", plane=["x=0"], format="obj", no_cap=True)
    r = mc.verify(tmp_path / "o")
    texts = {n: t for n, ok, t in r["checks"]}
    assert "nicht geprueft" in texts["wasserdicht"] and "nicht geprueft" in texts["euler"]


# ----------------------------------------------------------------------------- Lochfuellung: einzelne Entscheidungen
def _ngon(n, r=1.0):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a), np.zeros(n)], axis=1)


def test_fan_is_only_consistent_for_a_star_shaped_loop():
    star = _ngon(8)
    assert ed._fan_is_consistent(star, star.mean(axis=0), np.array([0.0, 0.0, 1.0]))
    c_shape = np.array([[0, 0, 0], [4, 0, 0], [4, 1, 0], [1, 1, 0], [1, 3, 0], [4, 3, 0], [4, 4, 0], [0, 4, 0]], float)
    assert not ed._fan_is_consistent(c_shape, c_shape.mean(axis=0), np.array([0.0, 0.0, 1.0]))


def test_projected_ear_clip_refuses_triangles_that_face_against_the_plane_normal():
    poly = _ngon(7)
    up, down = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    assert ed._projected_ear_clip(poly, poly.mean(axis=0), up) is not None and len(ed._projected_ear_clip(poly, poly.mean(axis=0), up)) == 5
    assert ed._projected_ear_clip(poly, poly.mean(axis=0), down) is None


def test_forced_fan_carries_uv_for_the_centre_point(tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "_projected_ear_clip", lambda *a: None)
    p = _skew_hexagon_cube(tmp_path)
    out = tmp_path / "o.obj"
    mc.remove(p, name="weg", out=out)
    new = ed.read_scene(out)
    assert all(c[1] >= 0 for f in new.faces[3:] for c in f.c)


def test_curved_fill_uses_two_rings(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    import edit_bench as eb
    assert len(ed.CURVED_RINGS) == 2
    m = eb.case_hole("sphere", 1.0, tmp_path, "rect")
    assert m["fuell_dreiecke"] == 24 * (2 * len(ed.CURVED_RINGS) + 1)           # Rand mit 24 Punkten: je Ringpaar 2 Dreiecke je Punkt, dazu der Mittelfaecher


def test_ring_uv_falls_back_to_linear_when_the_boundary_uv_has_a_seam():
    """Springt die UV am Rand (Naht 0,98 -> 0,02), wuerde eine Abbildung aus der Lage Unsinn liefern: dann wird linear zur Mitte gemittelt."""
    sc, rloop, Pr = _paraboloid_loop(list(range(12)))
    u = np.array([0.95, 0.97, 0.99, 0.01, 0.03, 0.05, 0.05, 0.03, 0.01, 0.99, 0.97, 0.95])
    sc.vt = ["%.6f %.6f" % (a, 0.5) for a in u]
    rloop = [(x[0], x[1], k, x[3], x[4]) for k, x in enumerate(rloop)]
    cen = Pr.mean(axis=0)
    cv = ed._curved_fill(sc, rloop, Pr, cen, np.array([0.0, 0.0, 1.0]))
    assert cv is not None and cv["star"]
    ref = sc.faces[0]
    n_vt_before = len(sc.vt)
    ed._fill_with_rings(sc, rloop, Pr, cen, np.array([0.0, 0.0, 1.0]), cv, ref, False)
    uvc = np.array([[float(t) for t in s.split()[:2]] for s in sc.vt[:12]]).mean(axis=0)
    arr = np.array([[float(t) for t in s.split()[:2]] for s in sc.vt[:12]])
    first_ring = np.array([[float(t) for t in s.split()[:2]] for s in sc.vt[n_vt_before:n_vt_before + 12]])
    f = ed.CURVED_RINGS[0]
    assert np.allclose(first_ring, uvc + f * (arr - uvc), atol=1e-6)


def test_non_obj_self_check_catches_a_changed_area_with_same_extent(tmp_path, monkeypatch):
    """Rahmen bleibt gleich, aber eine Flaeche ist zusammengefallen: nur der Flaechenvergleich bemerkt es."""
    src = tmp_path / "a.stl"
    src.write_bytes(ed.scene_bytes(ed.read_scene(scene_obj(tmp_path)), "stl"))
    real = ed.scene_bytes

    def squash(sc, fmt):
        t = sc.copy()
        f = t.faces[0]
        t.faces = [ed.Face([f.c[0], f.c[1], f.c[1]] + list(f.c[3:]), f.mat, f.obj, f.grp, f.smooth)] + t.faces[1:]
        return real(t, fmt)
    monkeypatch.setattr(ed, "scene_bytes", squash)
    out = tmp_path / "o.stl"
    with pytest.raises(mc.MeshcutError, match="Flaeche oder Ausdehnung"):
        mc.remove(src, region=2, out=out)
    assert not out.exists()


def test_concave_quad_is_not_split_as_a_convex_one(tmp_path):
    """Ein konkaves Viereck (Pfeilspitze), dessen erste Ecke eine Diagonale nach aussen hat, behaelt seine Flaeche im STL."""
    lines = ["v 4 0 0", "v 1 1 0", "v 0 4 0", "v 0 0 0", "v 9 9 0", "v 10 9 0", "v 9 10 0",
             "o A", "f 1 2 3 4", "o B", "f 5 6 7"]
    p = tmp_path / "q.obj"
    p.write_text("\n".join(lines) + "\n")
    out = tmp_path / "o.stl"
    mc.remove(p, name="B", out=out)
    m = mc.load_mesh(out)
    assert abs(mc._face_areas(m.V, m.F).sum() - 4.0) < 1e-4


def test_mtl_texture_names_with_a_path_are_never_followed(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    p = scene_obj(work)
    (work / "sub").mkdir()
    (work / "sub" / "x.png").write_bytes(b"X")
    (tmp_path / "evil.png").write_bytes(b"E")
    (work / "m.mtl").write_text("newmtl blue\nmap_Kd sub/x.png\nmap_Ks ../evil.png\nmap_Kd ok.png\n")
    (work / "ok.png").write_bytes(b"O")
    dst = tmp_path / "o" / "d"
    dst.mkdir(parents=True)
    mc.remove(p, region=2, out=dst / "o.obj")
    assert (dst / "ok.png").read_bytes() == b"O"
    assert not (dst / "sub").exists() and not (tmp_path / "o" / "evil.png").exists()


def test_ring_uv_fit_check_needs_a_loop_that_is_not_a_circle():
    """Bei einem Rand mit wechselndem Radius ist die Abbildung Lage -> UV bestimmt; springt die UV (Naht), muss trotzdem linear gemittelt werden."""
    ang = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    rad = 1.0 + 0.25 * (np.arange(12) % 2)
    f = lambda x, y: 0.3 * (x * x + y * y)
    Pr = np.array([[r * math.cos(a), r * math.sin(a), f(r * math.cos(a), r * math.sin(a))] for r, a in zip(rad, ang)])
    sc = ed.Scene()
    sc.V = np.vstack([Pr, [[1.6, 0, f(1.6, 0)], [1.5, 0.4, f(1.5, 0.4)]]])
    sc.faces = [ed.Face([(0, -1, -1), (12, -1, -1), (13, -1, -1)])]
    u = np.array([0.95, 0.97, 0.99, 0.01, 0.03, 0.05, 0.05, 0.03, 0.01, 0.99, 0.97, 0.95])
    sc.vt = ["%.6f %.6f" % (a, 0.5) for a in u]
    rloop = [(k, k, k, -1, 0) for k in range(12)]
    cen = Pr.mean(axis=0)
    cv = ed._curved_fill(sc, rloop, Pr, cen, np.array([0.0, 0.0, 1.0]))
    assert cv is not None and cv["star"]
    n_before = len(sc.vt)
    ed._fill_with_rings(sc, rloop, Pr, cen, np.array([0.0, 0.0, 1.0]), cv, sc.faces[0], False)
    arr = np.array([[float(t) for t in s.split()[:2]] for s in sc.vt[:12]])
    ring = np.array([[float(t) for t in s.split()[:2]] for s in sc.vt[n_before:n_before + 12]])
    uvc = arr.mean(axis=0)
    assert np.allclose(ring, uvc + ed.CURVED_RINGS[0] * (arr - uvc), atol=1e-6)


def test_ear_clip_refuses_a_loop_that_runs_against_the_normal():
    Pr = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]])
    n = np.array([0.0, 0.0, 1.0])
    cen = Pr.mean(axis=0)
    t = ed._projected_ear_clip(Pr, cen, n)
    assert t is not None and len(t) == 2
    assert ed._projected_ear_clip(Pr[::-1], cen, n) is None


# ----------------------------------------------------------------------------- Spock R8
def _grid_obj(n=8, hole=((2, 2), (3, 2)), lift=None, uv=False, mtl="m.mtl"):
    V = [(i / n, j / n, (lift or {}).get((i, j), 0.0)) for j in range(n + 1) for i in range(n + 1)]
    L = ["mtllib " + mtl] + ["v %r %r %r" % v for v in V]
    if uv:
        L += ["vt %r %r" % (v[0], v[1]) for v in V]
    cur = None
    for j in range(n):
        for i in range(n):
            m = "hole" if (i, j) in hole else "base"
            if m != cur:
                L.append("usemtl " + m)
                cur = m
            a, b, c, d = j * (n + 1) + i + 1, j * (n + 1) + i + 2, (j + 1) * (n + 1) + i + 2, (j + 1) * (n + 1) + i + 1
            quad = (a, b, c, d)
            L.append("f " + " ".join(("%d/%d" % (q, q)) if uv else str(q) for q in quad))
    return "\n".join(L) + "\n"


def test_latin1_names_survive_unchanged(tmp_path):
    p = scene_obj(tmp_path)
    p.write_bytes(p.read_bytes().replace(b"usemtl red", b"usemtl r\xe4d"))
    out = tmp_path / "o.obj"
    mc.remove(p, region=2, out=out)
    data = out.read_bytes()
    assert b"usemtl r\xe4d" in data and b"\xef\xbf\xbd" not in data


def test_utf8_bom_does_not_break_the_first_vertex(tmp_path):
    p = scene_obj(tmp_path)
    p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())
    sc = ed.read_scene(p)
    assert len(sc.V) == 16 and sc.mtllib == ["m.mtl"]
    out = tmp_path / "o.obj"
    mc.remove(p, region=2, out=out)
    assert len(mc.load_mesh(out).F) > 0


def test_failed_check_leaves_an_earlier_good_result_untouched(tmp_path, monkeypatch):
    src = tmp_path / "a.stl"
    src.write_bytes(ed.scene_bytes(ed.read_scene(scene_obj(tmp_path)), "stl"))
    out = tmp_path / "o.stl"
    mc.remove(src, region=2, out=out)
    good = out.read_bytes()
    real = ed.scene_bytes

    def shift(sc, fmt):
        t = sc.copy()
        t.V = t.V + 100.0
        return real(t, fmt)
    monkeypatch.setattr(ed, "scene_bytes", shift)
    with pytest.raises(mc.MeshcutError):
        mc.remove(src, region=2, out=out)
    assert out.read_bytes() == good
    assert not [f for f in tmp_path.iterdir() if f.name.startswith(".meshcut")]


def test_hole_on_a_uv_seam_gets_continuous_uv_values():
    sc = ed.Scene()
    sc.V = np.zeros((6, 3))
    sc.vt = ["0.95 0.5", "0.97 0.5", "0.99 0.5", "0.01 0.5", "0.03 0.5", "0.05 0.5"]
    rloop = [(k, k, k, -1, 0) for k in range(6)]
    out = ed._unwrap_uv_seam(sc, rloop)
    u = [float(sc.vt[x[2]].split()[0]) for x in out]
    assert max(u) - min(u) < 0.2 and np.all(np.diff(u) > 0)
    assert sc.vt[:6] == ["0.95 0.5", "0.97 0.5", "0.99 0.5", "0.01 0.5", "0.03 0.5", "0.05 0.5"]       # vorhandene Eintraege bleiben
    ring = [(k, k, k, -1, 0) for k in range(6)]
    sc.vt = ["%.2f 0.5" % x for x in (0.0, 0.4, 0.8, 0.2, 0.6, 0.9)]                             # Rand umschliesst die Textur (Pol): nicht entwirrbar
    assert ed._unwrap_uv_seam(sc, ring) == ring


def test_stl_far_from_the_origin_passes_the_self_check(tmp_path):
    p = tmp_path / "far.obj"
    lines = []
    for ln in _grid_obj().splitlines():
        if ln.startswith("v "):
            x, y, z = map(float, ln.split()[1:])
            ln = "v %r %r %r" % (x + 100000.1234567, y + 100000.1234567, z)
        lines.append(ln)
    p.write_text("\n".join(lines) + "\n")
    out = tmp_path / "o.stl"
    mc.remove(p, name="hole", out=out)
    assert out.exists() and len(mc.load_mesh(out).F) > 0


def test_mtllib_with_two_names_copies_both(tmp_path):
    work = tmp_path / "in"
    work.mkdir()
    p = work / "g.obj"
    p.write_text(_grid_obj(mtl="a.mtl b.mtl"))
    (work / "a.mtl").write_text("newmtl base\nKd 1 0 0\n")
    (work / "b.mtl").write_text("newmtl hole\nKd 0 1 0\n")
    dst = tmp_path / "out"
    dst.mkdir()
    mc.remove(p, name="hole", out=dst / "o.obj")
    assert (dst / "a.mtl").exists() and (dst / "b.mtl").exists()


def test_nan_uv_in_the_rim_is_caught_by_the_self_check(tmp_path):
    t = _grid_obj(uv=True).splitlines()
    k = next(i for i, ln in enumerate(t) if ln.startswith("vt "))
    n = 8
    idx = 2 * (n + 1) + 2                                          # ein Eckpunkt am Lochrand
    t[k + idx] = "vt nan nan"
    p = tmp_path / "nan.obj"
    p.write_text("\n".join(t) + "\n")
    out = tmp_path / "o.obj"
    with pytest.raises(mc.MeshcutError, match="ungueltige"):
        mc.remove(p, name="hole", out=out)
    assert not out.exists()


def test_swap_auto_says_when_no_candidate_fits(tmp_path, monkeypatch):
    sc, regs = regions_of(scene_obj(tmp_path))
    monkeypatch.setattr(ed, "rank_shapes", lambda *a, **k: [(0.4, "box", 0), (0.5, "sphere", 0)])
    _, _, rep = ed.edit_scene(sc, regs[1], "swap")
    assert any("Kein Kandidat" in h for h in rep["hinweise"])
    monkeypatch.setattr(ed, "rank_shapes", lambda *a, **k: [(0.01, "box", 0)])
    _, _, rep = ed.edit_scene(sc, regs[1], "swap")
    assert rep["hinweise"] == []


def test_uneven_rim_hole_is_filled_flat_but_says_so(tmp_path):
    lift = {(2, 2): 0.15, (3, 2): 0.0, (4, 2): 0.1, (4, 3): 0.0, (3, 3): 0.12, (2, 3): 0.0}
    p = tmp_path / "u.obj"
    p.write_text(_grid_obj(lift=lift))
    sc = ed.read_scene(p)
    regs = ed.find_regions(sc)
    _, _, rep = ed.edit_scene(sc, ed.choose_region(sc, regs, name="hole"), "remove")
    assert rep["loecher_geschlossen"] == 1 and any("nicht ebenem Rand" in h for h in rep["hinweise"])


def test_centre_fan_normals_point_along_the_winding(tmp_path, monkeypatch):
    real = ed._quality
    calls = []

    def fake(P):
        calls.append(1)
        return 0.001 if len(calls) <= 2 else real(P)
    monkeypatch.setattr(ed, "_quality", fake)
    out = tmp_path / "o.obj"
    mc.remove(scene_obj(tmp_path), region=4, out=out)
    monkeypatch.undo()
    new = ed.read_scene(out)
    for f in new.faces[-4:]:
        P = new.V[[c[0] for c in f.c]]
        g = np.cross(P[1] - P[0], P[2] - P[0])
        vn = np.array([float(x) for x in new.vn[f.c[0][2]].split()])
        assert float(g @ vn) > 0


def test_swap_takes_the_dominant_material(tmp_path):
    sc, regs = regions_of(scene_obj(tmp_path))
    out, kept, rep = ed.edit_scene(sc, regs[0], "swap")
    new_mats = {f.mat for f in out.faces[len(kept):]}
    assert new_mats == {"blue"}


# ----------------------------------------------------------------------------- Spock R9
def test_latin1_names_are_json_safe_in_the_region_list(tmp_path):
    p = scene_obj(tmp_path)
    p.write_bytes(p.read_bytes().replace(b"usemtl red", b"usemtl r\xe9d"))
    sc = ed.read_scene(p)
    regs = ed.find_regions(sc)
    text = json.dumps(regs, ensure_ascii=False).encode("utf-8")                 # so antwortet der HTTP-Dienst
    assert b"r" in text and len(text) > 10
    out = tmp_path / "o.obj"
    mc.remove(p, region=2, out=out)
    assert b"usemtl r\xe9d" in out.read_bytes()


def test_one_component_and_broken_vt_do_not_crash_the_fill(tmp_path):
    t = _grid_obj(uv=True).splitlines()
    t = [("vt " + ln.split()[1]) if ln.startswith("vt ") else ln for ln in t]
    p = tmp_path / "one.obj"
    p.write_text("\n".join(t) + "\n")
    out = tmp_path / "o.obj"
    mc.remove(p, name="hole", out=out)
    assert out.exists()
    t2 = _grid_obj(uv=True).splitlines()
    k = next(i for i, ln in enumerate(t2) if ln.startswith("vt "))
    t2[k + 2 * 9 + 2] = "vt abc def"
    q = tmp_path / "abc.obj"
    q.write_text("\n".join(t2) + "\n")
    with pytest.raises(mc.MeshcutError, match="ungueltige"):                  # saubere Meldung statt Traceback
        mc.remove(q, name="hole", out=tmp_path / "o2.obj")
    assert not (tmp_path / "o2.obj").exists()


def test_output_that_is_a_folder_is_refused_with_its_own_name(tmp_path):
    d = tmp_path / "ziel.obj"
    d.mkdir()
    with pytest.raises(mc.MeshcutError) as e:
        mc.remove(scene_obj(tmp_path), region=2, out=d)
    assert "ziel.obj" in str(e.value) and ".meshcut" not in str(e.value)


def test_a_file_that_looks_like_staging_is_never_touched(tmp_path):
    precious = tmp_path / ".meshcut-pruefung-st.obj"
    precious.write_text("KEEP")
    mc.remove(scene_obj(tmp_path), region=2, out=tmp_path / "st.obj")
    assert precious.read_text() == "KEEP"


def test_existing_mtl_and_texture_in_the_output_folder_are_not_overwritten(tmp_path):
    work = tmp_path / "in"
    work.mkdir()
    p = scene_obj(work)
    (work / "m.mtl").write_text("newmtl blue\nmap_Kd t.png\n")
    (work / "t.png").write_bytes(b"NEU")
    dst = tmp_path / "out"
    dst.mkdir()
    (dst / "m.mtl").write_text("ALT")
    (dst / "t.png").write_bytes(b"ALT")
    mc.remove(p, region=2, out=dst / "o.obj")
    assert (dst / "m.mtl").read_text() == "ALT" and (dst / "t.png").read_bytes() == b"ALT"
    dst2 = tmp_path / "out2"
    dst2.mkdir()
    (dst2 / "t.png").write_bytes(b"ALT")
    mc.remove(p, region=2, out=dst2 / "o.obj")
    assert (dst2 / "t.png").read_bytes() == b"ALT" and (dst2 / "m.mtl").exists()


def test_mtllib_name_with_a_space_stays_whole_when_the_file_exists(tmp_path):
    work = tmp_path / "in"
    work.mkdir()
    p = work / "g.obj"
    p.write_text(_grid_obj(mtl="my mat.mtl"))
    (work / "my mat.mtl").write_text("newmtl base\nKd 1 0 0\n")
    dst = tmp_path / "out"
    dst.mkdir()
    mc.remove(p, name="hole", out=dst / "o.obj")
    assert (dst / "my mat.mtl").exists()


@pytest.mark.parametrize("axis", [0, 1])
def test_seam_jump_of_seventy_percent_counts_as_a_seam_on_either_axis(axis):
    sc = ed.Scene()
    sc.V = np.zeros((6, 3))
    vals = [0.9, 0.95, 0.2, 0.25, 0.3, 0.35]
    sc.vt = [("%.2f 0.5" % v) if axis == 0 else ("0.5 %.2f" % v) for v in vals]
    rloop = [(k, k, k, -1, 0) for k in range(6)]
    out = ed._unwrap_uv_seam(sc, rloop)
    got = [float(sc.vt[x[2]].split()[axis]) for x in out]
    assert got == pytest.approx([0.9, 0.95, 1.2, 1.25, 1.3, 1.35])


def test_float32_allowance_does_not_hide_a_gross_shift_far_from_the_origin(tmp_path, monkeypatch):
    p = tmp_path / "far.obj"
    lines = []
    for ln in _grid_obj().splitlines():
        if ln.startswith("v "):
            x, y, z = map(float, ln.split()[1:])
            ln = "v %r %r %r" % (x + 100000.0, y + 100000.0, z)
        lines.append(ln)
    p.write_text("\n".join(lines) + "\n")
    real = ed.scene_bytes

    def shifted(sc, fmt):
        t = sc.copy()
        t.V = t.V + np.array([0.5, 0.0, 0.0])
        return real(t, fmt)
    monkeypatch.setattr(ed, "scene_bytes", shifted)
    with pytest.raises(mc.MeshcutError, match="Ausdehnung"):
        mc.remove(p, name="hole", out=tmp_path / "o.stl")


def test_nonfinite_normal_in_a_new_face_is_caught(tmp_path, monkeypatch):
    def bad(self, n):
        self.vn.append("nan nan nan")
        return len(self.vn) - 1
    monkeypatch.setattr(ed.Scene, "add_vn", bad)
    out = tmp_path / "o.obj"
    with pytest.raises(mc.MeshcutError, match="ungueltige"):
        mc.remove(scene_obj(tmp_path), region=4, out=out)
    assert not out.exists()


def test_rank_score_counts_both_directions_and_the_worst_point():
    """Ein L aus zwei Quaderseiten: jeder Eckpunkt liegt auf dem Quader (Original -> Kandidat 0), aber zwei Ecken des Quaders liegen weit vom L entfernt."""
    V = [(x, y, z) for z in (0, 2) for y in (0, 2) for x in (0, 2)]
    lines = ["v %d %d %d" % v for v in V] + ["f 1 3 7 5", "f 1 2 6 5"]
    sc = ed.parse_obj(("\n".join(lines) + "\n").encode())
    region = {"bbox": {"min": [0, 0, 0], "max": [2, 2, 2]}, "face_ids": [0, 1]}
    ranking = ed.rank_shapes(sc, region)
    score = {(k, a): v for v, k, a in ranking}[("box", 0)]
    diag = 2 * math.sqrt(3)
    assert score == pytest.approx(0.5 / 2 / diag + 0.5 * 2.0 / diag, rel=0.02)         # Mittel 0,5 (zwei von acht Ecken im Abstand 2) und schlechtester Punkt 2


def test_swap_replacement_covers_the_uv_box_of_the_original(tmp_path):
    sc, regs = regions_of(scene_obj(tmp_path))
    removed = regs[1]["face_ids"]
    orig = np.array([[float(x) for x in sc.vt[c[1]].split()[:2]] for i in removed for c in sc.faces[i].c if c[1] >= 0])
    out, kept, _ = ed.edit_scene(sc, regs[1], "swap")
    new = np.array([[float(x) for x in out.vt[c[1]].split()[:2]] for f in out.faces[len(kept):] for c in f.c if c[1] >= 0])
    assert np.allclose(new.min(axis=0), orig.min(axis=0)) and np.allclose(new.max(axis=0), orig.max(axis=0))


# ----------------------------------------------------------------------------- Spock R10
def _scene_with_names(tmp_path, obj_a, obj_b, mtllib="m.mtl"):
    lines = ["mtllib " + mtllib, "v 0 0 0", "v 1 0 0", "v 0 1 0", "v 5 5 5", "v 6 5 5", "v 5 6 5", "o " + obj_a, "f 1 2 3", "o " + obj_b, "f 4 5 6"]
    return ("\n".join(lines) + "\n")


def test_real_private_use_characters_survive_next_to_latin1_bytes(tmp_path):
    """Gemischt: ein Latin-1-Byte und ein echtes (gueltig kodiertes) Zeichen U+F7E9 - beides kommt unveraendert zurueck."""
    text = _scene_with_names(tmp_path, "caf\u00e9", "Z\uf7e9\uf700")
    data = text.encode("latin-1", errors="replace").replace(b"caf?", b"caf\xe9")
    data = data.replace(b"Z?", "Z\uf7e9\uf700".encode("utf-8")).replace(b"?", b"")
    p = tmp_path / "mix.obj"
    p.write_bytes(data)
    out = tmp_path / "o.obj"
    mc.remove(p, name="Z\uf7e9\uf700", out=out)
    got = out.read_bytes()
    assert b"o caf\xe9" in got and "Z\uf7e9".encode("utf-8") not in got


def test_valid_utf8_with_private_use_characters_is_untouched(tmp_path):
    p = tmp_path / "pua.obj"
    p.write_bytes(_scene_with_names(tmp_path, "A\uf7e9", "B").encode("utf-8"))
    out = tmp_path / "o.obj"
    mc.remove(p, name="B", out=out)
    assert "o A\uf7e9".encode("utf-8") in out.read_bytes()


def test_escape_round_trip_is_exact_for_all_bytes():
    raw = bytes(range(0x80, 0x100)) + "x\uf700\uf780\uf7ffy".encode("utf-8") + b"\xe9"
    text, esc = ed._decode_text(raw)
    assert esc and ed._encode_text(text, esc) == raw


def test_latin1_region_name_can_be_used_as_shown(tmp_path):
    p = tmp_path / "l.obj"
    p.write_bytes(_scene_with_names(tmp_path, "caf\u00e9", "B").encode("latin-1"))
    sc = ed.read_scene(p)
    regs = ed.find_regions(sc)
    assert any("café" in [r["name"]] + [x["name"] for x in r["also"]] for r in regs)
    out = tmp_path / "o.obj"
    mc.remove(p, name="caf\u00e9", out=out)
    assert out.exists()


def test_latin1_mtllib_file_is_copied_along(tmp_path):
    work = tmp_path / "in"
    work.mkdir()
    p = work / "l.obj"
    p.write_bytes(_scene_with_names(tmp_path, "A", "B", mtllib="caf\u00e9.mtl").encode("latin-1"))
    (work / "caf\u00e9.mtl").write_text("newmtl x\n")
    dst = tmp_path / "out"
    dst.mkdir()
    mc.remove(p, name="B", out=dst / "o.obj")
    assert (dst / "caf\u00e9.mtl").exists()


@pytest.mark.parametrize("sep", ["\u2028", "\x0b", "\x0c", "\u0085"])
def test_unusual_line_separators_do_not_cut_a_name(tmp_path, sep):
    p = tmp_path / "s.obj"
    p.write_bytes(_scene_with_names(tmp_path, "a" + sep + "b", "B").encode("utf-8"))
    sc = ed.read_scene(p)
    assert any(f.obj == "a" + sep + "b" for f in sc.faces)


def test_locked_target_error_names_the_target(tmp_path, monkeypatch):
    out = tmp_path / "o.obj"
    mc.remove(scene_obj(tmp_path), region=2, out=out)
    good = out.read_bytes()

    real = os.replace

    def locked(a, b):
        if Path(b) == out:
            raise PermissionError("in Benutzung")
        return real(a, b)
    monkeypatch.setattr(ed.os, "replace", locked)
    with pytest.raises(mc.MeshcutError) as e:
        mc.remove(scene_obj(tmp_path), region=2, out=out)
    assert "o.obj" in str(e.value) and ".meshcut" not in str(e.value)
    monkeypatch.undo()
    assert out.read_bytes() == good and not [f for f in tmp_path.iterdir() if f.name.startswith(".meshcut")]


# ----------------------------------------------------------------------------- Spock R11
def test_self_check_survives_real_pua_name_next_to_a_latin1_object(tmp_path):
    """Objekt A heisst mit einem echten U+F7E9, Objekt B hat ein Latin-1-Byte: Entfernt man B, ist die Ausgabe reines UTF-8 - die Pruefung darf nicht anschlagen."""
    lines = ["v 0 0 0", "v 1 0 0", "v 0 1 0", "v 5 5 5", "v 6 5 5", "v 5 6 5"]
    data = ("\n".join(lines) + "\no A\uf7e9\nf 1 2 3\no B").encode("utf-8") + b"\xe9\nf 4 5 6\n"
    p = tmp_path / "m.obj"
    p.write_bytes(data)
    out = tmp_path / "o.obj"
    mc.remove(p, region=2, out=out)
    assert "A\uf7e9".encode("utf-8") in out.read_bytes()


def test_bom_with_latin1_bytes_keeps_the_first_vertex(tmp_path):
    data = b"\xef\xbb\xbfv 0 0 0\nv 1 0 0\nv 0 1 0\nv 5 5 5\nv 6 5 5\nv 5 6 5\no caf\xe9\nf 1 2 3\no B\nf 4 5 6\n"
    sc = ed.parse_obj(data)
    assert len(sc.V) == 6 and sc.escaped


def test_typed_material_in_a_latin1_scene_is_written_as_the_same_byte(tmp_path):
    p = tmp_path / "l.obj"
    p.write_bytes(b"mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 5 5 5\nv 6 5 5\nv 5 6 5\nusemtl r\xe9d\no A\nf 1 2 3\nusemtl b\no B\nf 4 5 6\n")
    sc = ed.read_scene(p)
    regs = ed.find_regions(sc)
    out, kept, _ = ed.edit_scene(sc, regs[1], "swap", material="r\u00e9d")
    data = ed.obj_text(out)
    assert data.count(b"usemtl r\xe9d") >= 1 and b"r\xc3\xa9d" not in data


def test_region_materials_are_shown_plain_and_swap_keeps_the_raw_name(tmp_path):
    p = tmp_path / "l.obj"
    p.write_bytes(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 5 5 5\nv 6 5 5\nv 5 6 5\nusemtl r\xe9d\no A\nf 1 2 3\nusemtl b\no B\nf 4 5 6\n")
    sc = ed.read_scene(p)
    regs = ed.find_regions(sc)
    assert "r\u00e9d" in regs[0]["materials"]
    out, kept, _ = ed.edit_scene(sc, regs[0], "swap")
    assert ed.obj_text(out).count(b"usemtl r\xe9d") >= 1


def test_mtl_texture_with_a_latin1_name_is_copied(tmp_path):
    work = tmp_path / "in"
    work.mkdir()
    p = work / "o.obj"
    p.write_bytes(b"mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 5 5 5\nv 6 5 5\nv 5 6 5\no A\nf 1 2 3\no B\nf 4 5 6\n")
    (work / "m.mtl").write_bytes(b"newmtl x\nmap_Kd t\xe9x.png\n")
    (work / "t\u00e9x.png").write_bytes(b"P")
    dst = tmp_path / "out"
    dst.mkdir()
    mc.remove(p, region=2, out=dst / "o.obj")
    assert (dst / "t\u00e9x.png").exists()


@pytest.mark.parametrize("bad", [" a", "a ", "a\u2028b", "a\u0085b"])
def test_material_option_refuses_edge_spaces_and_line_separators(bad):
    with pytest.raises(mc.MeshcutError):
        ed._check_material(bad)


def test_names_keep_a_trailing_nbsp(tmp_path):
    p = tmp_path / "n.obj"
    p.write_bytes("v 0 0 0\nv 1 0 0\nv 0 1 0\no a\u00a0\nf 1 2 3\n".encode("utf-8"))
    assert ed.read_scene(p).faces[0].obj == "a\u00a0"


# ----------------------------------------------------------------------------- Spock R12
def _sphere_obj_with_cap(tmp_path, zc):
    m = shapes.sphere(48, 32)
    lines = ["v %.17g %.17g %.17g" % tuple(float(x) for x in p) for p in m.V]
    cur = None
    for f in m.F:
        mat = "red" if m.V[f].mean(axis=0)[2] > zc else "blue"
        if mat != cur:
            lines.append("usemtl " + mat)
            cur = mat
        lines.append("f %d %d %d" % tuple(int(i) + 1 for i in f))
    p = tmp_path / "s.obj"
    p.write_text("\n".join(lines) + "\n")
    return p


@pytest.mark.parametrize("zc", [0.7, 0.3, 0.0])
def test_big_cap_hole_never_gets_a_spike(tmp_path, zc):
    """Spock R12: eine Paraboloid-Fuellung eines grossen Lochs ragte als Spitze weit ueber die Kugel; jetzt bleibt sie nahe an der Kugel oder flach."""
    p = _sphere_obj_with_cap(tmp_path, zc)
    out = tmp_path / "o.obj"
    mc.remove(p, name="red", out=out)
    V = mc.load_mesh(out).V
    assert np.linalg.norm(V, axis=1).max() < 1.1 and V[:, 2].min() > -1.1


def test_curved_fill_refuses_a_rise_above_the_limit():
    ang = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    for k, expect in ((0.2, True), (3.0, False)):
        f = lambda x, y, k=k: k * (x * x + y * y)
        Pr = np.array([[math.cos(a), math.sin(a), f(math.cos(a), math.sin(a))] for a in ang])
        sc = ed.Scene()
        sc.V = np.vstack([Pr, [[1.6, 0, f(1.6, 0)], [1.5, 0.4, f(1.5, 0.4)]]])
        sc.faces = [ed.Face([(0, -1, -1), (12, -1, -1), (13, -1, -1)])]
        rloop = [(i, i, -1, -1, 0) for i in range(12)]
        got = ed._curved_fill(sc, rloop, Pr, Pr.mean(axis=0), np.array([0.0, 0.0, 1.0]))
        assert (got is not None) == expect


@pytest.mark.parametrize("scale", [0.8, 1.5, 2.0])
def test_swap_scale_does_not_change_the_chosen_shape(tmp_path, scale):
    sc, regs = regions_of(scene_obj(tmp_path))
    _, _, rep = ed.edit_scene(sc, regs[1], "swap", scale=scale)
    assert rep["ersatz"].startswith("Quader") and not rep["hinweise"]


def test_material_off_is_refused_with_a_clear_message():
    with pytest.raises(mc.MeshcutError, match="off"):
        ed._check_material("off")
    with pytest.raises(mc.MeshcutError, match="off"):
        ed._check_material("Off")


def test_material_with_a_trailing_nbsp_can_be_selected_by_name(tmp_path):
    p = tmp_path / "n.obj"
    p.write_bytes("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 5 5 5\nv 6 5 5\nv 5 6 5\nusemtl red\u00a0\no A\nf 1 2 3\nusemtl b\no B\nf 4 5 6\n".encode("utf-8"))
    out = tmp_path / "o.obj"
    mc.remove(p, name="red\u00a0", out=out)
    assert out.exists()
    ed._check_material("red\u00a0")


def test_service_errors_do_not_show_the_root_folder(tmp_path):
    import kkmeshcut_serve as serve
    tools = serve.Tools(serve.Sandbox(tmp_path))
    root = str(tmp_path.resolve())
    assert serve._scrub(tools, root + os.sep + "x.obj gibt es nicht.") == "x.obj gibt es nicht."
    assert root not in serve._scrub(tools, "Fehler in " + root.replace(os.sep, "/") + "/a/b.obj")


def test_label_csv_with_a_byte_order_mark_is_read(tmp_path):
    p = tmp_path / "l.csv"
    p.write_bytes(b"\xef\xbb\xbf0,1,0\n")
    assert list(mc.read_label_file(p)) == [0, 1, 0]


# ============================================================================= Spock R13
def _open_edges(path):
    F = mc.load_mesh(path).F
    cnt = {}
    for f in F:
        for k in range(3):
            e = tuple(sorted((int(f[k]), int(f[(k + 1) % 3]))))
            cnt[e] = cnt.get(e, 0) + 1
    return sum(1 for c in cnt.values() if c == 1)


@pytest.mark.parametrize("zc", [0.7, 0.3, 0.0])
def test_sphere_cap_hole_is_filled_on_the_sphere(tmp_path, zc):
    """Spock R13: ein Paraboloid ueberschoss Kugelkappen um bis zu 18 Prozent; die Fuellung folgt jetzt der Kugel."""
    p = _sphere_obj_with_cap(tmp_path, zc)
    out = tmp_path / "o.obj"
    mc.remove(p, name="red", out=out)
    V = mc.load_mesh(out).V
    assert np.linalg.norm(V, axis=1).max() < 1.01 and V[:, 2].max() > 0.98          # und nicht flach: die Mitte erreicht den Pol


@pytest.mark.parametrize("hole", [((2, 2), (3, 3)), ((3, 2), (2, 3)), ((1, 1), (2, 2), (3, 3)), ((2, 3), (3, 2), (4, 1))])
def test_hole_loops_split_at_a_touching_corner(tmp_path, hole):
    """Spock R13: zwei Loecher, die sich in einer Ecke beruehren, wurden als ein verschlungener Rand verfolgt und die Fuellung scheiterte."""
    p = tmp_path / "g.obj"
    p.write_text(_grid_obj(n=6, hole=hole))
    out = tmp_path / "o.obj"
    before = _open_edges(p)
    mc.remove(p, name="hole", out=out)
    assert _open_edges(out) == before


def test_exact_case_wins_between_names_that_differ_only_in_case(tmp_path):
    p = scene_obj(tmp_path)
    p.write_bytes(p.read_bytes().replace(b"usemtl red", b"usemtl Skin").replace(b"usemtl blue", b"usemtl skin"))
    out = tmp_path / "o.obj"
    mc.remove(p, name="Skin", out=out)
    assert b"usemtl Skin" not in out.read_bytes() and b"usemtl skin" in out.read_bytes()


def test_rejected_http_request_does_not_leave_its_body_as_a_second_request(tmp_path):
    """Spock R13: eine abgelehnte Anfrage las ihren Body nicht und liess die Verbindung offen; der Body wurde als naechste Anfrage ausgefuehrt."""
    import socket
    import threading
    import kkmeshcut_serve as serve
    srv = serve.make_http_server(serve.Tools(serve.Sandbox(tmp_path)), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        inner = ("GET /v1/health HTTP/1.1\r\nHost: localhost\r\n\r\n").encode()
        outer = ("POST /v1/info HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n" % len(inner)).encode() + inner
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        s.sendall(outer)
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        s.close()
        assert data.count(b"HTTP/1.1") == 1 and b"415" in data.split(b"\r\n")[0]
    finally:
        srv.shutdown()
        srv.server_close()


def test_service_success_reports_do_not_show_the_root_folder(tmp_path):
    import kkmeshcut_serve as serve
    tools = serve.Tools(serve.Sandbox(tmp_path))
    root = str(tmp_path.resolve())
    got = serve._scrub_obj(tools, {"report": "Geschrieben: " + root + os.sep + "o.obj", "n": 3, "l": ["x " + root]})
    assert root not in got["report"] and root not in got["l"][0] and got["n"] == 3
    dbl = root.replace(os.sep, os.sep * 2)
    assert dbl not in serve._scrub(tools, "OSError: " + dbl + os.sep * 2 + "a")


def test_service_calls_scrub_the_root_from_successful_answers(tmp_path):
    import kkmeshcut_serve as serve
    tools = serve.Tools(serve.Sandbox(tmp_path))
    root = str(tmp_path.resolve())
    tools.info = lambda a: {"report": "Geschrieben: " + root + os.sep + "o.obj"}
    assert root not in serve.call_tool(tools, "info", {})["report"]


def test_http_survives_a_lone_surrogate_in_the_answer(tmp_path):
    import threading
    import kkmeshcut_serve as serve
    srv = serve.make_http_server(serve.Tools(serve.Sandbox(tmp_path)), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        c.request("POST", "/v1/info", body=b'{"path": "a\\ud800b"}', headers={"Content-Type": "application/json"})
        assert c.getresponse().status in (400, 404, 500)
    finally:
        srv.shutdown()
        srv.server_close()


# ============================================================================= Spock R14
def _raw_http(tmp_path, request):
    import socket
    import threading
    import kkmeshcut_serve as serve
    srv = serve.make_http_server(serve.Tools(serve.Sandbox(tmp_path)), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=5)
        s.sendall(request)
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        s.close()
        return data
    finally:
        srv.shutdown()
        srv.server_close()


def test_get_with_a_body_is_refused_and_not_run_as_a_second_request(tmp_path):
    inner = b"GET /v1/health HTTP/1.1\r\nHost: localhost\r\n\r\n"
    outer = b"GET /v1/health HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: %d\r\n\r\n" % len(inner) + inner
    data = _raw_http(tmp_path, outer)
    assert data.count(b"HTTP/1.1") == 1 and b" 400 " in data.split(b"\r\n")[0]


@pytest.mark.parametrize("length", [b"1_6", b"+16", b"0x10", b"16, 16"])
def test_content_length_must_be_a_plain_decimal(tmp_path, length):
    req = b"POST /v1/info HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: " + length + b"\r\n\r\n{\"path\": \"a.obj\"}"
    data = _raw_http(tmp_path, req)
    assert b" 413 " in data.split(b"\r\n")[0] or b" 400 " in data.split(b"\r\n")[0]


def test_two_content_length_headers_are_refused(tmp_path):
    req = b"POST /v1/info HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 2\r\nContent-Length: 20\r\n\r\n{}"
    data = _raw_http(tmp_path, req)
    assert b" 413 " in data.split(b"\r\n")[0]


def test_non_finite_numbers_become_null_not_invalid_json(tmp_path):
    import kkmeshcut_serve as serve
    tools = serve.Tools(serve.Sandbox(tmp_path))
    tools.info = lambda a: {"volume": float("nan"), "area": float("inf"), "n": 3}
    got = serve.call_tool(tools, "info", {})
    assert got == {"volume": None, "area": None, "n": 3}


def test_unknown_argument_is_refused_instead_of_ignored(tmp_path):
    import kkmeshcut_serve as serve
    tools = serve.Tools(serve.Sandbox(tmp_path))
    with pytest.raises(mc.MeshcutError, match="nocap"):
        serve.call_tool(tools, "split", {"input": "a.stl", "outdir": "o", "nocap": True})


def test_segment_refuses_zero_parts_like_the_command_line(tmp_path):
    import kkmeshcut_serve as serve
    tools = serve.Tools(serve.Sandbox(tmp_path))
    with pytest.raises(mc.MeshcutError, match="parts"):
        serve.call_tool(tools, "segment", {"input": "a.stl", "outdir": "o", "parts": 0})


def test_unknown_swap_material_gets_a_hint(tmp_path):
    p = scene_obj(tmp_path)
    out = tmp_path / "o.obj"
    r = mc.swap(p, region=2, out=out, material="otehr")
    assert "otehr" in r["text"] and "Tippfehler" in r["text"]


def test_name_not_found_mentions_the_region_limit(tmp_path, monkeypatch):
    p = scene_obj(tmp_path)
    monkeypatch.setattr(ed, "MAX_REGIONS", 1)
    with pytest.raises(mc.MeshcutError, match="nicht nummeriert"):
        mc.remove(p, name="gibt-es-nicht", out=tmp_path / "o.obj")
