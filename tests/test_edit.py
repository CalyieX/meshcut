"""regions / remove / swap / tools: Elemente waehlen, entfernen, ersetzen - mit Material, UV und Normalen, und die Pruefung am Ende."""
import json
import math
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


def test_remove_non_planar_hole_gets_a_centre_fan_and_is_watertight(tmp_path):
    """Drei Seiten eines Quaders weg: der Rand ist ein windschiefes Sechseck."""
    h, b = cube_lines((0, 0, 0), (2, 2, 2), 0, 0, 0, ["a", "b", "c", "a", "b", "c"], "w")
    out, fi = [], 0
    for ln in h + b:
        if ln.startswith("f "):
            out.append("g " + ("weg" if fi in (1, 3, 4) else "rest"))
            fi += 1
        out.append(ln)
    p = tmp_path / "w.obj"
    p.write_text("\n".join(out) + "\n")
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
    assert "nur Geometrie" in r["text"]
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
