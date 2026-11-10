"""Regressionstests zu den Befunden der Pruefrunden: jeder Test hat vorher eine falsche Erfolgsmeldung oder einen Absturz gezeigt."""
import base64
import json
import os
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

import kkmeshcut as mc
import shapes

ROOT = Path(__file__).resolve().parents[1]


def split_to(tmp_path, mesh=None, fmt="obj", **opts):
    mesh = mesh or shapes.sphere(14, 9, r=5.0)
    src = tmp_path / "src.obj"
    src.write_bytes(mc.mesh_bytes("obj", mesh.V, mesh.F))
    out = tmp_path / "out"
    if not any(k in opts for k in ("plane", "grid", "max", "wedges", "labels")):
        opts["plane"] = ["x=0.4", "z=0.3"]
    mc.split(src, out, format=fmt, **opts)
    return src, out


def failed(r):
    return [c[0] for c in r["checks"] if not c[1]]


# ----------------------------------------------------------------------------- verify darf keine Beschaedigung durchlassen
def test_verify_detects_translated_parts(tmp_path):
    src, out = split_to(tmp_path)
    assert mc.verify(out)["ok"]
    for f in out.glob("part_*.obj"):
        lines = f.read_text().splitlines()
        f.write_text("\n".join("v %r %r %r" % (float(q[1]) + 100, float(q[2]), float(q[3])) if (q := l.split())[0] == "v" else l for l in lines) + "\n")
    r = mc.verify(out)
    assert not r["ok"] and "teile" in failed(r)
    assert not mc.verify(out, source=src)["ok"]


def test_verify_detects_a_replaced_triangle(tmp_path):
    src, out = split_to(tmp_path, shapes.cube(2.0), grid="2x2x2", no_cap=True)
    part = out / "part_0.obj"
    lines = part.read_text().splitlines()
    fi = [i for i, l in enumerate(lines) if l.startswith("f ")]
    lines[fi[1]] = lines[fi[0]]                                  # ein Dreieck durch ein anderes gleicher Flaeche ersetzen
    part.write_text("\n".join(lines) + "\n")
    r = mc.verify(out, source=src)
    assert not r["ok"] and "teile" in failed(r)


def test_verify_detects_cuts_off_the_requested_plane(tmp_path):
    """Die Teile stimmen mit ihrem Hash, aber die gespeicherten Ebenen passen nicht zu den Deckeln (Schnitt an falscher Stelle)."""
    src, out = split_to(tmp_path)
    side = json.loads((out / "meshcut.json").read_text())
    for p in side["planes"]:
        p["offset"] += 0.25
    (out / "meshcut.json").write_text(json.dumps(side))
    r = mc.verify(out)
    assert not r["ok"] and "ebenen" in failed(r)


def test_split_verify_names_the_failing_check_in_the_message(tmp_path, monkeypatch):
    real = mc.verify_folder

    def broken(folder, source=None):
        r = real(folder, source)
        r["checks"].append(("ebenen", False, "x"))
        r["ok"] = False
        return r
    monkeypatch.setattr(mc, "verify_folder", broken)
    src = tmp_path / "s.obj"
    m = shapes.cube(2.0)
    src.write_bytes(mc.mesh_bytes("obj", m.V, m.F))
    with pytest.raises(mc.MeshcutError, match="fehlgeschlagen: ebenen"):
        mc.split(src, tmp_path / "o", plane=["x=1"], verify=True)


# ----------------------------------------------------------------------------- join
def test_join_never_overwrites_its_own_input_folder(tmp_path):
    src, out = split_to(tmp_path)
    before = {f.name: f.read_bytes() for f in out.iterdir()}
    with pytest.raises(mc.MeshcutError, match="gehoert zu den Teilen"):
        mc.join(out, out / "part_0.obj")
    with pytest.raises(mc.MeshcutError, match="gehoert zu den Teilen"):
        mc.join([out], out / "PART_0.obj" if os.name == "nt" else out / "part_0.obj")
    assert {f.name: f.read_bytes() for f in out.iterdir()} == before
    assert mc.verify(out)["ok"]


# ----------------------------------------------------------------------------- Formate
def test_3mf_units_are_converted_to_millimeters(tmp_path):
    m = shapes.cube(1.0)
    data = mc.threemf_bytes([("c", m.V, m.F)])
    z = zipfile.ZipFile(__import__("io").BytesIO(data))
    model = z.read("3D/3dmodel.model").decode().replace('unit="millimeter"', 'unit="inch"')
    bio = __import__("io").BytesIO()
    with zipfile.ZipFile(bio, "w") as w:
        for n in z.namelist():
            w.writestr(n, model if n == "3D/3dmodel.model" else z.read(n))
    (tmp_path / "inch.3mf").write_bytes(bio.getvalue())
    back = mc.load_mesh(tmp_path / "inch.3mf")
    assert np.allclose(back.V.max(0) - back.V.min(0), 25.4)
    bad = model.replace('unit="inch"', 'unit="parsec"')
    bio = __import__("io").BytesIO()
    with zipfile.ZipFile(bio, "w") as w:
        for n in z.namelist():
            w.writestr(n, bad if n == "3D/3dmodel.model" else z.read(n))
    (tmp_path / "bad.3mf").write_bytes(bio.getvalue())
    with pytest.raises(mc.MeshcutError, match="Einheit"):
        mc.load_mesh(tmp_path / "bad.3mf")


def test_concave_obj_polygon_keeps_its_area(tmp_path):
    # U-foermiges Polygon, Flaeche 7
    pts = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]
    txt = "".join("v %g %g 0\n" % p for p in pts) + "f " + " ".join(str(i + 1) for i in range(len(pts))) + "\n"
    (tmp_path / "u.obj").write_text(txt)
    objs = mc.read_obj(txt.encode())
    P = objs[0][1]
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1).sum()
    assert abs(area - 7.0) < 1e-9


def test_obj_with_indented_lines_and_uppercase_ascii_stl(tmp_path):
    m = shapes.cube(1.0)
    txt = "o c\n" + "".join("\tv %r %r %r\n" % tuple(float(x) for x in v) for v in m.V) + "".join("  f %d %d %d\n" % tuple(i + 1 for i in f) for f in m.F)
    (tmp_path / "i.obj").write_text(txt)
    assert len(mc.load_mesh(tmp_path / "i.obj").F) == 12
    stl = "SOLID X\n" + "".join("FACET NORMAL 0 0 0\nOUTER LOOP\n" + "".join("VERTEX %r %r %r\n" % tuple(float(x) for x in v) for v in m.V[f]) + "ENDLOOP\nENDFACET\n" for f in m.F) + "ENDSOLID X\n"
    (tmp_path / "u.stl").write_text(stl)
    assert len(mc.load_mesh(tmp_path / "u.stl").F) == 12


def _glb_with_node(node_extra):
    m = shapes.cube(2.0)
    data = bytearray(mc.glb_bytes([("c", m.V, m.F)]))
    ln = struct.unpack("<I", data[12:16])[0]
    js = json.loads(bytes(data[20:20 + ln]).decode())
    js["nodes"][0].update(node_extra)
    jb = json.dumps(js, separators=(",", ":")).encode()
    jb += b" " * ((-len(jb)) % 4)
    rest = bytes(data[20 + ln:])
    total = 12 + 8 + len(jb) + len(rest)
    return b"glTF" + struct.pack("<II", 2, total) + struct.pack("<II", len(jb), 0x4E4F534A) + jb + rest


def test_glb_mirrored_node_keeps_outward_orientation(tmp_path):
    (tmp_path / "m.glb").write_bytes(_glb_with_node({"scale": [-1, 1, 1]}))
    m = mc.load_mesh(tmp_path / "m.glb")
    assert mc.volume_area(m.V, m.F)[0] == pytest.approx(8.0)
    assert mc.mesh_stats(m.V, m.F)["watertight"]


# ----------------------------------------------------------------------------- Gitter, Zahlen, Grenzen
def test_cli_grid_keeps_cell_numbers(tmp_path):
    a = shapes.cube(1.0)
    b = shapes.cube(1.0, lo=(3, 0, 0))
    V, F = mc.weld_triangles(np.concatenate([a.V[a.F], b.V[b.F]]))
    src, out = split_to(tmp_path, mc.Mesh(V, F), grid="4x1x1")
    side = json.loads((out / "meshcut.json").read_text())
    assert sorted(p["id"] for p in side["parts"]) == [0, 3]
    assert mc.verify(out, source=src)["ok"]


def test_grid_and_max_limits_are_checked_before_cutting(tmp_path):
    src = tmp_path / "c.obj"
    m = shapes.cube(2.0)
    src.write_bytes(mc.mesh_bytes("obj", m.V, m.F))
    for opts in ({"grid": "40x40x1"}, {"grid": "99999x99999x99999"}, {"max": "0.0001x0.0001x0.0001"}):
        with pytest.raises(mc.MeshcutError, match="Zellen"):
            mc.split(src, tmp_path / "o", **opts)


def test_sidecar_arrays_cannot_blow_up_memory():
    ok = mc.enc_array(np.arange(6, dtype=np.int64))
    assert mc.dec_array(ok).tolist() == list(range(6))
    bomb = {"dtype": "<f8", "shape": [4], "b64": base64.b64encode(zlib.compress(b"\0" * (1 << 26))).decode()}
    with pytest.raises(mc.MeshcutError, match="beschaedigt"):
        mc.dec_array(bomb)
    huge = dict(ok, shape=[1 << 40])
    with pytest.raises(mc.MeshcutError, match="beschaedigt"):
        mc.dec_array(huge)
    with pytest.raises(mc.MeshcutError, match="beschaedigt"):
        mc.dec_array(dict(ok, dtype="|O"))


@pytest.mark.parametrize("offset,fmt", [(0.0, "obj"), (1e6, "obj"), (1e8, "obj"), (1e10, "obj"), (0.0, "stl"), (1e4, "stl"), (1e6, "stl")])
def test_far_from_the_origin(tmp_path, offset, fmt):
    m = shapes.cube(1.0, lo=(offset, offset, offset))
    src = tmp_path / ("s." + fmt)
    src.write_bytes(mc.mesh_bytes(fmt, m.V, m.F))
    out = tmp_path / "o"
    mc.split(src, out, plane=["x=%r" % (offset + 0.5)], format=fmt)
    r = mc.verify(out, source=src)
    assert r["ok"], r["checks"]


# ----------------------------------------------------------------------------- Python-Schnittstelle
def test_api_always_raises_meshcut_error(tmp_path):
    bad = tmp_path / "bad.3mf"
    bio = __import__("io").BytesIO()
    with zipfile.ZipFile(bio, "w") as w:
        w.writestr("3D/3dmodel.model", "<model><oops")
    bad.write_bytes(bio.getvalue())
    with pytest.raises(mc.MeshcutError):
        mc.info(bad)
    with pytest.raises(mc.MeshcutError):
        mc.info(None)
    (tmp_path / "e.glb").write_bytes(b"glTF" + struct.pack("<II", 2, 40) + struct.pack("<II", 2, 0x4E4F534A) + b"[]  " + struct.pack("<II", 0, 0x004E4942))
    with pytest.raises(mc.MeshcutError):
        mc.load_mesh(tmp_path / "e.glb")
    m = shapes.cube(2.0)
    src = tmp_path / "c.obj"
    src.write_bytes(mc.mesh_bytes("obj", m.V, m.F))
    with pytest.raises(mc.MeshcutError):
        mc.split(src, tmp_path / "o", help=True)
    mc.split(src, tmp_path / "o", plane=(1, 0, 0, 1))              # vier Zahlen = EINE Ebene
    assert len(json.loads((tmp_path / "o" / "meshcut.json").read_text())["planes"]) == 1


# ----------------------------------------------------------------------------- harter Abbruch beim Ersetzen
CRASH = """
import os, sys
import kkmeshcut as mc
n = int(sys.argv[3])
count = [0]
real = os.replace
def replace(a, b, *x, **k):
    count[0] += 1
    if count[0] == n:
        os._exit(77)
    return real(a, b, *x, **k)
os.replace = replace
mc.main(["split", sys.argv[1], sys.argv[2], "--plane", "z=0.6", "--plane", "y=0.1"])
"""


@pytest.mark.parametrize("n", range(1, 14))
def test_hard_kill_during_replace_restores_a_consistent_set(tmp_path, n):
    src, out = split_to(tmp_path, plane=["x=0.4", "z=0.3"])
    driver = tmp_path / "crash.py"
    driver.write_text(CRASH)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, str(driver), str(src), str(out), str(n)], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert r.returncode in (0, 77), r.stderr
    for d in out.iterdir():                      # der Absturz liegt laenger zurueck: erst dann raeumen auch lesende Befehle auf
        if d.name.startswith(".meshcut_"):
            os.utime(d, (1, 1))
    v = mc.verify(out, source=src)               # stellt bei Bedarf zurueck; danach muss der Satz vollstaendig und in Ordnung sein
    assert v["ok"], (n, r.returncode, v["checks"])
    assert not [d for d in out.iterdir() if d.name.startswith(".meshcut_")] or r.returncode == 77
    if r.returncode == 77:
        names = {f.name for f in out.iterdir() if not f.name.startswith(".")}
        side = json.loads((out / "meshcut.json").read_text())
        assert names == {p["file"] for p in side["parts"]} | {"meshcut.json"}


@pytest.mark.parametrize("order", [("z=0.6", "y=0.1"), ("y=0.1", "z=0.6")])
def test_collinear_filler_triangles_do_not_duplicate_vertices(order):
    """Eine Ebene schneidet ein flaches Fuelldreieck (drei Punkte auf einer Linie) des ersten Deckels: der Schnittpunkt darf nur
    einmal entstehen, sonst zerfaellt das Teil beim Speichern in float32 (zwei Eckpunkte an derselben Stelle)."""
    m = shapes.sphere(14, 9, r=5.0)
    res = mc.split_planes(m, [mc.parse_plane(o) for o in order])
    assert res.report["cap_emergencies"] == 0
    for p in res.parts:
        back = mc.read_meshes_bytes("stl", mc.stl_bytes(p.V, p.F), "x")[0]
        assert mc.edge_report(back.F, len(back.V))["watertight"]
        assert len(back.V) == len(p.V)
