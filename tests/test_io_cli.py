import io
import json
import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import kkmeshcut as mc
import shapes

ROOT = Path(__file__).resolve().parents[1]


def cube_file(tmp_path, name="c.stl", fmt=None, **kw):
    m = shapes.cube(2.0, **kw)
    p = tmp_path / name
    p.write_bytes(mc.mesh_bytes(fmt or p.suffix[1:], m.V, m.F))
    return p, m


# ----------------------------------------------------------------------------- Formate
@pytest.mark.parametrize("fmt", mc.FORMATS)
def test_format_roundtrip(tmp_path, fmt):
    m = shapes.sphere(9, 6)
    p = tmp_path / ("m." + fmt)
    p.write_bytes(mc.mesh_bytes(fmt, m.V, m.F))
    back = mc.load_mesh(p)
    prec = "float32" if fmt in mc.FLOAT32_FORMATS else "float64"
    assert mc.triangle_hash(back.V, back.F, prec) == mc.triangle_hash(m.V, m.F, prec)


def test_ascii_stl_and_garbage(tmp_path):
    m = shapes.cube(1.0)
    txt = "solid x\n" + "".join("facet normal 0 0 0\nouter loop\n" + "".join("vertex %r %r %r\n" % tuple(float(x) for x in v) for v in m.V[f]) + "endloop\nendfacet\n" for f in m.F) + "endsolid x\n"
    (tmp_path / "a.stl").write_text(txt)
    assert mc.mesh_stats(*(lambda mm: (mm.V, mm.F))(mc.load_mesh(tmp_path / "a.stl")))["watertight"]
    (tmp_path / "bad.stl").write_bytes(b"not a mesh at all")
    with pytest.raises(mc.MeshcutError):
        mc.load_mesh(tmp_path / "bad.stl")
    (tmp_path / "t.stl").write_bytes(mc.stl_bytes(m.V, m.F)[:-20])      # abgeschnitten
    with pytest.raises(mc.MeshcutError):
        mc.load_mesh(tmp_path / "t.stl")
    with pytest.raises(mc.MeshcutError, match="gibt es nicht"):
        mc.load_mesh(tmp_path / "nope.stl")
    with pytest.raises(mc.MeshcutError, match="Ordner"):
        mc.load_mesh(tmp_path)
    (tmp_path / "a.xyz").write_text("x")
    with pytest.raises(mc.MeshcutError, match="endung"):
        mc.load_mesh(tmp_path / "a.xyz")


def test_nan_is_refused(tmp_path):
    m = shapes.cube(1.0)
    V = m.V.copy()
    V[0, 0] = np.nan
    (tmp_path / "n.stl").write_bytes(mc.stl_bytes(V, m.F))
    with pytest.raises(mc.MeshcutError, match="NaN"):
        mc.load_mesh(tmp_path / "n.stl")


def test_obj_variants(tmp_path):
    (tmp_path / "q.obj").write_text("# c\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\nf 1/1/1 2/2/2 3/3/3 4/4/4\nf -1 -4 -3\n")
    m = mc.load_mesh(tmp_path / "q.obj")
    assert len(m.F) == 3                                         # Viereck -> 2 Dreiecke + 1
    (tmp_path / "r.obj").write_text("v 0 0 0\nv 1 0 0\nf 1 2 9\n")
    with pytest.raises(mc.MeshcutError, match="nicht gibt"):
        mc.load_mesh(tmp_path / "r.obj")


def test_3mf_transform_and_components(tmp_path):
    m = shapes.cube(1.0)
    xml = ['<?xml version="1.0"?><model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>',
           '<object id="1"><mesh><vertices>']
    xml += ['<vertex x="%r" y="%r" z="%r"/>' % tuple(float(x) for x in v) for v in m.V]
    xml += ['</vertices><triangles>'] + ['<triangle v1="%d" v2="%d" v3="%d"/>' % tuple(f) for f in m.F]
    xml += ['</triangles></mesh></object><object id="2"><components><component objectid="1" transform="1 0 0 0 1 0 0 0 1 10 0 0"/></components></object>',
            '</resources><build><item objectid="2" transform="2 0 0 0 2 0 0 0 2 0 5 0"/></build></model>']
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as z:
        z.writestr("3D/3dmodel.model", "".join(xml))
    (tmp_path / "t.3mf").write_bytes(bio.getvalue())
    t = mc.load_mesh(tmp_path / "t.3mf")
    assert np.allclose(t.V.min(0), [20, 5, 0]) and np.allclose(t.V.max(0), [22, 7, 2])


def test_glb_node_transform(tmp_path):
    m = shapes.cube(1.0)
    data = bytearray(mc.glb_bytes([("c", m.V, m.F)]))
    # Knoten verschieben: JSON-Teil austauschen
    jl = struct.unpack("<I", data[12:16])[0]
    js = json.loads(bytes(data[20:20 + jl]).decode())
    js["nodes"][0]["translation"] = [0, 0, 7]
    js["nodes"][0]["scale"] = [3, 3, 3]
    jb = json.dumps(js).encode()
    jb += b" " * ((-len(jb)) % 4)
    rest = bytes(data[20 + jl:])
    out = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(jb) + len(rest)) + struct.pack("<II", len(jb), 0x4E4F534A) + jb + rest
    (tmp_path / "g.glb").write_bytes(out)
    g = mc.load_mesh(tmp_path / "g.glb")
    assert np.allclose(g.V.min(0), [0, 0, 7]) and np.allclose(g.V.max(0), [3, 3, 10])


def test_multi_object_3mf_parts_are_separate_objects(tmp_path):
    p, m = cube_file(tmp_path)
    side = mc.split(p, tmp_path / "o", grid="2x1x1", format="3mf")
    assert [q["file"] for q in side["parts"]] == ["parts.3mf"] * 2
    assert len(mc.read_meshes(tmp_path / "o" / "parts.3mf")) == 2
    assert mc.verify(tmp_path / "o", source=p)["ok"]


# ----------------------------------------------------------------------------- Ausgabeordner
def snapshot(folder):
    return {f.name: f.read_bytes() for f in Path(folder).iterdir() if f.is_file()}


def leftovers(folder):
    return [f.name for f in Path(folder).iterdir() if f.name.startswith(".meshcut_")]


def test_foreign_file_is_never_touched(tmp_path):
    p, _ = cube_file(tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    (out / "part_0.stl").write_bytes(b"mine")
    with pytest.raises(mc.MeshcutError, match="nicht von meshcut"):
        mc.split(p, out, grid="2x1x1")
    assert (out / "part_0.stl").read_bytes() == b"mine" and not (out / SIDECAR).exists() and not leftovers(out)
    (out / "meshcut.json").write_text("{}")
    (out / "part_0.stl").unlink()
    with pytest.raises(mc.MeshcutError, match="keine meshcut"):
        mc.split(p, out, grid="2x1x1")


SIDECAR = mc.SIDECAR


@pytest.mark.parametrize("k", range(1, 7))
def test_crash_during_install_restores_previous_set(tmp_path, monkeypatch, k):
    p, _ = cube_file(tmp_path)
    out = tmp_path / "o"
    mc.split(p, out, grid="3x1x1")
    before = snapshot(out)
    real = os.replace
    calls = {"n": 0}

    def flaky(a, b, *x, **y):
        calls["n"] += 1
        if calls["n"] == k:
            raise OSError("Absturz Nr. %d" % k)
        return real(a, b, *x, **y)
    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises((OSError, mc.MeshcutError)):
        mc.split(p, out, grid="2x1x1")
    monkeypatch.undo()
    assert snapshot(out) == before and not leftovers(out)


def test_rerun_replaces_set_and_drops_stale_parts_with_readonly(tmp_path):
    p, _ = cube_file(tmp_path)
    out = tmp_path / "o"
    mc.split(p, out, grid="3x1x1")
    os.chmod(out / "part_2.stl", 0o444)
    mc.split(p, out, grid="2x1x1")
    assert sorted(f.name for f in out.iterdir()) == ["meshcut.json", "part_0.stl", "part_1.stl"]
    assert mc.verify(out, source=p)["ok"]
    mc.split(p, out, grid="2x1x1", format="obj")                 # anderes Format: alte STL-Teile gehoeren dazu und verschwinden
    assert sorted(f.name for f in out.iterdir()) == ["meshcut.json", "part_0.obj", "part_1.obj"]


def test_input_inside_output_is_refused(tmp_path):
    p, _ = cube_file(tmp_path, "part_0.stl")
    with pytest.raises(mc.MeshcutError):
        mc.split(p, tmp_path, grid="2x1x1")
    assert p.is_file()


def test_output_must_not_be_a_file_or_link(tmp_path, monkeypatch):
    p, _ = cube_file(tmp_path)
    (tmp_path / "f").write_text("x")
    with pytest.raises(mc.MeshcutError, match="kein Ordner"):
        mc.split(p, tmp_path / "f", grid="2x1x1")
    (tmp_path / "L").mkdir()
    monkeypatch.setattr(mc, "link_like", lambda x: Path(x).name == "L")
    with pytest.raises(mc.MeshcutError, match="Link"):
        mc.split(p, tmp_path / "L", grid="2x1x1")


def test_join_output_cannot_overwrite_input_and_is_atomic(tmp_path):
    p, _ = cube_file(tmp_path)
    mc.split(p, tmp_path / "o", grid="2x1x1")
    files = sorted((tmp_path / "o").glob("part_*.stl"))
    with pytest.raises(mc.MeshcutError, match="auch Eingabe"):
        mc.join(files, files[0])
    j = tmp_path / "j.stl"
    mc.join(tmp_path / "o", j)
    assert j.is_file() and not leftovers(tmp_path)


def test_tampered_sidecar_and_missing_part(tmp_path):
    p, _ = cube_file(tmp_path)
    mc.split(p, tmp_path / "o", grid="2x1x1")
    (tmp_path / "o" / "part_1.stl").unlink()
    with pytest.raises(mc.MeshcutError, match="fehlt"):
        mc.join(tmp_path / "o", tmp_path / "j.stl")
    side = json.loads((tmp_path / "o" / "meshcut.json").read_text())
    side["parts"][0]["file"] = "../evil.stl"
    (tmp_path / "o" / "meshcut.json").write_text(json.dumps(side))
    with pytest.raises(mc.MeshcutError, match="Begleitdatei"):
        mc.join(tmp_path / "o", tmp_path / "j.stl")


# ----------------------------------------------------------------------------- Schnittstelle und Kommandozeile
def test_api_errors_and_silence(tmp_path, capsys):
    p, _ = cube_file(tmp_path)
    mc.split(p, tmp_path / "o", plane=["x=1"], verify=True)
    mc.join(tmp_path / "o", tmp_path / "j.stl")
    mc.verify(tmp_path / "o")
    mc.info(p)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    for bad in (dict(), dict(grid="2x2", plane=["x=1"]), dict(plane=["q=3"]), dict(grid="0x1x1"), dict(wedges=3), dict(max="a")):
        with pytest.raises(mc.MeshcutError):
            mc.split(p, tmp_path / "b", **bad)
    with pytest.raises(mc.MeshcutError):
        mc.split(p, tmp_path / "b", bogus=True)                  # unbekannte Option: kein SystemExit
    with pytest.raises(mc.MeshcutError):
        mc.split(tmp_path / "missing.stl", tmp_path / "b", grid="2x1x1")


def test_negative_plane_value_and_flags(tmp_path):
    m = shapes.cube(2.0, lo=(-1, -1, -1))
    p = tmp_path / "c.stl"
    p.write_bytes(mc.mesh_bytes("stl", m.V, m.F))
    side = mc.split(p, tmp_path / "o", plane=["x=-0.5"])
    assert len(side["parts"]) == 2


def test_max_gives_parts_that_fit(tmp_path):
    m = shapes.cube(10.0)
    p = tmp_path / "c.stl"
    p.write_bytes(mc.mesh_bytes("stl", m.V, m.F))
    side = mc.split(p, tmp_path / "o", max="4x4x20")
    assert len(side["parts"]) == 9
    for q in side["parts"]:
        part = mc.load_mesh(tmp_path / "o" / q["file"])
        assert (part.V.max(0) - part.V.min(0) <= [4.0000001, 4.0000001, 20]).all()
    with pytest.raises(mc.MeshcutError, match="passt schon"):
        mc.split(p, tmp_path / "o2", max="50x50x50")


def test_labels_command_and_grid_labels(tmp_path):
    p, m = cube_file(tmp_path, sub=2)
    mc.labels(p, tmp_path / "l.npy", grid="2x2x1")
    lab = np.load(tmp_path / "l.npy")
    assert lab.shape == (len(m.F),) and set(lab.tolist()) == {0, 1, 2, 3}
    side = mc.split(p, tmp_path / "o", labels=tmp_path / "l.npy")
    assert side["mode"] == "labels" and len(side["parts"]) == 4
    assert mc.verify(tmp_path / "o", source=p)["ok"]
    for bad in ([1, 2, 3], [0.5] * len(m.F), [-1] * len(m.F), [0] * len(m.F)):
        np.save(tmp_path / "bad.npy", np.array(bad))
        with pytest.raises(mc.MeshcutError):
            mc.split(p, tmp_path / "ob", labels=tmp_path / "bad.npy")


def run_cli(*args, cwd=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, str(ROOT / "kkmeshcut.py"), *args], capture_output=True, text=True, cwd=cwd, env=env, encoding="utf-8")


def test_command_line_end_to_end(tmp_path):
    p, _ = cube_file(tmp_path)
    r = run_cli("split", str(p), str(tmp_path / "o"), "--grid", "2x2x2", "--verify")
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout and "FEHLER" not in r.stdout
    r = run_cli("join", str(tmp_path / "o"), str(tmp_path / "j.obj"))
    assert r.returncode == 0 and "=" in r.stdout
    r = run_cli("info", str(p))
    assert r.returncode == 0 and "Wasserdicht:  ja" in r.stdout
    r = run_cli("verify", str(tmp_path / "o"), "--source", str(p))
    assert r.returncode == 0 and "Alles in Ordnung" in r.stdout
    r = run_cli("split", str(p), str(tmp_path / "o3"))
    assert r.returncode != 0 and "Fehler" in r.stderr and "Traceback" not in r.stderr
    r = run_cli("split", str(tmp_path / "nope.stl"), str(tmp_path / "o3"), "--grid", "2x1x1")
    assert r.returncode != 0 and "gibt es nicht" in r.stderr and "Traceback" not in r.stderr
    r = run_cli("--version")
    assert mc.__version__ in r.stdout


def test_browser_copy_is_identical():
    docs = ROOT / "docs" / "kkmeshcut.py"
    assert docs.read_bytes() == (ROOT / "kkmeshcut.py").read_bytes(), "docs/kkmeshcut.py ist veraltet - Kopie erneuern"
