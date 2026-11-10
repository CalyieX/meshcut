"""Gezielte Tests gegen Mutanten, die die Pruefrunden uebersehen haben (siehe tests/mutate.py): jede Pruefung muss auch dann
anschlagen, wenn alle anderen sauber sind."""
import base64
import json
import os
import struct
import subprocess
import sys
import tracemalloc
import zlib
from pathlib import Path

import numpy as np
import pytest

import kkmeshcut as mc
import shapes
from test_review import CRASH, ROOT, failed, split_to


def write_obj(path, mesh):
    Path(path).write_bytes(mc.mesh_bytes("obj", mesh.V, mesh.F))
    return Path(path)


# ----------------------------------------------------------------------------- Begleitdatei-Arrays
def test_dec_array_size_limit_is_checked_before_decompressing(monkeypatch):
    monkeypatch.setattr(mc, "MAX_ARRAY_BYTES", 64)
    assert mc.dec_array(mc.enc_array(np.arange(4, dtype=np.int64))).tolist() == [0, 1, 2, 3]
    with pytest.raises(mc.MeshcutError, match="beschaedigt"):
        mc.dec_array(mc.enc_array(np.arange(16, dtype=np.int64)))


def test_dec_array_never_unpacks_more_than_promised():
    bomb = {"dtype": "<f8", "shape": [4], "b64": base64.b64encode(zlib.compress(b"\0" * (1 << 27))).decode()}
    tracemalloc.start()
    try:
        with pytest.raises(mc.MeshcutError, match="beschaedigt"):
            mc.dec_array(bomb)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 16 << 20, peak


@pytest.mark.parametrize("dtype", ["<f4", ">i8", "<i4", "|u1"])
def test_dec_array_accepts_only_the_two_written_dtypes(dtype):
    a = np.arange(4, dtype=np.dtype(dtype))
    d = {"dtype": a.dtype.str, "shape": [4], "b64": base64.b64encode(zlib.compress(a.tobytes())).decode()}
    with pytest.raises(mc.MeshcutError, match="beschaedigt"):
        mc.dec_array(d)


# ----------------------------------------------------------------------------- Eingabepruefung
def test_truncated_ascii_stl_is_reported_as_truncated():
    text = b"solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nvertex 1 1 0\nendloop\nendfacet\nendsolid x\n"
    with pytest.raises(mc.MeshcutError, match="abgeschnitten"):
        mc.read_stl(text)


def test_nan_coordinates_are_refused_on_load(tmp_path):
    m = shapes.cube(1.0)
    P = m.V[m.F].astype(np.float32)
    P[0, 0, 0] = np.nan
    rec = np.zeros(len(P), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["v"] = P
    (tmp_path / "nan.stl").write_bytes(b"x" * 80 + struct.pack("<I", len(P)) + rec.tobytes())
    with pytest.raises(mc.MeshcutError, match=r"nan\.stl enthaelt NaN"):
        mc.load_mesh(tmp_path / "nan.stl")


def test_negative_part_numbers_are_refused():
    m = shapes.cube(1.0)
    labels = np.zeros(len(m.F), np.int64)
    labels[:3] = -1
    with pytest.raises(mc.MeshcutError, match="negativ"):
        mc.split_labels(m, labels)


def test_inward_oriented_mesh_is_refused_with_a_hint(tmp_path):
    m = shapes.cube(2.0)
    src = tmp_path / "in.obj"
    src.write_bytes(mc.mesh_bytes("obj", m.V, m.F[:, ::-1]))
    with pytest.raises(mc.MeshcutError, match="negatives Volumen"):
        mc.split(src, tmp_path / "o", plane=["x=1"])


def test_edge_report_counts_doubled_faces_as_bad():
    m = shapes.cube(1.0)
    F = np.concatenate([m.F, m.F[:1]])                          # ein Dreieck doppelt: jede Kante hat ihr Gegenstueck, aber eine ist doppelt
    er = mc.edge_report(F, len(m.V))
    assert er["open"] == 0 and er["bad"] > 0 and not er["watertight"]


def test_plane_with_negative_normal_keeps_its_position(tmp_path):
    m = shapes.cube(2.0, lo=(-1, -1, -1))
    src = write_obj(tmp_path / "c.obj", m)
    mc.split(src, tmp_path / "o", plane=(-2.0, 0.0, 0.0, 1.0), format="obj")           # -2x = 1  ->  x = -0.5
    xs = set()
    for f in (tmp_path / "o").glob("part_*.obj"):
        for line in f.read_text().splitlines():
            if line.startswith("v "):
                xs.add(round(float(line.split()[1]), 9))
    assert -0.5 in xs and 0.5 not in xs


def test_glb_triangle_strips_are_refused_by_name(tmp_path):
    m = shapes.cube(1.0)
    data = bytearray(mc.glb_bytes([("c", m.V, m.F)]))
    jlen = struct.unpack_from("<I", data, 12)[0]
    js = json.loads(bytes(data[20:20 + jlen]).decode())
    for pr in js["meshes"][0]["primitives"]:
        pr["mode"] = 5
    new = json.dumps(js).encode()
    new += b" " * (-len(new) % 4)
    rest = bytes(data[20 + jlen:])
    out = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(new) + len(rest)) + struct.pack("<II", len(new), 0x4E4F534A) + new + rest
    (tmp_path / "s.glb").write_bytes(out)
    with pytest.raises(mc.MeshcutError, match="Dreiecksstreifen"):
        mc.load_mesh(tmp_path / "s.glb")


# ----------------------------------------------------------------------------- Python-Schnittstelle
@pytest.mark.parametrize("opt", ["help", "h", "version"])
def test_api_rejects_help_and_version(tmp_path, opt):
    src = write_obj(tmp_path / "c.obj", shapes.cube(2.0))
    with pytest.raises(mc.MeshcutError, match="gibt es in der Python-Schnittstelle nicht"):
        mc.split(src, tmp_path / "o", **{opt: True})


def test_api_turns_argparse_errors_into_meshcut_error(tmp_path):
    src = write_obj(tmp_path / "c.obj", shapes.cube(2.0))
    with pytest.raises(mc.MeshcutError):
        mc.split(src, tmp_path / "o", plane=["x=1"], nonsense_option=3)


# ----------------------------------------------------------------------------- join
def test_join_detects_a_part_with_a_removed_triangle(tmp_path):
    src, out = split_to(tmp_path)
    mid = out / "part_1.obj"
    lines = mid.read_text().splitlines()
    k = max(i for i, l in enumerate(lines) if l.startswith("f "))
    del lines[k]
    mid.write_text("\n".join(lines) + "\n")
    with pytest.raises(mc.MeshcutError, match="veraendert"):
        mc.join(out, tmp_path / "back.obj")


def test_join_checks_the_triangle_count_of_the_original(tmp_path):
    src, out = split_to(tmp_path)
    side = json.loads((out / "meshcut.json").read_text())
    side["source"]["faces"] += 1
    (out / "meshcut.json").write_text(json.dumps(side))
    with pytest.raises(mc.MeshcutError, match="ergeben nicht alle"):
        mc.join(out, tmp_path / "back.obj")


@pytest.mark.skipif(not hasattr(os, "link"), reason="keine harten Links")
def test_join_refuses_a_hard_link_to_one_of_its_parts(tmp_path):
    src, out = split_to(tmp_path)
    alias = tmp_path / "alias.obj"
    try:
        os.link(out / "part_0.obj", alias)
    except OSError:
        pytest.skip("harte Links nicht moeglich")
    before = alias.read_bytes()
    with pytest.raises(mc.MeshcutError, match="gehoert zu den Teilen"):
        mc.join(out, alias)
    assert alias.read_bytes() == before


# ----------------------------------------------------------------------------- verify: jede Pruefung einzeln
def edit_sidecar(out, fn):
    p = out / "meshcut.json"
    side = json.loads(p.read_text())
    fn(side)
    p.write_text(json.dumps(side))


def test_verify_hash_check_alone(tmp_path):
    src, out = split_to(tmp_path)
    edit_sidecar(out, lambda s: s["source"].update(hash="0" * 64))
    assert failed(mc.verify(out)) == ["hash"]


def test_verify_volume_check_alone(tmp_path):
    src, out = split_to(tmp_path)
    edit_sidecar(out, lambda s: s["source"].update(volume=s["source"]["volume"] * 1.5))
    assert failed(mc.verify(out)) == ["volumen"]


def test_verify_area_check_alone(tmp_path):
    src, out = split_to(tmp_path)
    edit_sidecar(out, lambda s: s["source"].update(area=s["source"]["area"] * 1.5))
    assert failed(mc.verify(out)) == ["flaeche"]


def test_verify_watertight_check_after_a_consistent_edit(tmp_path):
    """Ein Deckel-Dreieck fehlt, und Begleitdatei samt Hash wurden passend nachgezogen: nur die Teile selbst verraten es."""
    src, out = split_to(tmp_path)
    side = json.loads((out / "meshcut.json").read_text())
    k = next(i for i, m in enumerate(side["parts"]) if m["cap_faces"])
    pid, V, F = [p for p in mc.load_parts(out)[1]][k]
    F2 = F[:-1]
    (out / side["parts"][k]["file"]).write_bytes(mc.mesh_bytes("obj", V, F2))
    side["parts"][k].update(hash=mc.triangle_hash(V, F2, side["source"]["precision"]), faces=len(F2), cap_faces=side["parts"][k]["cap_faces"] - 1)
    (out / "meshcut.json").write_text(json.dumps(side))
    assert "wasserdicht" in failed(mc.verify(out))


# ----------------------------------------------------------------------------- Ausgabeordner
def test_rollback_after_a_failure_while_installing(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    for n in ("a", "b", "c"):
        (stage / n).write_text("neu " + n)
    (out / "b").write_text("alt b")
    real = os.replace

    def replace(src, dst, *a, **k):
        if Path(src).parent == stage and Path(src).name == "c":
            raise OSError("Platte voll")
        return real(src, dst, *a, **k)
    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(OSError):
        mc.publish_files(stage, ["a", "b", "c"], out)
    monkeypatch.undo()
    assert sorted(f.name for f in out.iterdir()) == ["b"]
    assert (out / "b").read_text() == "alt b"


def test_a_finished_journal_is_not_rolled_back(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "part_0.obj").write_text("neu")
    (out / "part_0.obj").write_text("alt")

    def boom(folder):
        raise KeyboardInterrupt
    monkeypatch.setattr(mc, "rmtree_force", boom)
    with pytest.raises(KeyboardInterrupt):
        mc.publish_files(stage, ["part_0.obj"], out)
    monkeypatch.undo()
    mc.recover_outdir(out)
    assert (out / "part_0.obj").read_text() == "neu"
    assert not [d for d in out.iterdir() if d.name.startswith(".meshcut_")]


def test_prepare_outdir_refuses_an_input_that_is_one_of_the_outputs(tmp_path):
    src, out = split_to(tmp_path)
    with pytest.raises(mc.MeshcutError, match="eine der Dateien"):
        mc.prepare_outdir(out, ["part_0.obj", "meshcut.json"], [out / "part_0.obj"])


@pytest.mark.parametrize("n", range(1, 14))
def test_split_after_a_hard_kill_recovers_by_itself(tmp_path, n):
    src, out = split_to(tmp_path, plane=["x=0.4", "z=0.3"])
    driver = tmp_path / "crash.py"
    driver.write_text(CRASH)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, str(driver), str(src), str(out), str(n)], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert r.returncode in (0, 77), r.stderr
    mc.split(src, out, plane=["z=0.6", "y=0.1"])                 # ohne vorheriges verify
    assert mc.verify(out, source=src)["ok"]
    assert not [d for d in out.iterdir() if d.name.startswith(".meshcut_")]


# ----------------------------------------------------------------------------- Rechengenauigkeit
def test_float32_files_far_from_the_origin_pass_the_hausdorff_check(tmp_path):
    m = shapes.sphere(14, 9, r=5.0, c=(1e5, 1e5, 1e5))
    src = tmp_path / "s.stl"
    src.write_bytes(mc.mesh_bytes("stl", m.V, m.F))
    mc.split(src, tmp_path / "o", plane=["x=%r" % (1e5 + 0.3)], format="stl")
    r = mc.verify(tmp_path / "o", source=src)
    assert r["ok"], r["checks"]


def test_snap_is_relative_to_the_size_of_the_mesh():
    """Eckpunkte, die 1e-8 neben der Ebene liegen, bei einem Netz von 1e6 Groesse: gelten als auf der Ebene (kein Splitter)."""
    m = shapes.cube(1e6, sub=2)
    exact = mc.split_planes(m, [mc.parse_plane("x=500000")])
    near = mc.split_planes(m, [mc.parse_plane("x=%r" % (500000 + 1e-8))])
    assert sorted(len(p.F) for p in exact.parts) == sorted(len(p.F) for p in near.parts)


def test_snap_allows_for_the_rounding_of_the_coordinates_themselves():
    m = shapes.cube(1e6, sub=2)
    edge = 1e-12 * 1e6                                            # Schwelle aus snap * Groesse
    exact = mc.split_planes(m, [mc.parse_plane("x=500000")])
    near = mc.split_planes(m, [mc.parse_plane("x=%r" % (500000 + edge * 1.0009))])
    assert sorted(len(p.F) for p in exact.parts) == sorted(len(p.F) for p in near.parts)


def test_hash_depends_on_the_precision_it_was_made_with():
    m = shapes.cube(1.0)                                          # Koordinaten in float32 exakt: nur die Praezision selbst unterscheidet
    assert mc.triangle_hash(m.V, m.F, "float32") != mc.triangle_hash(m.V, m.F, "float64")


def test_non_watertight_part_gets_a_hint(tmp_path, monkeypatch, capsys):
    src = write_obj(tmp_path / "c.obj", shapes.cube(2.0))
    real = mc.edge_report

    def leaky(F, nv=None):
        r = dict(real(F, nv))
        if len(F) and nv is not None:
            r.update(watertight=False, open=2, bad=0)
        return r
    monkeypatch.setattr(mc, "edge_report", leaky)
    monkeypatch.setattr(mc, "check_cuttable", lambda *a, **k: None)
    mc.main(["split", str(src), str(tmp_path / "o"), "--plane", "x=1", "--format", "obj"])
    assert "ist nicht wasserdicht" in capsys.readouterr().out


def test_api_turns_a_system_exit_with_text_into_meshcut_error(tmp_path, monkeypatch):
    def quits(a):
        raise SystemExit("Fehler: kaputt")
    monkeypatch.setattr(mc, "cmd_split", quits)
    with pytest.raises(mc.MeshcutError) as e:
        mc.split(tmp_path / "x.obj", tmp_path / "o", plane=["x=1"])
    assert str(e.value) == "kaputt"


def test_zero_area_triangle_in_the_plane_goes_with_its_neighbours():
    """Ein flaches Dreieck in der Schnittebene hat keine Normale: es geht dorthin, wo seine Nachbarn liegen (hier: unter die Ebene)."""
    V = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, -1], [1, -1, -1], [5, 5, 1], [6, 5, 1], [5, 6, 1]], float)
    F = np.array([[0, 1, 2], [1, 0, 3], [2, 1, 4], [5, 6, 7]])
    res = mc.split_planes(mc.Mesh(V, F, "t"), [mc.parse_plane("z=0")], cap=False)
    below = next(p for p in res.parts if p.id == 0)
    assert sorted(below.src.tolist()) == [0, 1, 2]


def test_ear_clipping_never_leaves_a_flat_remainder():
    P = np.array([[0, 0], [1, 0], [2, 0], [1, 1]], float)       # die Punkte (0,0),(1,0),(2,0) liegen auf einer Linie
    tris, emergencies = mc.ear_clip(P, [3, 0, 1, 2])             # beginnt an der Spitze: sie wuerde zuerst als Ohr genommen
    assert emergencies == 0 and len(tris) == 2
    for a, b, c in tris:
        assert abs((P[b][0] - P[a][0]) * (P[c][1] - P[a][1]) - (P[b][1] - P[a][1]) * (P[c][0] - P[a][0])) > 1e-9


# ----------------------------------------------------------------------------- Dreiecksreihenfolge (der Mengen-Hash sieht sie nicht)
def test_verify_and_join_notice_swapped_triangle_order(tmp_path):
    mesh = shapes.cube(2.0, sub=2)                               # 48 Dreiecke, abwechselnd auf zwei gleich grosse Teile
    src = write_obj(tmp_path / "w.obj", mesh)
    lab = tmp_path / "l.txt"
    lab.write_text("\n".join(str(i % 2) for i in range(len(mesh.F))))
    out = tmp_path / "o"
    mc.split(src, out, labels=lab)
    assert mc.verify(out, src)["ok"]
    side_path = out / "meshcut.json"
    side = json.loads(side_path.read_text())
    a, b = side["parts"][0], side["parts"][1]
    assert a["faces"] == b["faces"]
    a["src"], b["src"] = b["src"], a["src"]                      # gleiche Dreiecksmenge, andere Reihenfolge -> derselbe Mengen-Hash
    side_path.write_text(json.dumps(side))
    r = mc.verify(out, src)
    assert failed(r) == ["reihenfolge"], r["checks"]
    assert r["source_hash"] == r["joined_hash"]
    cli = subprocess.run([sys.executable, str(ROOT / "kkmeshcut.py"), "join", str(out), str(tmp_path / "j.obj")],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert cli.returncode != 0 and "Reihenfolge" in cli.stdout + cli.stderr


def test_sequence_hash_is_order_dependent_and_triangle_hash_is_not():
    mesh = shapes.cube(1.0)
    perm = np.arange(len(mesh.F))[::-1]
    assert mc.triangle_hash(mesh.V, mesh.F) == mc.triangle_hash(mesh.V, mesh.F[perm])
    assert mc.sequence_hash(mesh.V, mesh.F) != mc.sequence_hash(mesh.V, mesh.F[perm])
