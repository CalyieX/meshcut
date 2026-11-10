"""Gezielte Tests gegen Mutanten, die die Pruefrunden uebersehen haben (siehe tests/mutate.py): jede Pruefung muss auch dann
anschlagen, wenn alle anderen sauber sind."""
import base64
import json
import os
import struct
import subprocess
import sys
import tracemalloc
import zipfile
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


# ----------------------------------------------------------------------------- flaechenlose Dreiecke (T-Verbindung): Rundlauf bleibt bitgenau
def _sliver_mesh():
    m = shapes.cube(2.0)
    V, F = m.V.tolist(), m.F.tolist()
    vid = lambda p: next(i for i, v in enumerate(V) if np.allclose(v, p))
    a, c = vid([0, 0, 0]), vid([2, 0, 0])
    V.append([1, 0, 0])
    mi = len(V) - 1
    new = []
    for f in F:
        cyc = [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
        if (a, c) in cyc:
            k = cyc.index((a, c))
            r = f[k:] + f[:k]
            new += [[a, mi, r[2]], [mi, c, r[2]]]
        else:
            new.append(f)
    new.append([a, c, mi])                                       # flaechenloses Fuellstueck auf der Kante a-c
    return mc.Mesh(np.array(V), np.array(new))


@pytest.mark.parametrize("plane", ["x=1", "x=0.5", "x=1.5", "1,1,0,1", "y=1", "z=0.7"])
def test_flat_triangles_round_trip_exactly(tmp_path, plane):
    src = write_obj(tmp_path / "s.obj", _sliver_mesh())
    mc.split(src, tmp_path / "o", plane=[plane], verify=True)
    r = mc.verify(tmp_path / "o", src)
    assert r["ok"], r["checks"]
    assert r["joined_hash"] == r["source_hash"]
    mc.join(tmp_path / "o", tmp_path / "back.obj")                # auch der Weg ueber die Kommandozeilen-Funktion


# ----------------------------------------------------------------------------- lesende Befehle stoeren keinen laufenden Schreiber
def test_reading_commands_leave_young_staging_folders_alone(tmp_path):
    src, out = split_to(tmp_path)
    young = out / ".meshcut_laeuft"
    young.mkdir()
    (young / mc.MARKER).write_text("meshcut")
    (young / "x").write_text("gehoert einem laufenden split")
    assert mc.verify(out, src)["ok"]
    mc.join(out, tmp_path / "back.obj")
    mc.info(out / "part_0.obj") if (out / "part_0.obj").exists() else None
    assert (young / "x").exists()                                # frisch: unangetastet
    os.utime(young, (1, 1))
    assert mc.verify(out, src)["ok"]
    assert not young.exists()                                    # alt: aufgeraeumt


# ----------------------------------------------------------------------------- Dateien, die sich beim Aufloesen vervielfachen
def _glb(js, binary=b""):
    jb = json.dumps(js).encode()
    jb += b" " * ((-len(jb)) % 4)
    return (b"glTF" + struct.pack("<II", 2, 12 + 8 + len(jb) + 8 + len(binary)) + struct.pack("<II", len(jb), 0x4E4F534A) + jb
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


def _fan_glb(depth):
    nodes = [{"mesh": 0, "children": [i + 1, i + 1]} for i in range(depth)] + [{"mesh": 0}]      # jede Ebene verdoppelt die Besuche
    js = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes,
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 4}]}],
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
          "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}], "buffers": [{"byteLength": 36}]}
    return _glb(js, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "<f4").tobytes())


def test_glb_instance_explosion_is_stopped():
    assert len(mc.read_glb(_fan_glb(3))) == 15                   # kleine Faecher bleiben erlaubt
    with pytest.raises(mc.MeshcutError, match="vervielfacht"):
        mc.read_glb(_fan_glb(20))


def test_glb_triangle_budget_is_enforced(monkeypatch):
    monkeypatch.setattr(mc, "MAX_READ_TRIANGLES", 10)
    with pytest.raises(mc.MeshcutError, match="Dreiecke nach dem Aufloesen"):
        mc.read_glb(_fan_glb(4))


def _3mf(xml):
    import io
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("3D/3dmodel.model", xml)
    return b.getvalue()


def _component_tower(n, levels):
    objs = ['<object id="1"><mesh><vertices><vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/></vertices>'
            '<triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh></object>']
    for k in range(2, levels + 2):
        objs.append('<object id="%d"><components>%s</components></object>' % (k, "".join('<component objectid="%d"/>' % (k - 1) for _ in range(n))))
    return ('<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>%s</resources>'
            '<build><item objectid="%d"/></build></model>' % ("".join(objs), levels + 1))


def test_3mf_component_explosion_is_stopped():
    assert len(mc.read_3mf(_3mf(_component_tower(2, 3)))[0][1]) == 8
    with pytest.raises(mc.MeshcutError, match="vervielfacht"):
        mc.read_3mf(_3mf(_component_tower(100, 7)))


def test_3mf_triangle_budget_is_enforced(monkeypatch):
    monkeypatch.setattr(mc, "MAX_READ_TRIANGLES", 5)
    with pytest.raises(mc.MeshcutError, match="Dreiecke nach dem Aufloesen"):
        mc.read_3mf(_3mf(_component_tower(2, 3)))


def test_3mf_zip_bomb_is_not_unpacked(monkeypatch):
    monkeypatch.setattr(mc, "MAX_XML_BYTES", 2000)
    with pytest.raises(mc.MeshcutError, match="entpackt"):
        mc.read_3mf(_3mf("<model>" + " " * 100000 + "</model>"))


def test_huge_polygons_are_refused(monkeypatch):
    monkeypatch.setattr(mc, "MAX_POLYGON", 10)
    ring = "\n".join("v %f %f 0" % (np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False))
    with pytest.raises(mc.MeshcutError, match="Ecken"):
        mc.read_meshes_bytes("obj", (ring + "\nf " + " ".join(str(i + 1) for i in range(12)) + "\n").encode(), "x")
    ok = (ring + "\nf " + " ".join(str(i + 1) for i in range(10)) + "\n").encode()
    assert len(mc.read_meshes_bytes("obj", ok, "x")[0].F) == 8


# ----------------------------------------------------------------------------- Dauerhaftigkeit: erst fsync, dann umbenennen
def test_files_are_synced_before_they_are_renamed(tmp_path, monkeypatch):
    order = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (order.append("replace"), real_replace(a, b))[1])
    src, out = split_to(tmp_path)
    assert order.count("fsync") >= 7 and "replace" in order        # vier Teile + Begleitdatei + zwei Journal-Staende
    assert order.index("fsync") < order.index("replace")          # Teile und Begleitdatei sind auf der Platte, bevor sie ihren Namen bekommen
    order.clear()
    mc.join(out, tmp_path / "b.obj")
    assert order[:2] == ["fsync", "replace"]                       # auch die zusammengesetzte Datei


# ----------------------------------------------------------------------------- beschaedigte Begleitdatei: immer eine saubere Meldung, nie ein roher Fehler
def _paths(o, pre=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield pre + (k,)
            yield from _paths(v, pre + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o[:2]):
            yield pre + (i,)
            yield from _paths(v, pre + (i,))


@pytest.mark.parametrize("fmt", ["obj", "3mf"])
def test_damaged_sidecar_gives_clean_errors_only(tmp_path, fmt):
    import copy
    src = write_obj(tmp_path / "s.obj", shapes.cube(2.0, sub=2))
    mc.split(src, tmp_path / "o", plane=["x=1"], format=fmt)
    side_path = tmp_path / "o" / "meshcut.json"
    base = json.loads(side_path.read_text())
    raw = []
    for p in _paths(base):
        for mode in ("del", "none", "str", "neg"):
            s = copy.deepcopy(base)
            par = s
            for k in p[:-1]:
                par = par[k]
            if mode == "del" and isinstance(par, dict):
                del par[p[-1]]
            else:
                par[p[-1]] = {"del": None, "none": None, "str": "x", "neg": -1}[mode]
            side_path.write_text(json.dumps(s))
            for call in (lambda: mc.verify(tmp_path / "o"), lambda: mc.join(tmp_path / "o", tmp_path / "j.obj")):
                try:
                    call()
                except mc.MeshcutError:
                    pass
                except Exception as e:                           # roher Fehler = Luecke
                    raw.append((p, mode, type(e).__name__))
    assert not raw, raw[:5]


# ----------------------------------------------------------------------------- Runde 4
def test_reading_commands_only_touch_their_own_staging_folders(tmp_path):
    src, out = split_to(tmp_path)
    foreign = out / ".meshcut_notes"                             # gleicher Namensanfang, aber nicht von meshcut
    foreign.mkdir()
    (foreign / "wichtig.txt").write_text("bleibt")
    os.utime(foreign, (1, 1))
    mine = out / ".meshcut_abgebrochen"
    mine.mkdir()
    (mine / mc.MARKER).write_text("meshcut")
    os.utime(mine, (1, 1))
    assert mc.verify(out, src)["ok"]
    assert (foreign / "wichtig.txt").read_text() == "bleibt" and not mine.exists()


def test_recovery_never_follows_links_out_of_the_folder(tmp_path, capsys):
    src, out = split_to(tmp_path)
    outside = tmp_path / "aussen"
    outside.mkdir()
    (outside / "datei.txt").write_text("bleibt")
    link = out / ".meshcut_evil"
    if os.name == "nt":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True, stdin=subprocess.DEVNULL, creationflags=0x08000000)
        if r.returncode:
            pytest.skip("Junction nicht moeglich")
    else:
        try:
            os.symlink(outside, link, target_is_directory=True)
        except OSError:
            pytest.skip("keine Symlinks erlaubt")
    (outside / mc.MARKER).write_text("meshcut")                    # selbst mit Marker: ein Link wird nie angefasst
    capsys.readouterr()
    mc.recover_outdir(out)
    assert capsys.readouterr().err == ""                         # nicht einmal ein Versuch, den Link zu raeumen
    assert (outside / "datei.txt").read_text() == "bleibt" and (outside / mc.MARKER).exists()
    mc.split(src, out, plane=["x=0.3"])                          # und split laeuft weiter


def test_flat_triangle_off_the_surface_does_not_fail_hausdorff(tmp_path):
    T = np.array([[[2.0, 0, 0], [2, 0, 0], [0, 3, 0]], [[1.0, 1, 0], [1, 3, 0], [2, 0, 0]]])
    V, F = mc.weld_triangles(T)
    src = tmp_path / "a.stl"
    src.write_bytes(mc.mesh_bytes("stl", V, F, "x"))
    mc.split(src, tmp_path / "o", plane=["1,1,0,2", "1,-1,0,0"], no_cap=True)
    r = mc.verify(tmp_path / "o", source=src)
    assert r["ok"], r["checks"]


@pytest.mark.parametrize("cmd", ["info", "split", "grid", "labels", "join"])
def test_empty_meshes_get_one_clean_message(tmp_path, cmd):
    e = tmp_path / "e.ply"
    e.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nelement face 0\n"
                  b"property list uchar int vertex_indices\nend_header\n")
    args = {"info": ["info", str(e)], "split": ["split", str(e), str(tmp_path / "o"), "--plane", "x=0"],
            "grid": ["split", str(e), str(tmp_path / "o"), "--grid", "2"], "labels": ["split", str(e), str(tmp_path / "o"), "--labels", str(e)],
            "join": ["join", str(e), str(tmp_path / "j.stl")]}[cmd]
    r = subprocess.run([sys.executable, str(ROOT / "kkmeshcut.py")] + args, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert r.returncode != 0 and "keine Dreiecke" in r.stdout + r.stderr and "Traceback" not in r.stderr


def test_many_polygons_share_one_work_budget(monkeypatch):
    monkeypatch.setattr(mc, "MAX_POLYGON_WORK", 500)
    ring = "\n".join("v %f %f 0" % (np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False))
    faces = "\n".join("f 1 2 3 4 5 6 7 8" for _ in range(20))     # jede Flaeche 64 Einheiten: 8 passen, 20 nicht
    with pytest.raises(mc.MeshcutError, match="zu aufwendig"):
        mc.read_meshes_bytes("obj", (ring + "\n" + faces + "\n").encode(), "x")
    assert len(mc.read_meshes_bytes("obj", (ring + "\n" + "\n".join(faces.split("\n")[:7]) + "\n").encode(), "x")[0].F) == 7 * 6


def test_instances_of_a_big_mesh_do_not_reparse_it(tmp_path):
    import time
    nv, nc = 100_000, 400
    vs = "".join('<vertex x="%d" y="%d" z="%d"/>' % (i, i % 7, i % 5) for i in range(nv))
    xml = ('<model unit="millimeter"><resources><object id="1"><mesh><vertices>%s</vertices><triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh></object>'
           '<object id="2"><components>%s</components></object></resources><build><item objectid="2"/></build></model>') % (vs, "".join('<component objectid="1"/>' for _ in range(nc)))
    t0 = time.time()
    got = mc.read_3mf(_3mf(xml))
    assert len(got[0][1]) == nc and time.time() - t0 < 20            # frueher ~105 ms je Instanz (40 s)


def test_production_limits_hold_without_monkeypatching():
    one = ("<object id=\"1\"><mesh><vertices>" + "".join('<vertex x="%d" y="0" z="0"/>' % i for i in range(1002)) + "</vertices><triangles>"
           + "".join('<triangle v1="%d" v2="%d" v3="%d"/>' % (i, i + 1, i + 2) for i in range(1000)) + "</triangles></mesh></object>")
    xml = ('<model unit="millimeter"><resources>%s<object id="2"><components>%s</components></object></resources><build><item objectid="2"/></build></model>'
           % (one, "".join('<component objectid="1"/>' for _ in range(mc.MAX_READ_TRIANGLES // 1000 + 5))))
    with pytest.raises(mc.MeshcutError, match="Dreiecke nach dem Aufloesen"):
        mc.read_3mf(_3mf(xml))
    with pytest.raises(mc.MeshcutError, match="entpackt"):
        mc.read_3mf(_3mf("<model>" + " " * (mc.MAX_XML_BYTES + 10) + "</model>"))


def test_glb_instances_share_one_parse_of_the_mesh(monkeypatch):
    calls = []
    real = np.frombuffer
    monkeypatch.setattr(np, "frombuffer", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    nodes = [{"mesh": 0} for _ in range(200)]
    js = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(200))}], "nodes": nodes,
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 4}]}],
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
          "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}], "buffers": [{"byteLength": 36}]}
    got = mc.read_glb(_glb(js, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "<f4").tobytes()))
    assert sum(len(t) for _, t in got) == 200 and len(calls) <= 3        # einmal gelesen, nicht 200 mal
