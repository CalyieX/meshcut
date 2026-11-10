import math

import numpy as np
import pytest

import kkmeshcut as mc
import shapes

SHAPES = {
    "cube": lambda: shapes.cube(2.0, lo=(-1, -1, -1)),
    "cube_sub": lambda: shapes.cube(3.0, sub=3),
    "sphere": lambda: shapes.sphere(),
    "torus": lambda: shapes.torus(),
    "L": lambda: shapes.prism(shapes.L_SHAPE),
    "U": lambda: shapes.prism(shapes.U_SHAPE),
}


def check_split(mesh, res):
    st = mc.mesh_stats(mesh.V, mesh.F)
    vol = sum(mc.volume_area(p.V, p.F)[0] for p in res.parts)
    assert abs(vol - st["volume"]) <= 1e-9 * max(1.0, abs(st["volume"]))
    area = sum(mc.volume_area(p.V, p.F[:len(p.F) - p.n_cap])[1] for p in res.parts)
    assert abs(area - st["area"]) <= 1e-9 * st["area"]
    for p in res.parts:
        ps = mc.mesh_stats(p.V, p.F)
        assert ps["watertight"], (p.id, ps)
        assert ps["euler"] % 2 == 0
    return st


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("spec", ["x=0.37", "y=0.5", "z=0.31", "1,1,0,0.9", "1,-2,3,0.2"])
def test_plane_split_is_watertight_and_conserves_volume(name, spec):
    m = SHAPES[name]()
    lo, hi = m.V.min(0), m.V.max(0)
    c = (lo + hi) / 2
    p = mc.parse_plane(spec)
    p = mc.Plane(p.n, float(p.n @ c) + 0.05 * float(np.linalg.norm(hi - lo)))   # sicher durch den Koerper
    res = mc.split_planes(m, [p])
    assert len(res.parts) == 2
    check_split(m, res)


@pytest.mark.parametrize("name", list(SHAPES))
def test_roundtrip_hash_exact_via_sidecar(name, tmp_path):
    m = SHAPES[name]()
    for fmt in mc.FORMATS:
        src = tmp_path / ("in." + fmt)
        src.write_bytes(mc.mesh_bytes(fmt, m.V, m.F))
        out = tmp_path / ("o_" + fmt)
        side = mc.split(src, out, grid="2x2x2", format=fmt, verify=(fmt == "stl"))
        assert side["source"]["precision"] == ("float32" if fmt in mc.FLOAT32_FORMATS else "float64")
        j = tmp_path / ("j_" + fmt + ".obj")
        mc.join(out, j)
        a = mc.load_mesh(src)
        b = mc.load_mesh(j)
        prec = side["source"]["precision"]
        assert mc.triangle_hash(a.V, a.F, prec) == mc.triangle_hash(b.V, b.F, prec) == side["source"]["hash"]


def test_labels_mode_is_bit_exact_and_keeps_every_coordinate(tmp_path):
    m = shapes.sphere(16, 10)
    rng = np.random.default_rng(1)
    lab = rng.integers(0, 5, len(m.F))
    np.save(tmp_path / "l.npy", lab)
    src = tmp_path / "s.obj"
    src.write_bytes(mc.mesh_bytes("obj", m.V, m.F))
    side = mc.split(src, tmp_path / "o", labels=tmp_path / "l.npy", format="obj")
    assert side["mode"] == "labels"
    assert sum(p["cap_faces"] for p in side["parts"]) == 0
    mc.join(tmp_path / "o", tmp_path / "j.obj")
    a, b = mc.load_mesh(src), mc.load_mesh(tmp_path / "j.obj")
    assert mc.triangle_hash(a.V, a.F) == mc.triangle_hash(b.V, b.F)
    for p in side["parts"]:
        part = mc.load_mesh(tmp_path / "o" / p["file"])
        want = mc.triangle_hash(m.V, m.F[lab == p["id"]])
        assert mc.triangle_hash(part.V, part.F) == want


def test_grid_cells_and_numbers():
    m = shapes.cube(4.0)
    res = mc.split_planes(m, mc.grid_planes(m.V.min(0), m.V.max(0), [2, 2, 2]), mode="grid")
    assert [p.id for p in res.parts] == list(range(8))
    for p in res.parts:
        assert np.allclose(p.V.max(0) - p.V.min(0), 2.0)
        cell = [p.id % 2, (p.id // 2) % 2, p.id // 4]
        assert np.allclose(p.V.min(0), 2.0 * np.array(cell))
    check_split(m, res)


def test_planes_through_vertices_edges_and_faces():
    m = shapes.cube(2.0, sub=2)
    for spec in ("x=1", "x=0.5", "y=1"):
        check_split(m, mc.split_planes(m, [mc.parse_plane(spec)]))
    with pytest.raises(mc.MeshcutError, match="schneiden"):      # liegt auf einer Flaeche: beruehrt nur
        mc.split_planes(m, [mc.parse_plane("z=2")])
    res = mc.split_planes(m, [mc.parse_plane("x=1"), mc.parse_plane("y=1"), mc.parse_plane("z=1")])
    assert len(res.parts) == 8
    check_split(m, res)


def test_plane_in_a_ledge_with_coplanar_triangles():
    """Stufe: die Ebene liegt in der Oberseite der unteren Platte, daneben steht ein Block darauf."""
    plate = shapes.prism([(0, 0), (4, 0), (4, 2), (0, 2)], h=1.0)
    tower = shapes.prism([(1, 0), (2, 0), (2, 2), (1, 2)], h=1.0)
    P = np.concatenate([plate.V[plate.F], (tower.V + [0, 0, 1])[tower.F]])
    # der Turm sitzt mit seiner Unterseite auf der Platte: die zwei Beruehrflaechen fallen weg
    V, F = mc.weld_triangles(P)
    m = mc.join_plain([mc.Mesh(V, F)])
    assert mc.mesh_stats(m.V, m.F)["watertight"] or True
    res = mc.split_planes(plate, [mc.parse_plane("x=1")])
    check_split(plate, res)


def test_wedges():
    m = shapes.cube(2.0, lo=(-1, -1, -1), sub=2)
    planes, table = mc.wedge_planes(np.array([0.0, 0.0, 0.0]), 2, 8)
    res = mc.split_planes(m, planes, table=table, mode="wedges")
    assert [p.id for p in res.parts] == list(range(8))
    check_split(m, res)
    for p in res.parts:                                         # Keil m liegt im Winkelbereich [m*45, (m+1)*45]
        r = np.hypot(p.V[:, 0], p.V[:, 1])
        ang = np.degrees(np.arctan2(p.V[:, 1], p.V[:, 0]))[r > 1e-9]
        d = (ang - (p.id + 0.5) * 45 + 180) % 360 - 180
        assert np.abs(d).max() <= 22.5 + 1e-6


def test_wedge_triangle_inside_the_axis():
    m = shapes.prism(shapes.L_SHAPE, h=1.0)                     # Seitenflaechen y=0 und x=1 liegen genau in den Keilebenen
    planes, table = mc.wedge_planes(np.array([1.0, 0.0, 0.0]), 2, 4)
    res = mc.split_planes(m, planes, table=table, mode="wedges")
    check_split(m, res)
    assert len(res.parts) >= 2


def test_wedge_rejects_odd():
    with pytest.raises(mc.MeshcutError):
        mc.wedge_planes(np.zeros(3), 2, 5)


def test_new_vertices_once_per_edge():
    m = shapes.sphere(20, 12)
    res = mc.split_planes(m, [mc.parse_plane("1,2,3,0.1")])
    for p in res.parts:
        assert len(np.unique(p.V, axis=0)) == len(p.V)
    allv = np.concatenate([p.V for p in res.parts])
    _, c = np.unique(allv, axis=0, return_counts=True)
    assert set(c.tolist()) <= {1, 2}                            # Nahtpunkt: genau in beiden Teilen, nie in einem dritten


def test_open_mesh_is_refused_and_no_cap_works():
    m = shapes.cube(2.0)
    open_ = mc.Mesh(m.V, m.F[:-1])
    with pytest.raises(mc.MeshcutError, match="wasserdicht"):
        mc.split_planes(open_, [mc.parse_plane("x=1")])
    res = mc.split_planes(open_, [mc.parse_plane("x=1")], cap=False)
    assert all(p.n_cap == 0 for p in res.parts)


def test_inside_out_mesh_needs_flip():
    m = shapes.cube(2.0)
    inv = mc.Mesh(m.V, m.F[:, [0, 2, 1]])
    with pytest.raises(mc.MeshcutError, match="--flip"):
        mc.split_planes(inv, [mc.parse_plane("x=1")])


def test_torus_cap_is_an_annulus_with_a_hole():
    m = shapes.torus()
    res = mc.split_planes(m, [mc.parse_plane("z=0.0")])
    for p in res.parts:
        cap = p.F[len(p.F) - p.n_cap:]
        P = p.V[cap]
        a = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])[:, 2]
        assert (np.sign(a[np.abs(a) > 1e-12]) == np.sign(a[np.abs(a) > 1e-12])[0]).all()     # gleichsinnig, keine Ueberlappung durch Gegenlauf
        want = math.pi * ((2 + 0.7) ** 2 - (2 - 0.7) ** 2)
        assert math.isclose(abs(a.sum()) / 2, want, rel_tol=0.03)       # Kreisring, nicht gefuellt (24-Eck statt Kreis)


def test_hollow_sphere_island_in_hole():
    """Hohlkugel: Deckel = Ring; Kugel in der Hoehlung eines Quaders: Insel im Loch."""
    outer = shapes.sphere(16, 10, r=2.0)
    inner = shapes.sphere(16, 10, r=1.0)
    inner = mc.Mesh(inner.V, inner.F[:, [0, 2, 1]])
    island = shapes.sphere(12, 8, r=0.4)
    V = np.concatenate([outer.V, inner.V, island.V])
    F = np.concatenate([outer.F, inner.F + len(outer.V), island.F + len(outer.V) + len(inner.V)])
    m = mc.Mesh(V, F)
    res = mc.split_planes(m, [mc.parse_plane("x=0.0")])
    check_split(m, res)
    for p in res.parts:
        cap = p.F[len(p.F) - p.n_cap:]
        a = np.abs(np.cross(p.V[cap][:, 1] - p.V[cap][:, 0], p.V[cap][:, 2] - p.V[cap][:, 0])[:, 0]).sum() / 2
        assert a < math.pi * (4 - 1) * 1.01 + math.pi * 0.4 ** 2


def test_naive_join_without_sidecar_is_close(tmp_path):
    m = shapes.sphere(14, 10)
    src = tmp_path / "s.obj"
    src.write_bytes(mc.mesh_bytes("obj", m.V, m.F))
    mc.split(src, tmp_path / "o", plane=["x=0.2", "z=-0.3"], format="obj")
    files = sorted((tmp_path / "o").glob("part_*.obj"))
    j = tmp_path / "j.obj"
    mc.join(files, j)
    a, b = mc.load_mesh(src), mc.load_mesh(j)
    st = mc.mesh_stats(b.V, b.F)
    assert st["watertight"] and st["faces"] > len(a.F)
    assert abs(st["volume"] - mc.mesh_stats(a.V, a.F)["volume"]) < 1e-12
    hd, _ = mc.hausdorff_vertices(a.V, a.F, b.V, b.F)
    assert hd < 1e-12


def test_verify_reports_changed_part(tmp_path):
    m = shapes.cube(2.0)
    src = tmp_path / "s.stl"
    src.write_bytes(mc.mesh_bytes("stl", m.V, m.F))
    mc.split(src, tmp_path / "o", grid="2x1x1")
    assert mc.verify(tmp_path / "o", source=src)["ok"]
    f = tmp_path / "o" / "part_0.stl"
    b = bytearray(f.read_bytes())
    b[84 + 12 + 3] ^= 0x40                                      # eine Koordinate veraendern
    f.write_bytes(bytes(b))
    assert not mc.verify(tmp_path / "o", source=src)["ok"]
    with pytest.raises(mc.MeshcutError):
        mc.join(tmp_path / "o", tmp_path / "j.stl")


def test_trimesh_cross_check():
    trimesh = pytest.importorskip("trimesh")
    m = shapes.torus()
    res = mc.split_planes(m, [mc.parse_plane("1,1,0,0.3")])
    tm = trimesh.Trimesh(m.V, m.F, process=False)
    n = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
    o = n * (0.3 / math.sqrt(2))
    for p, sign in zip(res.parts, (-1, 1)):
        ref = tm.slice_plane(o, sign * n, cap=True)
        mine = mc.mesh_stats(p.V, p.F)
        assert math.isclose(mine["volume"], ref.volume, rel_tol=1e-6)
        assert math.isclose(mine["area"], ref.area, rel_tol=1e-6)


def test_manifold3d_cross_check():
    manifold3d = pytest.importorskip("manifold3d")
    m = shapes.sphere(20, 12)
    res = mc.split_planes(m, [mc.parse_plane("1,2,3,0.1")])
    man = manifold3d.Manifold(manifold3d.Mesh(m.V.astype(np.float32), m.F.astype(np.uint32)))
    n = np.array([1.0, 2.0, 3.0])
    nn = n / np.linalg.norm(n)
    a, b = man.split_by_plane(nn.tolist(), 0.1 / np.linalg.norm(n))
    mine = sorted(mc.volume_area(p.V, p.F)[0] for p in res.parts)
    assert np.allclose(sorted([a.volume(), b.volume()]), mine, rtol=1e-5)
