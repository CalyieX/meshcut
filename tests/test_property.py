"""Eigenschaftstests (hypothesis): zufaellige Ebenen, Ebenen durch Eckpunkte, Gitter, Zuordnungskarten."""
import math

import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

import kkmeshcut as mc
import shapes

import os
SET = settings(max_examples=int(os.environ.get("MESHCUT_EXAMPLES", "25")), deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])

MESHES = [
    shapes.cube(2.0, lo=(-1, -1, -1)),
    shapes.cube(3.0, sub=3),
    shapes.sphere(10, 6),
    shapes.torus(n=14, m=8),
    shapes.prism(shapes.L_SHAPE),
    shapes.prism(shapes.U_SHAPE),
]


def roundtrip(mesh, res, prec="float64"):
    stats = mc.mesh_stats(mesh.V, mesh.F)
    files = [("part_%d.obj" % p.id, None) for p in res.parts]
    side = mc.make_sidecar(res, "obj", stats, files)
    joined = mc.join_with_sidecar([(p.id, p.V, p.F) for p in res.parts], side)
    return side, joined


def check(mesh, res):
    st0 = mc.mesh_stats(mesh.V, mesh.F)
    assert abs(sum(mc.volume_area(p.V, p.F)[0] for p in res.parts) - st0["volume"]) <= 1e-9 * max(1.0, abs(st0["volume"]))
    for p in res.parts:
        ps = mc.mesh_stats(p.V, p.F)
        if ps["open_edges"] == 0 and ps["bad_edges"] > 0:
            continue                                            # bekannte, im README genannte Grenze: eine Ebene durch eine konkave Kante laesst zwei Huellen an einer Kante beruehren (geschlossen, aber nichtmanifold)
        assert ps["watertight"] and ps["euler"] % 2 == 0
    side, joined = roundtrip(mesh, res)
    assert mc.triangle_hash(joined.V, joined.F) == mc.triangle_hash(mesh.V, mesh.F) == side["source"]["hash"]


def try_split(mesh, planes, **kw):
    try:
        return mc.split_planes(mesh, planes, **kw)
    except mc.MeshcutError as e:
        assert "schneiden" in str(e)                            # erlaubt ist nur: Ebene trifft das Netz nicht
        return None


@SET
@given(st.integers(0, len(MESHES) - 1), st.lists(st.floats(-1, 1), min_size=3, max_size=3).filter(lambda v: sum(x * x for x in v) > 1e-3),
       st.floats(0.05, 0.95))
def test_random_plane(mi, n, t):
    mesh = MESHES[mi]
    n = np.array(n)
    pr = mesh.V @ (n / np.linalg.norm(n))
    d = float(pr.min() + t * (pr.max() - pr.min())) * np.linalg.norm(n)
    res = try_split(mesh, [mc.Plane(n, d)])
    if res:
        check(mesh, res)


@SET
@given(st.integers(0, len(MESHES) - 1), st.integers(0, 2), st.integers(0, 10 ** 6))
def test_axis_plane_on_a_vertex_coordinate(mi, axis, k):
    mesh = MESHES[mi]
    v = mesh.V[k % len(mesh.V)]
    n = np.zeros(3)
    n[axis] = 1.0
    res = try_split(mesh, [mc.Plane(n, float(v[axis]))])
    if res:
        check(mesh, res)


@SET
@given(st.integers(0, len(MESHES) - 1), st.integers(0, 10 ** 6), st.integers(0, 10 ** 6), st.integers(0, 10 ** 6))
def test_plane_through_three_vertices(mi, a, b, c):
    mesh = MESHES[mi]
    P = [mesh.V[i % len(mesh.V)] for i in (a, b, c)]
    n = np.cross(P[1] - P[0], P[2] - P[0])
    if np.linalg.norm(n) < 1e-9:
        return
    res = try_split(mesh, [mc.Plane(n, float(n @ P[0]))])
    if res:
        check(mesh, res)


@SET
@given(st.integers(0, len(MESHES) - 1), st.integers(1, 3), st.integers(1, 3), st.integers(1, 3))
def test_grid(mi, nx, ny, nz):
    mesh = MESHES[mi]
    if nx * ny * nz == 1:
        return
    planes = mc.grid_planes(mesh.V.min(0), mesh.V.max(0), [nx, ny, nz])
    res = try_split(mesh, planes, mode="grid")
    if res:
        check(mesh, res)
        ids = {p.id for p in res.parts}
        assert max(ids) < nx * ny * nz


@SET
@given(st.integers(0, len(MESHES) - 1), st.integers(1, 3), st.integers(0, 2 ** 31))
def test_wedges(mi, half, seed):
    mesh = MESHES[mi]
    ax = seed % 3
    c = (mesh.V.min(0) + mesh.V.max(0)) / 2
    planes, table = mc.wedge_planes(c, ax, 2 * half)
    res = try_split(mesh, planes, table=table, mode="wedges")
    if res:
        check(mesh, res)


@SET
@given(st.integers(0, len(MESHES) - 1), st.integers(2, 6), st.integers(0, 2 ** 31))
def test_random_labels_roundtrip_bit_exact(mi, k, seed):
    mesh = MESHES[mi]
    lab = np.random.default_rng(seed).integers(0, k, len(mesh.F))
    if len(np.unique(lab)) < 2:
        return
    res = mc.split_labels(mesh, lab)
    side, joined = roundtrip(mesh, res)
    assert mc.triangle_hash(joined.V, joined.F) == mc.triangle_hash(mesh.V, mesh.F)
    # Geometrie unveraendert: dieselben Koordinaten, nichts Neues
    assert sum(len(p.F) for p in res.parts) == len(mesh.F)
    assert all(np.isin(p.V, mesh.V).all() for p in res.parts)


@SET
@given(st.integers(0, len(MESHES) - 1), st.integers(0, 2 ** 31))
def test_hash_ignores_order_and_start_corner(mi, seed):
    mesh = MESHES[mi]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(mesh.F))
    rot = rng.integers(0, 3, len(mesh.F))
    F = mesh.F[perm]
    F = np.take_along_axis(F, (rot[:, None] + np.arange(3)) % 3, 1)
    vp = rng.permutation(len(mesh.V))
    inv = np.empty_like(vp)
    inv[vp] = np.arange(len(vp))
    assert mc.triangle_hash(mesh.V[vp], inv[F]) == mc.triangle_hash(mesh.V, mesh.F)
    # Umlaufsinn ist Teil des Hashs
    assert mc.triangle_hash(mesh.V, mesh.F[:, [0, 2, 1]]) != mc.triangle_hash(mesh.V, mesh.F)


@SET
@given(st.integers(0, len(MESHES) - 1), st.floats(0.1, 0.9), st.floats(0.1, 0.9))
def test_two_planes_sequence_equals_both_orders(mi, t1, t2):
    mesh = MESHES[mi]
    lo, hi = mesh.V.min(0), mesh.V.max(0)
    p1 = mc.Plane([1, 0, 0], float(lo[0] + t1 * (hi[0] - lo[0])), weight=1)
    p2 = mc.Plane([0, 1, 0], float(lo[1] + t2 * (hi[1] - lo[1])), weight=2)
    a = try_split(mesh, [p1, p2])
    b = try_split(mesh, [mc.Plane([0, 1, 0], p2.d, weight=1), mc.Plane([1, 0, 0], p1.d, weight=2)])
    if a and b:
        check(mesh, a)
        va = sorted(round(mc.volume_area(p.V, p.F)[0], 9) for p in a.parts)
        vb = sorted(round(mc.volume_area(p.V, p.F)[0], 9) for p in b.parts)
        assert va == vb
