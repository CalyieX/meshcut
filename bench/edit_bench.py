"""Messpruefstand fuer `remove` und `swap` mit synthetischer Wahrheit (Vorbild: pixelcut\\bench\\swap_bench.py).

Prinzip: Ein Netz wird so gebaut, dass die Wahrheit bekannt ist. Auf einer Platte oder Kugel sitzt ein Hoecker (gleiche Randpunkte) oder ein
freies Teil; das Netz OHNE den Hoecker ist die Wahrheit. Danach entfernt/ersetzt meshcut ihn, und es wird gemessen. Jeder Fall laeuft in drei
Massstaeben (0,001 / 1 / 1000) und mit verschobenem Ursprung - feste Einheiten in der Auswertung wuerden dort auffallen.

Die Schwellen unten stehen VOR dem ersten Messlauf fest und werden nicht nachtraeglich an Ergebnisse angepasst.
Aufruf: python bench/edit_bench.py [--json]   (Ausgabe: je Fall Zahlen, am Ende "x von y Faellen in den Schwellen"; Exit 1 bei Verfehlung)
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kkmeshcut as mc          # noqa: E402
import kkmeshcut_edit as ed     # noqa: E402

# ----------------------------------------------------------------------------- eingefrorene Schwellen
TH = {
    "aussen_gleich": 1.0,             # Anteil der unberuehrten Flaechen, die im Ergebnis unveraendert sind (Eckpunkt, UV, Normale, Material)
    "offene_kanten": 0,               # offene Kanten im Ergebnis minus offene Kanten der Wahrheit
    "fuell_abweichung": 0.02,         # Loch: groesster Abstand der Fuellung zur Wahrheit, relativ zur Diagonale des Netzes
    "fuell_flaeche": 0.05,            # Loch: Flaeche der Fuellung gegen die Flaeche der Wahrheit an dieser Stelle (relativer Fehler)
    "normale_grad": 30.0,             # Loch: mittlerer Winkel der Fuellnormalen zur Wahrheit
    "min_guete": 0.05,                # Loch: schlechtestes Dreieck der Fuellung (1 = gleichseitig, 0 = flach)
    "ersatz_abweichung": 0.05,        # swap: Hausdorff-Abstand Ersatz <-> Original, relativ zur Diagonale des Originals
    "doppellauf": True,               # zweimal ausgefuehrt gibt dieselben Bytes
}
SCALES = (1e-3, 1.0, 1e3)
OFFSET = np.array([7.0, -3.0, 5.0])


# ----------------------------------------------------------------------------- Netze mit Wahrheit
def plate(n=12, block=(4, 4, 4), lift=0.25):
    """Platte n x n Zellen; ein Block von Zellen ist ein Hoecker. Rueckgabe: V, F, Objektname je Flaeche, Wahrheit-V (Hoecker flach)."""
    V0, idx = [], {}
    for j in range(n + 1):
        for i in range(n + 1):
            idx[(i, j)] = len(V0)
            V0.append((i / n, j / n, 0.0))
    V0 = np.array(V0)
    a, b, k = block
    V = V0.copy()
    for j in range(b + 1, b + k):
        for i in range(a + 1, a + k):
            V[idx[(i, j)], 2] = lift
    F, obj = [], []
    for j in range(n):
        for i in range(n):
            q = [idx[(i, j)], idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]]
            inside = a <= i < a + k and b <= j < b + k
            for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                F.append(tri)
                obj.append("bump" if inside else "base")
    return V, np.array(F), obj, V0


def sphere(n=24, m=14, block=(5, 4, 4), lift=0.12):
    """UV-Kugel (geschlossen); ein Block von Zellen wird radial angehoben. Wahrheit: Kugel ohne Hoecker."""
    V0, idx = [(0.0, 0.0, 1.0)], {}
    for j in range(1, m):
        th = math.pi * j / m
        for i in range(n):
            ph = 2 * math.pi * i / n
            idx[(i, j)] = len(V0)
            V0.append((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
    south = len(V0)
    V0.append((0.0, 0.0, -1.0))
    V0 = np.array(V0)
    a, b, k = block
    V = V0.copy()
    for j in range(b + 1, b + k):
        for i in range(a + 1, a + k):
            V[idx[(i % n, j)]] *= 1 + lift
    F, obj = [], []

    def add(t, bump):
        F.append(t)
        obj.append("bump" if bump else "base")
    for i in range(n):
        add((0, idx[(i, 1)], idx[((i + 1) % n, 1)]), False)
    for j in range(1, m - 1):
        for i in range(n):
            q = [idx[(i, j)], idx[((i + 1) % n, j)], idx[((i + 1) % n, j + 1)], idx[(i, j + 1)]]
            inside = a <= i < a + k and b <= j < b + k
            add((q[0], q[1], q[2]), inside)
            add((q[0], q[2], q[3]), inside)
    for i in range(n):
        add((south, idx[((i + 1) % n, m - 1)], idx[(i, m - 1)]), False)
    return V, np.array(F), obj, V0


def prim(kind, lo, hi, seg=24):
    """Freies Teil als Dreiecksnetz: Quader, Kugel oder Zylinder (z). Rueckgabe: V, F."""
    tmp = ed.Scene()
    faces = ed.make_primitive(tmp, kind, 2, np.array(lo, float), np.array(hi, float), None, (0, 0, 1, 1), False, False, "", segments=seg)
    F = []
    for f in faces:
        c = [x[0] for x in f.c]
        F += [(c[0], c[k], c[k + 1]) for k in range(1, len(c) - 1)]
    return tmp.V.copy(), np.array(F)


def obj_bytes(V, F, objs, mats, scale):
    """OBJ mit Material, UV und Normalen je Flaeche, verschoben und skaliert."""
    P = (V + OFFSET) * scale
    lines = ["mtllib m.mtl"] + ["v %.12g %.12g %.12g" % tuple(p) for p in P]
    lines += ["vt %.6f %.6f" % (float(V[i][0]) % 1.0, float(V[i][1]) % 1.0) for i in range(len(V))]
    vn = []
    for t in F:
        n = np.cross(V[t[1]] - V[t[0]], V[t[2]] - V[t[0]])
        n = n / (np.linalg.norm(n) or 1.0)
        vn.append(n)
    lines += ["vn %.6f %.6f %.6f" % tuple(n) for n in vn]
    cur = None
    for k, t in enumerate(F):
        key = (objs[k], mats[k])
        if key != cur:
            lines += ["o " + objs[k], "usemtl " + mats[k]]
            cur = key
        lines.append("f " + " ".join("%d/%d/%d" % (a + 1, a + 1, k + 1) for a in t))
    return ("\n".join(lines) + "\n").encode()


# ----------------------------------------------------------------------------- Messung
def tri_normals(P, T):
    n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


def tri_area(P, T):
    return float(np.linalg.norm(np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), axis=1).sum() / 2)


def new_triangles(sc, first):
    T, _ = sc.triangles(range(first, len(sc.faces)))
    return T


def case_hole(kind, scale, work):
    """Hoecker entfernen: das Loch muss wie die Wahrheit aussehen."""
    V, F, objs, V0 = (plate if kind == "plate" else sphere)()
    diag = float(np.linalg.norm(V0.max(axis=0) - V0.min(axis=0))) * scale
    mats = ["red" if o == "bump" else "blue" for o in objs]
    src = work / ("%s_%g.obj" % (kind, scale))
    src.write_bytes(obj_bytes(V, F, objs, mats, scale))
    (work / "m.mtl").write_text("newmtl blue\nKd 0 0 1\nnewmtl red\nKd 1 0 0\n")
    out1, out2 = work / ("o1_%s_%g.obj" % (kind, scale)), work / ("o2_%s_%g.obj" % (kind, scale))
    r = mc.remove(src, name="bump", out=out1)
    mc.remove(src, name="bump", out=out2)
    sc0, sc1 = ed.read_scene(src), ed.read_scene(out1)
    kept = [i for i, o in enumerate(objs) if o != "bump"]
    want = [sc0.signature(sc0.faces[i]) for i in kept]
    got = [sc1.signature(f) for f in sc1.faces[:len(kept)]]
    aussen = sum(1 for a, b in zip(want, got) if a == b) / max(1, len(want))
    pid0, pid1 = ed.position_ids(sc0), ed.position_ids(sc1)
    Vt = (V0 + OFFSET) * scale
    Ft = F
    truth_open = sum(1 for v in ed.edge_counts(_truth_scene(Vt, Ft), ed.position_ids(_truth_scene(Vt, Ft))).values() if v == 1)
    offen = sum(1 for v in ed.edge_counts(sc1, pid1).values() if v == 1) - truth_open
    fill = new_triangles(sc1, len(kept))
    tri_t = Ft[[i for i, o in enumerate(objs) if o == "bump"]]
    samp = np.vstack([sc1.V[fill].reshape(-1, 3), sc1.V[fill].mean(axis=1)])
    dev = float(mc.dist_to_mesh(samp, Vt, Ft).max()) / diag
    area_t, area_f = tri_area(Vt, tri_t), tri_area(sc1.V, fill)
    # Normalen: Fuellung gegen die Wahrheit (naechster Wahrheitsdreieck-Schwerpunkt)
    cen_t = Vt[Ft].mean(axis=1)
    cen_f = sc1.V[fill].mean(axis=1)
    near = np.argmin(((cen_f[:, None, :] - cen_t[None, :, :]) ** 2).sum(axis=2), axis=1)
    cosang = np.einsum("ij,ij->i", tri_normals(sc1.V, fill), tri_normals(Vt, Ft)[near])
    ang = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))).mean())
    guete = min(ed._quality(sc1.V[t]) for t in fill)
    mat_ok = all(f.mat == "blue" for f in sc1.faces[len(kept):])
    return {
        "aussen_gleich": aussen, "offene_kanten": offen, "fuell_abweichung": dev,
        "fuell_flaeche": abs(area_f - area_t) / (area_t or 1.0), "normale_grad": ang, "min_guete": guete,
        "fuell_dreiecke": len(fill), "material_der_umgebung": mat_ok, "doppellauf": out1.read_bytes() == out2.read_bytes(),
    }


def _truth_scene(V, F):
    sc = ed.Scene()
    sc.V = np.asarray(V, float)
    sc.vtext = ["%.17g %.17g %.17g" % tuple(p) for p in sc.V]
    sc.faces = [ed.Face([(int(a), -1, -1) for a in t]) for t in F]
    return sc


def case_swap(kind, scale, work):
    """Freies Teil (Quader/Kugel/Zylinder) auf einer Kugel durch einen Grundkoerper ersetzen: die richtige Form muss gewaehlt werden."""
    Vb, Fb, _, _ = sphere(block=(0, 0, 0))
    lo, hi = (1.4, -0.3, -0.3), (2.0, 0.3, 0.3)
    Vp, Fp = prim(kind, lo, hi)
    V = np.vstack([Vb, Vp])
    F = np.vstack([Fb, Fp + len(Vb)])
    objs = ["base"] * len(Fb) + ["part"] * len(Fp)
    mats = ["blue"] * len(Fb) + ["green"] * len(Fp)
    src = work / ("swap_%s_%g.obj" % (kind, scale))
    src.write_bytes(obj_bytes(V, F, objs, mats, scale))
    (work / "m.mtl").write_text("newmtl blue\nKd 0 0 1\nnewmtl green\nKd 0 1 0\n")
    out1, out2 = work / ("s1_%s_%g.obj" % (kind, scale)), work / ("s2_%s_%g.obj" % (kind, scale))
    r = mc.swap(src, name="part", out=out1)
    mc.swap(src, name="part", out=out2)
    sc0, sc1 = ed.read_scene(src), ed.read_scene(out1)
    kept = list(range(len(Fb)))
    aussen = sum(1 for i in kept if sc0.signature(sc0.faces[i]) == sc1.signature(sc1.faces[i])) / len(kept)
    new = new_triangles(sc1, len(kept))
    Pw = (V + OFFSET) * scale
    orig = Pw[Fp + len(Vb)]
    diag = float(np.linalg.norm(np.array(hi) - np.array(lo))) * scale
    d1 = mc.dist_to_mesh(sc1.V[np.unique(new)], Pw, Fp + len(Vb))
    d2 = mc.dist_to_mesh(Pw[np.unique(Fp + len(Vb))], sc1.V, new)
    haus = max(float(d1.max()), float(d2.max())) / diag
    pid1 = ed.position_ids(sc1)
    offen = sum(1 for v in ed.edge_counts(sc1, pid1).values() if v == 1)
    want = {"box": "Quader", "sphere": "Kugel", "cylinder": "Zylinder"}[kind]
    return {
        "aussen_gleich": aussen, "ersatz_abweichung": haus, "offene_kanten": offen,
        "richtige_form": ("durch " + want) in r["text"], "material_gleich": all(f.mat == "green" for f in sc1.faces[len(kept):]),
        "doppellauf": out1.read_bytes() == out2.read_bytes(),
    }


# ----------------------------------------------------------------------------- Auswertung
def judge(name, m):
    bad = []
    for k, lim in TH.items():
        if k not in m:
            continue
        v = m[k]
        if k in ("aussen_gleich",):
            ok = v >= lim
        elif k == "doppellauf":
            ok = bool(v) is lim
        elif k == "offene_kanten":
            ok = v <= lim
        else:
            ok = v <= lim if k != "min_guete" else v >= lim
        if not ok:
            bad.append("%s=%.4g" % (k, v) if not isinstance(v, bool) else "%s=%s" % (k, v))
    for k in ("material_der_umgebung", "richtige_form", "material_gleich"):
        if k in m and not m[k]:
            bad.append("%s=False" % k)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    cases = [("loch", k) for k in ("plate", "sphere")] + [("swap", k) for k in ("box", "sphere", "cylinder")]
    total, passed, rows = 0, 0, []
    with tempfile.TemporaryDirectory() as td:
        for sc_i, scale in enumerate(SCALES):
            work = Path(td) / ("s%d" % sc_i)
            work.mkdir()
            for typ, kind in cases:
                m = (case_hole if typ == "loch" else case_swap)(kind, scale, work)
                bad = judge("%s_%s" % (typ, kind), m)
                total += 1
                passed += not bad
                rows.append({"fall": "%s %s x%g" % (typ, kind, scale), "ok": not bad, "fehler": bad, "werte": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}})
    for r in rows:
        print("%-5s %-22s %s" % ("OK" if r["ok"] else "FEHL", r["fall"], r["fehler"] if r["fehler"] else json.dumps(r["werte"])))
    print("\n%d von %d Faellen in den Schwellen." % (passed, total))
    if a.json:
        print(json.dumps(rows))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
