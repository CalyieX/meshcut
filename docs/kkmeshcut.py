#!/usr/bin/env python3
"""K&K meshcut - zerlegt 3D-Netze exakt und setzt sie wieder zusammen.

Die meisten Schneidewerkzeuge fuer 3D-Netze sind Handarbeit in einer Oberflaeche. meshcut ist ein Werkzeug fuer die
Kommandozeile und fuer Python, das das Zerlegen mit einem Beweis fuer den Rundlauf verbindet:

  Zuordnungsmodus (--labels)   Die Geometrie bleibt unveraendert, nur die Dreiecke werden auf Teile verteilt. Die Teile
                               ergeben beim Zusammensetzen bitgenau das Original (SHA-256 ueber die Dreiecke).
  Schnittmodus (--plane, --grid, --max, --wedges)
                               Dreiecke werden an Ebenen zerlegt. Neue Eckpunkte entstehen genau einmal pro Kante, beide
                               Seiten teilen sie, an der Schnittflaeche entsteht je Teil ein Deckel. Mit der Begleitdatei
                               meshcut.json setzt `join` die urspruenglichen Dreiecke wieder her (auch hier bitgenau).

Befehle
  split   Netz zerlegen (Ebenen, Gitter, Druckbettmass, Keile, Zuordnungskarte); schreibt Teile + meshcut.json
  join    Teile wieder zusammensetzen (mit meshcut.json bitgenau, ohne naeherungsweise)
  verify  Rundlauf pruefen: Volumen, Flaeche, wasserdicht, Euler, Hash, Hausdorff
  info    Kennzahlen eines Netzes
  labels  Zuordnungskarte (ein Wert pro Dreieck) fuer ein Gitter schreiben

`python kkmeshcut.py BEFEHL -h` zeigt die Optionen.
"""
import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

import numpy as np

__version__ = "0.1.0"

MAX_PARTS = 1024
MAX_PLANES = 60
PART_NAME = re.compile(r"part_(0|[1-9]\d*)\.(stl|obj)")
MULTI_NAME = {"3mf": "parts.3mf", "glb": "parts.glb"}
SIDECAR = "meshcut.json"
FORMATS = ("stl", "obj", "3mf", "glb")
FLOAT32_FORMATS = ("stl", "glb")


class MeshcutError(Exception):
    """Wird von der Python-Schnittstelle geworfen (die Kommandozeile beendet sich mit derselben Meldung)."""


def fail(msg):
    raise MeshcutError(msg)


# ============================================================================= Netz
class Mesh:
    """Dreiecksnetz: V (n,3) float64, F (m,3) int64. Nur Geometrie (keine UV, Farben, Materialien)."""

    def __init__(self, V, F, name=""):
        V = np.ascontiguousarray(V, dtype=np.float64).reshape(-1, 3)
        F = np.ascontiguousarray(F, dtype=np.int64).reshape(-1, 3)
        if not np.isfinite(V).all():
            fail("Das Netz enthaelt NaN oder unendliche Koordinaten.")
        if F.size and (F.min() < 0 or F.max() >= len(V)):
            fail("Das Netz verweist auf Eckpunkte, die es nicht gibt.")
        self.V, self.F, self.name = V + 0.0, F, name           # + 0.0 macht aus -0.0 ein 0.0

    def __repr__(self):
        return "Mesh(%d Eckpunkte, %d Dreiecke)" % (len(self.V), len(self.F))

    def tris(self):
        return self.V[self.F]


def weld_triangles(P):
    """P: (m,3,3) Dreiecke als Koordinaten -> (V, F). Gleiche Koordinaten = ein Eckpunkt, Reihenfolge des ersten Auftretens."""
    m = len(P)
    if m == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    flat = (np.ascontiguousarray(P, dtype=np.float64).reshape(-1, 3)) + 0.0
    u, first, inv = np.unique(flat, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    return u[order], rank[inv.reshape(-1)].reshape(m, 3)


def welded(V, F, name=""):
    """Mesh aus Eckpunkten/Dreiecken, gleiche Koordinaten werden zu einem Eckpunkt verschmolzen, Unbenutztes faellt weg."""
    V = np.asarray(V, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    if not np.isfinite(V).all():
        fail("Das Netz enthaelt NaN oder unendliche Koordinaten.")
    if F.size and (F.min() < 0 or F.max() >= len(V)):
        fail("Das Netz verweist auf Eckpunkte, die es nicht gibt.")
    V2, F2 = weld_triangles(V[F])
    return Mesh(V2, F2, name)


def quantize(V, precision):
    V = np.asarray(V, dtype=np.float64)
    return (V.astype(np.float32).astype(np.float64) if precision == "float32" else V) + 0.0


def triangle_hash(V, F, precision="float64"):
    """SHA-256 ueber die Dreiecke, unabhaengig von Eckpunkt-/Dreiecksreihenfolge und Startecke, abhaengig von Umlaufsinn
    und Koordinaten (nach Rundung auf `precision`)."""
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    h = hashlib.sha256()
    h.update(b"meshcut-tris-1:" + precision.encode() + b":")
    if len(F):
        Vq = quantize(V, precision)
        u, inv = np.unique(Vq, axis=0, return_inverse=True)
        ids = inv.reshape(-1)[F]
        rots = [ids, ids[:, [1, 2, 0]], ids[:, [2, 0, 1]]]

        def less(A, B):
            return ((A[:, 0] < B[:, 0]) | ((A[:, 0] == B[:, 0]) & ((A[:, 1] < B[:, 1]) | ((A[:, 1] == B[:, 1]) & (A[:, 2] < B[:, 2])))))
        best = rots[0]
        for r in rots[1:]:
            best = np.where(less(r, best)[:, None], r, best)
        order = np.lexsort((best[:, 2], best[:, 1], best[:, 0]))
        h.update(np.ascontiguousarray(u[best[order]], dtype="<f8").tobytes())
    return h.hexdigest()


def volume_area(V, F):
    """Vorzeichenbehaftetes Volumen und Flaeche (relativ zum Schwerpunkt der Eckpunkte, das haelt die Rundung klein)."""
    if len(F) == 0:
        return 0.0, 0.0
    c = V.mean(axis=0)
    P = V[F] - c
    cr = np.cross(P[:, 1], P[:, 2])
    vol = float(np.einsum("ij,ij->i", P[:, 0], cr).sum() / 6.0)
    area = float(np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1).sum() / 2.0)
    return vol, area


def directed_edges(F):
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    a = F.reshape(-1)
    b = F[:, [1, 2, 0]].reshape(-1)
    keep = a != b                                               # entartete Kanten (a, a) zaehlen nicht
    return a[keep], b[keep]


def edge_report(F, nv=None):
    """Kanten-Kennzahlen: Anzahl Kanten, offene Kanten, nichtmanifold/falsch orientierte Kanten, wasserdicht?"""
    a, b = directed_edges(F)
    if len(a) == 0:
        return {"edges": 0, "open": 0, "bad": 0, "watertight": False}
    n = int(max(a.max(), b.max())) + 1 if nv is None else int(nv)
    fwd = a * n + b
    rev = b * n + a
    ukeys, cnt = np.unique(fwd, return_counts=True)
    dup = int((cnt > 1).sum())                                  # gleiche Richtung mehrfach = falsch orientiert / nichtmanifold
    has_rev = np.isin(ukeys, rev)
    open_ = int((~has_rev).sum())
    und = np.unique(np.minimum(a, b) * n + np.maximum(a, b))
    return {"edges": int(len(und)), "open": open_, "bad": dup, "watertight": open_ == 0 and dup == 0}


def components(F):
    """Anzahl zusammenhaengender Huellen (Union-Find ueber Eckpunkte der Dreiecke)."""
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    if len(F) == 0:
        return 0
    used, inv = np.unique(F, return_inverse=True)
    G = inv.reshape(-1, 3)
    parent = list(range(len(used)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, c in G.tolist():
        ra, rb, rc = find(a), find(b), find(c)
        if ra != rb:
            parent[rb] = ra
        if ra != rc:
            parent[find(rc)] = find(ra)
    return len({find(i) for i in range(len(used))})


def pinch_count(F):
    """Anzahl zusätzlicher Fächer an Eckpunkten, an denen sich Flächenteile nur in einem Punkt berühren (nichtmanifold-Eckpunkt)."""
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    if len(F) == 0:
        return 0
    n = int(F.max()) + 1
    parent = list(range(3 * len(F)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    slot = {}                                                   # (Eckpunkt, Kante-Gegenpunkt) -> Ecke einer Flaeche
    for i, (a, b, c) in enumerate(F.tolist()):
        for k, (v, w1, w2) in enumerate(((a, b, c), (b, c, a), (c, a, b))):
            node = 3 * i + k
            for w in (w1, w2):
                key = v * n + w
                if key in slot:
                    ra, rb = find(slot[key]), find(node)
                    if ra != rb:
                        parent[rb] = ra
                else:
                    slot[key] = node
    comps = {}
    for i, row in enumerate(F.tolist()):
        for k, v in enumerate(row):
            comps.setdefault(v, set()).add(find(3 * i + k))
    return int(sum(len(c) - 1 for c in comps.values()))


def mesh_stats(V, F, precision="float64"):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    vol, area = volume_area(V, F)
    er = edge_report(F, len(V))
    used = np.unique(F) if len(F) else np.zeros(0, np.int64)
    nv = len(used)
    pinch = pinch_count(F)
    chi = nv + pinch - er["edges"] + len(F)                     # Eckpunkte, an denen sich Teile nur beruehren, zaehlen getrennt
    if len(used):
        lo, hi = V[used].min(axis=0), V[used].max(axis=0)
    else:
        lo = hi = np.zeros(3)
    shells = components(F)
    return {"vertices": int(nv), "faces": int(len(F)), "edges": er["edges"], "open_edges": er["open"], "bad_edges": er["bad"],
            "watertight": er["watertight"], "euler": int(chi), "pinched": pinch, "shells": int(shells), "volume": vol, "area": area,
            "bbox_min": lo.tolist(), "bbox_max": hi.tolist(), "diag": float(np.linalg.norm(hi - lo)),
            "hash": triangle_hash(V, F, precision)}


# ============================================================================= Ebenen
class Plane:
    """Ebene n.p = d mit Einheitsnormale n. Achsparallele Ebenen rechnen auf der Koordinate selbst (exakt)."""

    def __init__(self, normal, offset, weight=None):
        n = np.asarray(normal, dtype=np.float64).reshape(3)
        ln = float(np.linalg.norm(n))
        if not np.isfinite(n).all() or not math.isfinite(float(offset)) or ln == 0:
            fail("Ungueltige Ebene (Normale 0 oder keine Zahl).")
        axis = [i for i in range(3) if n[i] != 0]
        if len(axis) == 1:                                      # achsparallel: exakt, ohne Normalisierung
            k = axis[0]
            c = float(n[k])
            n = np.zeros(3)
            n[k] = 1.0 if c > 0 else -1.0
            d = float(offset) / abs(c)
            self.axis = k
        else:
            n = n / ln
            d = float(offset) / ln
            self.axis = None
        self.n, self.d, self.weight = n, d, weight

    def signed(self, V):
        if self.axis is not None:
            return (V[:, self.axis] * self.n[self.axis]) - self.d
        return V @ self.n - self.d

    def __repr__(self):
        return "Plane(n=%s, d=%r)" % (self.n.tolist(), self.d)

    def to_json(self):
        return {"normal": self.n.tolist(), "offset": self.d, "weight": self.weight}


AXES = {"x": 0, "y": 1, "z": 2}


def parse_plane(spec):
    """'x=1.5' | 'z=-2' | 'a,b,c,d' (a*x+b*y+c*z = d) | Plane | (normal, offset)."""
    if isinstance(spec, Plane):
        return spec
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return Plane(spec[0], spec[1])
    s = str(spec).strip().lower().replace(" ", "")
    m = re.fullmatch(r"([xyz])=([-+]?[0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", s)
    if m:
        n = np.zeros(3)
        n[AXES[m.group(1)]] = 1.0
        return Plane(n, float(m.group(2)))
    parts = s.split(",")
    if len(parts) == 4:
        try:
            a, b, c, d = (float(p) for p in parts)
        except ValueError:
            pass
        else:
            return Plane([a, b, c], d)
    fail("Ebene '%s' nicht verstanden - Beispiele: x=12.5   z=-3   1,1,0,4 (Normale a,b,c und Abstand d)" % spec)


def parse_counts(text, what):
    m = re.fullmatch(r"(\d+)x(\d+)(?:x(\d+))?", str(text).strip().lower())
    if not m:
        fail("%s '%s' nicht verstanden - Beispiel: 2x2x3 (x mal y mal z) oder 4x3" % (what, text))
    vals = [int(g) for g in m.groups() if g is not None]
    if len(vals) == 2:
        vals.append(1)
    if min(vals) < 1:
        fail("%s: jede Zahl muss mindestens 1 sein." % what)
    return vals


def parse_sizes(text):
    m = re.fullmatch(r"([0-9.]+)x([0-9.]+)(?:x([0-9.]+))?", str(text).strip().lower())
    if not m:
        fail("--max '%s' nicht verstanden - Beispiel: 220x220x250 (Druckraum in x, y, z)" % text)
    try:
        vals = [float(g) for g in m.groups() if g is not None]
    except ValueError:
        fail("--max '%s' nicht verstanden." % text)
    if len(vals) == 2:
        vals.append(math.inf)
    if min(vals) <= 0:
        fail("--max: jede Groesse muss groesser als 0 sein.")
    return vals


def check_cell_count(counts):
    cells = 1
    for c in counts:
        cells *= int(c)
    if cells > MAX_PARTS:
        fail("Das ergibt %d Zellen - mehr als %d Teile sind nicht erlaubt." % (cells, MAX_PARTS))


def grid_planes(lo, hi, counts):
    """Gitter ueber den Begrenzungsquader: counts=(nx,ny,nz) Zellen. Ebenen aufsteigend je Achse, Gewicht = Zellenschritt."""
    planes = []
    step = 1
    for ax in range(3):
        n = counts[ax]
        for i in range(1, n):
            normal = np.zeros(3)
            normal[ax] = 1.0
            p = Plane(normal, lo[ax] + (hi[ax] - lo[ax]) * i / n, weight=step)
            planes.append(p)
        step *= n
    return planes


def wedge_planes(center, axis, n):
    """n Keile (gerade Zahl) um eine Achse: n/2 Ebenen durch die Achse. Rueckgabe: (Ebenen, Muster->Keilnummer)."""
    if n < 2 or n % 2:
        fail("--wedges: gerade Zahl noetig, mindestens 2 (ungerade Keile brauchen Halbebenen, das kann meshcut noch nicht).")
    if n // 2 > MAX_PLANES:
        fail("--wedges: zu viele Keile.")
    u, v = (axis + 1) % 3, (axis + 2) % 3
    delta = 2 * math.pi / n
    planes = []
    for k in range(n // 2):
        th = k * delta
        normal = np.zeros(3)
        normal[u], normal[v] = -math.sin(th), math.cos(th)
        # Ebene durch die Achse: n.p = n.center  (center liegt auf der Achse)
        planes.append(Plane(normal, float(normal @ center), weight=1 << k))
    table = {}
    for m in range(n):
        phi = (m + 0.5) * delta
        mask = 0
        for k in range(n // 2):
            if math.sin(phi - k * delta) > 0:
                mask |= 1 << k
        table[mask] = m
    return planes, table


# ============================================================================= Deckel (Dreiecke in einer Schnittflaeche)
def _cross(P, a, b, c):
    return (P[b][0] - P[a][0]) * (P[c][1] - P[a][1]) - (P[b][1] - P[a][1]) * (P[c][0] - P[a][0])


def _signed_area(P, loop):
    s = 0.0
    n = len(loop)
    for i in range(n):
        x1, y1 = P[loop[i]]
        x2, y2 = P[loop[(i + 1) % n]]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _in_poly(P, loop, pt):
    x, y = pt
    inside = False
    n = len(loop)
    j = n - 1
    for i in range(n):
        xi, yi = P[loop[i]]
        xj, yj = P[loop[j]]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def trace_loops(edges, P):
    """Gerichtete Randkanten (a,b) zu geschlossenen Zyklen. An Verzweigungen nimmt man die staerkste Linkskurve, dann
    bleiben beruehrende Teile getrennte, einfache Umrisse."""
    out = {}
    for i, (a, b) in enumerate(edges):
        out.setdefault(a, []).append(i)
    used = [False] * len(edges)
    loops = []
    for i0 in range(len(edges)):
        if used[i0]:
            continue
        loop = []
        i = i0
        start = edges[i0][0]
        while True:
            used[i] = True
            a, b = edges[i]
            loop.append(a)
            cands = [j for j in out.get(b, ()) if not used[j]]
            if b == start:
                cands.append(i0)
            if not cands:
                raise MeshcutError("Interner Fehler: Rand der Schnittflaeche ist nicht geschlossen.")
            if len(cands) == 1:
                j = cands[0]
            else:
                dx, dy = P[b][0] - P[a][0], P[b][1] - P[a][1]
                best, j = -10.0, cands[0]
                for jj in cands:
                    ex, ey = P[edges[jj][1]][0] - P[b][0], P[edges[jj][1]][1] - P[b][1]
                    ang = math.atan2(dx * ey - dy * ex, dx * ex + dy * ey)
                    if ang > best:
                        best, j = ang, jj
            if j == i0:
                break
            i = j
        loops.append(loop)
    return loops


def _segment_blocked(P, h, c, ex, ey):
    """Schneidet die Strecke h-c eine der Kanten (ex[k]->ey[k])? Beruehren an den Endpunkten h/c zaehlt nicht."""
    hx, hy = P[h]
    cx, cy = P[c]
    A, B = ex, ey
    o1 = (cx - hx) * (A[:, 1] - hy) - (cy - hy) * (A[:, 0] - hx)
    o2 = (cx - hx) * (B[:, 1] - hy) - (cy - hy) * (B[:, 0] - hx)
    dx, dy = B[:, 0] - A[:, 0], B[:, 1] - A[:, 1]
    o3 = dx * (hy - A[:, 1]) - dy * (hx - A[:, 0])
    o4 = dx * (cy - A[:, 1]) - dy * (cx - A[:, 0])
    touch = (((A[:, 0] == hx) & (A[:, 1] == hy)) | ((B[:, 0] == hx) & (B[:, 1] == hy))
             | ((A[:, 0] == cx) & (A[:, 1] == cy)) | ((B[:, 0] == cx) & (B[:, 1] == cy)))
    proper = (o1 * o2 < 0) & (o3 * o4 < 0)
    seg2 = (cx - hx) ** 2 + (cy - hy) ** 2

    def on_seg(px, py, o):
        t = ((px - hx) * (cx - hx) + (py - hy) * (cy - hy))
        return (o == 0) & (t > 0) & (t < seg2)
    onseg = on_seg(A[:, 0], A[:, 1], o1) | on_seg(B[:, 0], B[:, 1], o2)
    return bool(((proper | onseg) & ~touch).any())


def _in_cone(P, prv, v, nxt, t):
    """Liegt die Richtung v->t im Innenwinkel bei v (Polygon gegen den Uhrzeigersinn, Innen links)?"""
    left_next = _cross(P, v, nxt, t) > 0
    left_prev = _cross(P, prv, v, t) > 0
    if _cross(P, prv, v, nxt) > 0:
        return left_next and left_prev
    return left_next or left_prev


def bridge_holes(P, outer, holes):
    """Loecher durch Brueckenkanten in den Aussenumriss einhaengen (Ergebnis: ein schwach einfaches Polygon, gegen den Uhrzeigersinn)."""
    poly = list(outer)
    pending = sorted(holes, key=lambda h: min((P[i][0], P[i][1]) for i in h))
    while pending:
        hole = pending.pop(0)
        hs = min(range(len(hole)), key=lambda k: (P[hole[k]][0], P[hole[k]][1]))
        h = hole[hs]
        rot = hole[hs:] + hole[:hs]
        edges_a, edges_b = [], []
        for loop in [poly] + pending + [hole]:
            L = len(loop)
            for k in range(L):
                edges_a.append(P[loop[k]])
                edges_b.append(P[loop[(k + 1) % L]])
        EA, EB = np.array(edges_a, dtype=np.float64), np.array(edges_b, dtype=np.float64)
        uniq = list(dict.fromkeys(poly))
        hx, hy = P[h]
        uniq.sort(key=lambda c: (P[c][0] - hx) ** 2 + (P[c][1] - hy) ** 2)
        done = False
        for c in uniq:
            positions = [k for k, vv in enumerate(poly) if vv == c]
            pos = None
            L = len(poly)
            for k in positions:
                if _in_cone(P, poly[(k - 1) % L], poly[k], poly[(k + 1) % L], h):
                    pos = k
                    break
            if pos is None:
                continue
            if c != h and P[c] != P[h]:
                if _segment_blocked(P, h, c, EA, EB):
                    continue
                mid = ((P[h][0] + P[c][0]) / 2, (P[h][1] + P[c][1]) / 2)
                if not _in_poly(P, poly, mid):
                    continue
            poly = poly[:pos + 1] + rot + [h] + [c] + poly[pos + 1:]
            done = True
            break
        if not done:
            raise MeshcutError("Interner Fehler: Loch der Schnittflaeche liess sich nicht anbinden.")
    return poly


def ear_clip(P, poly):
    """Ohrenschnitt. Jeder Eckpunkt bleibt erhalten (sonst entstuenden T-Kreuzungen zur Nachbarflaeche); gerade Eckpunkte
    ergeben im Notfall flache Dreiecke. Rueckgabe: Dreiecke (a,b,c) gegen den Uhrzeigersinn, Anzahl Notfaelle."""
    L = len(poly)
    prv = [(i - 1) % L for i in range(L)]
    nxt = [(i + 1) % L for i in range(L)]
    pts = [P[v] for v in poly]

    def cr(i, j, k):
        return (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) - (pts[j][1] - pts[i][1]) * (pts[k][0] - pts[i][0])
    reflex = {i for i in range(L) if cr(prv[i], i, nxt[i]) <= 0}
    tris = []
    emergencies = 0
    remaining = L
    cur = 0

    def clip(b):
        nonlocal remaining
        a, c = prv[b], nxt[b]
        tris.append((poly[a], poly[b], poly[c]))
        area_left[0] -= cr(a, b, c) / 2.0
        nxt[a], prv[c] = c, a
        remaining -= 1
        for q in (a, c):
            if q in reflex and cr(prv[q], q, nxt[q]) > 0:
                reflex.discard(q)
        reflex.discard(b)

    area0 = abs(sum(cr(0, i, nxt[i]) for i in range(L) if i != 0 and nxt[i] != 0)) / 2.0
    area_left = [area0]

    def is_ear(b):
        a, c = prv[b], nxt[b]
        t = cr(a, b, c)
        if not t > 0:
            return False
        if remaining > 3 and area_left[0] - t / 2.0 <= 1e-12 * area0:   # der Rest waere flach: gerade Eckpunkte ohne Flaeche
            return False
        pa, pb, pc = pts[a], pts[b], pts[c]
        for p in reflex:
            if p == a or p == b or p == c:
                continue
            q = pts[p]
            if q == pa or q == pb or q == pc:
                continue
            d1 = (pb[0] - pa[0]) * (q[1] - pa[1]) - (pb[1] - pa[1]) * (q[0] - pa[0])
            d2 = (pc[0] - pb[0]) * (q[1] - pb[1]) - (pc[1] - pb[1]) * (q[0] - pb[0])
            d3 = (pa[0] - pc[0]) * (q[1] - pc[1]) - (pa[1] - pc[1]) * (q[0] - pc[0])
            if d1 >= 0 and d2 >= 0 and d3 >= 0:
                return False
        return True
    while remaining > 3:
        found = None
        b = cur
        for _ in range(remaining):
            if is_ear(b):
                found = b
                break
            b = nxt[b]
        if found is None:                                       # kein echtes Ohr: gerade Eckpunkte ohne Spitze, dann irgendeins
            b = cur
            pick = None
            for _ in range(remaining):
                a, c = prv[b], nxt[b]
                if cr(a, b, c) == 0 and ((pts[b][0] - pts[a][0]) * (pts[c][0] - pts[b][0]) + (pts[b][1] - pts[a][1]) * (pts[c][1] - pts[b][1])) > 0:
                    pick = b
                    break
                b = nxt[b]
            if pick is None:
                emergencies += 1
                b = cur
                pick, bestv = cur, -math.inf
                for _ in range(remaining):
                    v = cr(prv[b], b, nxt[b])
                    if v > bestv:
                        pick, bestv = b, v
                    b = nxt[b]
            found = pick
        cur = nxt[found]
        clip(found)
    if remaining == 3:
        a = cur
        tris.append((poly[a], poly[nxt[a]], poly[nxt[nxt[a]]]))
    return tris, emergencies


def cap_triangles(edges, P3, normal, flip):
    """Deckel fuer die Randkanten eines offenen Teils. `edges`: gerichtete Randkanten (a,b) des Teils; der Deckel braucht die
    Gegenrichtung. P3: Koordinaten der Eckpunkte, normal: Ebenennormale. flip=True: Deckel schaut gegen `normal`.
    Rueckgabe: Liste von Dreiecken (Eckpunktnummern), Anzahl Notfaelle."""
    n = np.asarray(normal, dtype=np.float64)
    e0 = int(np.argmin(np.abs(n)))
    ref = np.zeros(3)
    ref[e0] = 1.0
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    verts = sorted({x for e in edges for x in e})
    X = P3[verts]
    X = X - X.mean(axis=0)                  # um den Schwerpunkt rechnen: weit vom Ursprung loescht sich sonst die Flaeche aus
    sx = -1.0 if flip else 1.0
    P = {vid: (float(sx * (X[k] @ u)), float(X[k] @ v)) for k, vid in enumerate(verts)}
    rev = [(b, a) for a, b in edges]
    loops = trace_loops(rev, P)
    outers, holes = [], []
    for lp in loops:
        if len(lp) < 3:
            continue
        a = _signed_area(P, lp)
        (outers if a > 0 else holes).append((abs(a), lp))
    if holes and not outers:
        raise MeshcutError("Die Schnittflaeche hat nur Loecher - das Netz ist vermutlich nach innen orientiert (--flip probieren).")
    owner = {i: [] for i in range(len(outers))}
    for _, hp in holes:
        samples = hp[:5]
        best, bi = None, None
        for i, (oa, ol) in enumerate(outers):
            votes = sum(_in_poly(P, ol, P[s]) for s in samples)
            if votes * 2 > len(samples) and (best is None or oa < best):
                best, bi = oa, i
        if bi is None:
            raise MeshcutError("Interner Fehler: ein Loch der Schnittflaeche liegt in keinem Umriss.")
        owner[bi].append(hp)
    tris, emerg = [], 0
    for i, (_, ol) in enumerate(outers):
        poly = bridge_holes(P, ol, owner[i]) if owner[i] else list(ol)
        t, e = ear_clip(P, poly)
        tris.extend(t)
        emerg += e
    return tris, emerg


# ============================================================================= Schnitt
class _State:
    def __init__(self, V, F, src, kind, label):
        self.V, self.F, self.src, self.kind, self.label = V, F, src, kind, label


def _settle_flat_faces(F, flat_idx, bit, crossing, label):
    """Dreiecke ohne Flaeche, die in der Schnittebene liegen, haben keine Normale. Sie gehen auf die Seite ihrer Nachbarn
    (gleiches Teil), sonst wuerde ein Teil an ihnen in zwei Gruppen zerfallen."""
    unresolved = set(flat_idx.tolist())
    edges = {}
    for i in range(len(F)):
        a, b, c = F[i].tolist()
        for u, v in ((a, b), (b, c), (c, a)):
            if u != v:
                edges.setdefault((min(u, v), max(u, v)), []).append(i)
    for _ in range(64):
        if not unresolved:
            return
        progress = False
        for i in sorted(unresolved):
            a, b, c = F[i].tolist()
            votes = []
            for u, v in ((a, b), (b, c), (c, a)):
                for j in edges.get((min(u, v), max(u, v)), ()):
                    if j != i and j not in unresolved and not crossing[j] and label[j] == label[i]:
                        votes.append(int(bit[j]))
            if votes:
                bit[i] = 1 if sum(votes) * 2 >= len(votes) else 0
                unresolved.discard(i)
                progress = True
        if not progress:
            break
    for i in unresolved:
        bit[i] = 1


def _alias_flat_edges(V, Fc, Sc, emap, nV):
    """Flache Dreiecke (a,b,c mit b auf der Strecke a-c) entstehen als Fuellstuecke zwischen Deckel und Schnittkante. Schneidet die
    Ebene die lange Kante a-c, so liegt der Schnittpunkt auf a-b oder b-c - dort gibt es ihn schon (die Nachbarn teilen diese
    Kanten). Ohne diese Verknuepfung entstuenden zwei Eckpunkte an derselben Stelle."""
    e = np.stack([V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 1]], V[Fc[:, 0]] - V[Fc[:, 2]]], axis=1)       # Kanten 0-1, 1-2, 2-0
    ln = np.linalg.norm(e, axis=2)
    area2 = np.linalg.norm(np.cross(e[:, 0], -e[:, 2]), axis=1)
    longest = ln.max(axis=1)
    flat = area2 <= 1e-9 * longest ** 2
    for r in np.nonzero(flat)[0].tolist():
        m = int(np.argmax(ln[r]))                                # lange Kante: Ecke m -> m+1, gegenueber liegt b
        ia, ib, ic = Fc[r, m], Fc[r, (m + 1) % 3], Fc[r, (m + 2) % 3]
        a, c, b = int(ia), int(ib), int(ic)
        sa, sc, sb = int(Sc[r, m]), int(Sc[r, (m + 1) % 3]), int(Sc[r, (m + 2) % 3])
        key = min(a, c) * nV + max(a, c)
        if key not in emap:
            continue
        if sa * sb < 0:
            emap[key] = emap[min(a, b) * nV + max(a, b)]
        elif sb * sc < 0:
            emap[key] = emap[min(b, c) * nV + max(b, c)]
        elif sb == 0:
            emap[key] = b


def _cut_once(st, plane, cap, report, snap):
    V, F = st.V, st.F
    s = plane.signed(V)
    sg = np.sign(s).astype(np.int8)
    if snap > 0:                                                # Rundungsrauschen (z. B. cos(90 Grad)) gilt als 'auf der Ebene'
        extent = float((V.max(axis=0) - V.min(axis=0)).max())
        mag = float(np.abs(V).max()) + abs(plane.d)
        # relativ zur Groesse des Netzes (nicht zur Entfernung vom Ursprung) + die Rechenungenauigkeit der Koordinaten selbst
        sg[np.abs(s) <= snap * extent + 8 * np.finfo(np.float64).eps * mag * (snap > 0) + 1e-300] = 0
    S = sg[F]
    zero_all = (S == 0).all(axis=1)
    ge = (S >= 0).all(axis=1)
    le = (S <= 0).all(axis=1)
    crossing = ~(ge | le)
    bit = np.zeros(len(F), np.int64)
    bit[ge & ~zero_all] = 1
    if zero_all.any():                                          # Dreieck liegt in der Ebene: Material liegt hinter ihm
        zi = np.nonzero(zero_all)[0]
        e1, e2 = V[F[zi, 1]] - V[F[zi, 0]], V[F[zi, 2]] - V[F[zi, 0]]
        nrm = np.cross(e1, e2)
        bit[zi] = np.where(nrm @ plane.n > 0, 0, 1)
        flat = np.linalg.norm(nrm, axis=1) <= 1e-12 * np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
        if flat.any():                                          # Flaeche null: keine Normale, die Seite der Nachbarn nehmen
            _settle_flat_faces(F, zi[flat], bit, crossing, st.label)
    w = int(plane.weight)

    ci = np.nonzero(crossing)[0]
    nV = len(V)
    newV = np.zeros((0, 3))
    emap = {}
    if len(ci):
        Fc, Sc = F[ci], S[ci]
        los, his = [], []
        for k in range(3):
            a, b = Fc[:, k], Fc[:, (k + 1) % 3]
            m = (Sc[:, k].astype(np.int16) * Sc[:, (k + 1) % 3].astype(np.int16)) < 0
            los.append(np.minimum(a, b)[m])
            his.append(np.maximum(a, b)[m])
        lo, hi = np.concatenate(los), np.concatenate(his)
        keys = np.unique(lo * nV + hi)
        klo, khi = keys // nV, keys % nV
        t = s[klo] / (s[klo] - s[khi])                          # Endpunkte sortiert: beide Seiten rechnen dasselbe
        newV = V[klo] + t[:, None] * (V[khi] - V[klo])
        emap = {int(k): nV + i for i, k in enumerate(keys.tolist())}
        _alias_flat_edges(V, Fc, Sc, emap, nV)
    V2 = np.concatenate([V, newV]) if len(newV) else V
    onp = np.concatenate([sg == 0, np.ones(len(newV), bool)])

    frag_f, frag_bit, frag_src, frag_kind, frag_lab = [], [], [], [], []
    Fl, Sl = F[ci].tolist(), S[ci].tolist()
    srcc, kindc, labc = st.src[ci].tolist(), st.kind[ci].tolist(), st.label[ci].tolist()
    for f, g, sr, kd, lb in zip(Fl, Sl, srcc, kindc, labc):
        pos, neg = [], []
        for k in range(3):
            i, j = f[k], f[(k + 1) % 3]
            gi, gj = g[k], g[(k + 1) % 3]
            if gi >= 0:
                pos.append(i)
            if gi <= 0:
                neg.append(i)
            if gi * gj < 0:
                e = emap[min(i, j) * nV + max(i, j)]
                pos.append(e)
                neg.append(e)
        for poly, bt in ((pos, 1), (neg, 0)):
            r0 = poly.index(min(poly))                          # Faecher von der kleinsten Eckpunktnummer: gespiegelte Deckel
            poly = poly[r0:] + poly[:r0]                        # zerfallen so in dieselben Dreiecke (Join ohne Begleitdatei)
            for q in range(1, len(poly) - 1):
                if poly[0] == poly[q] or poly[q] == poly[q + 1] or poly[0] == poly[q + 1]:
                    continue                                   # Dreieck mit doppeltem Eckpunkt: keine Flaeche, keine Kante
                frag_f.append((poly[0], poly[q], poly[q + 1]))
                frag_bit.append(bt)
                frag_src.append(sr)
                frag_kind.append(kd)
                frag_lab.append(lb)
    keep = ~crossing
    Fn = np.concatenate([F[keep], np.array(frag_f, dtype=np.int64).reshape(-1, 3)])
    bitn = np.concatenate([bit[keep], np.array(frag_bit, dtype=np.int64)])
    srcn = np.concatenate([st.src[keep], np.array(frag_src, dtype=np.int64)])
    kindn = np.concatenate([st.kind[keep], np.array(frag_kind, dtype=np.int8)])
    labn = np.concatenate([st.label[keep], np.array(frag_lab, dtype=np.int64)])

    cap_f, cap_src, cap_kind, cap_lab, cap_bit = [], [], [], [], []
    if cap and len(Fn):
        key = labn * 2 + bitn
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = np.nonzero(np.r_[True, ks[1:] != ks[:-1]])[0]
        ends = np.r_[starts[1:], len(ks)]
        pos_groups = {}
        for s0, e0 in zip(starts, ends):
            kv = int(ks[s0])
            if kv & 1:
                pos_groups[kv >> 1] = order[s0:e0]
        for lab, idx in pos_groups.items():
            Fg = Fn[idx]
            a, b = directed_edges(Fg)
            n = len(V2)
            fwd, rv = a * n + b, b * n + a
            bnd = ~np.isin(fwd, rv)
            if not bnd.any():
                continue
            ba, bb = a[bnd], b[bnd]
            if not (onp[ba] & onp[bb]).all():
                raise MeshcutError("Das Netz ist nicht geschlossen: der Schnitt trifft eine offene Kante (--no-cap schneidet ohne Deckel).")
            tris, emerg = cap_triangles(list(zip(ba.tolist(), bb.tolist())), V2, plane.n, flip=True)
            report["cap_emergencies"] += emerg
            for (x, y, z) in tris:
                cap_f.append((x, y, z)); cap_bit.append(1)
                cap_f.append((x, z, y)); cap_bit.append(0)
                cap_lab.extend([lab, lab])
    if cap_f:
        Fn = np.concatenate([Fn, np.array(cap_f, dtype=np.int64)])
        bitn = np.concatenate([bitn, np.array(cap_bit, dtype=np.int64)])
        srcn = np.concatenate([srcn, np.full(len(cap_f), -1, np.int64)])
        kindn = np.concatenate([kindn, np.ones(len(cap_f), np.int8)])
        labn = np.concatenate([labn, np.array(cap_lab, dtype=np.int64)])
    return _State(V2, Fn, srcn, kindn, labn + w * bitn)


class Part:
    def __init__(self, id, V, F, src, n_cap):
        self.id, self.V, self.F, self.src, self.n_cap = id, V, F, src, n_cap     # Dreiecke: erst Originalfragmente, dann Deckel

    def mesh(self):
        return Mesh(self.V, self.F)


class SplitResult:
    def __init__(self, parts, mode, planes, source, split_ids, split_coords, report):
        self.parts, self.mode, self.planes, self.source = parts, mode, planes, source
        self.split_ids, self.split_coords, self.report = split_ids, split_coords, report


def _compact(V, F):
    used, inv = np.unique(F, return_inverse=True)
    return V[used], inv.reshape(-1, 3).astype(np.int64)


def _build_parts(V, F, src, kind, label, table=None, compress=False):
    parts = []
    ids = np.unique(label) if len(label) else []
    if compress:
        table = {int(l): i for i, l in enumerate(ids)}
    for lab in ids:
        idx = np.nonzero(label == lab)[0]
        idx = idx[np.argsort(kind[idx], kind="stable")]         # Originalfragmente zuerst, Deckel danach
        Vp, Fp = _compact(V, F[idx])
        pid = int(lab) if table is None else table[int(lab)]
        parts.append(Part(pid, Vp, Fp, src[idx][kind[idx] == 0].copy(), int((kind[idx] == 1).sum())))
    parts.sort(key=lambda p: p.id)
    return parts


def check_cuttable(mesh, flip=False):
    er = edge_report(mesh.F, len(mesh.V))
    if not er["watertight"]:
        fail("Das Netz ist nicht wasserdicht (%d offene, %d mehrfach/falsch orientierte Kanten) - ein Deckel ist nicht moeglich. "
             "Mit --no-cap ohne Deckel schneiden." % (er["open"], er["bad"]))
    vol, _ = volume_area(mesh.V, mesh.F)
    if vol < 0 and not flip:
        fail("Das Netz ist nach innen orientiert (negatives Volumen) - mit --flip umdrehen.")


def split_planes(mesh, planes, cap=True, table=None, mode="plane", snap=1e-12):
    """Zerlegt `mesh` an den Ebenen. Jede Ebene hat ein Gewicht; Teilnummer = Summe der Gewichte der Ebenen, hinter denen
    das Teil liegt (Gitter: Zellnummer, sonst Bitmaske)."""
    if not planes:
        fail("Keine Ebene angegeben.")
    if len(mesh.F) == 0:
        fail("Das Netz hat keine Dreiecke.")
    if cap:
        check_cuttable(mesh)
    st = _State(mesh.V.copy(), mesh.F.copy(), np.arange(len(mesh.F), dtype=np.int64),
                np.zeros(len(mesh.F), np.int8), np.zeros(len(mesh.F), np.int64))
    report = {"cap_emergencies": 0}
    for k, p in enumerate(planes):
        if p.weight is None:
            p.weight = 1 << k
        st = _cut_once(st, p, cap, report, snap)
    parts = _build_parts(st.V, st.F, st.src, st.kind, st.label, table, compress=(mode == "plane"))
    if len(parts) > MAX_PARTS:
        fail("Das ergibt %d Teile - mehr als %d sind nicht erlaubt." % (len(parts), MAX_PARTS))
    if len(parts) < 2:
        fail("Die Ebene(n) schneiden das Netz nicht - es bleibt ein einziges Teil.")
    srcs = np.concatenate([p.src for p in parts])
    u, c = np.unique(srcs, return_counts=True)
    split_ids = u[c > 1]
    split_coords = mesh.V[mesh.F[split_ids]]
    return SplitResult(parts, mode, planes, mesh, split_ids, split_coords, report)


def split_labels(mesh, labels):
    labels = np.asarray(labels)
    if labels.shape != (len(mesh.F),):
        fail("Die Zuordnungskarte hat %s Werte, das Netz %d Dreiecke - ein Wert pro Dreieck noetig." % (labels.size, len(mesh.F)))
    if labels.dtype.kind == "f":
        if not np.isfinite(labels).all() or (labels != np.floor(labels)).any():
            fail("Die Zuordnungskarte muss ganze Zahlen enthalten.")
    elif labels.dtype.kind not in "iub":
        fail("Die Zuordnungskarte muss ganze Zahlen enthalten.")
    labels = labels.astype(np.int64)
    if len(labels) and labels.min() < 0:
        fail("Teilnummern duerfen nicht negativ sein.")
    ids = np.unique(labels)
    if len(ids) > MAX_PARTS:
        fail("Das sind %d Teile - mehr als %d sind nicht erlaubt." % (len(ids), MAX_PARTS))
    if len(ids) < 2:
        fail("Die Zuordnungskarte enthaelt nur ein Teil.")
    parts = []
    for lab in ids:
        idx = np.nonzero(labels == lab)[0]
        Vp, Fp = _compact(mesh.V, mesh.F[idx])
        parts.append(Part(int(lab), Vp, Fp, idx.astype(np.int64), 0))
    return SplitResult(parts, "labels", [], mesh, np.zeros(0, np.int64), np.zeros((0, 3, 3)), {"cap_emergencies": 0})


# ============================================================================= Zusammensetzen
def join_with_sidecar(parts_data, side):
    """parts_data: Liste (id, V, F) in Dateiform. Setzt die urspruenglichen Dreiecke wieder her (bitgenau)."""
    src_meta = side["parts"]
    by_id = {p[0]: p for p in parts_data}
    tri_list, src_list = [], []
    for meta in src_meta:
        pid = meta["id"]
        if pid not in by_id:
            fail("Teil %d fehlt." % pid)
        _, V, F = by_id[pid]
        ncap = meta["cap_faces"]
        if len(F) != meta["faces"]:
            fail("Teil %d hat %d Dreiecke, laut %s waren es %d - die Datei wurde veraendert." % (pid, len(F), SIDECAR, meta["faces"]))
        src = dec_array(meta["src"])
        nonc = len(F) - ncap
        if len(src) != nonc:
            fail("Die Begleitdatei passt nicht zu Teil %d." % pid)
        tri_list.append(V[F[:nonc]])
        src_list.append(src)
    P = np.concatenate(tri_list) if tri_list else np.zeros((0, 3, 3))
    S = np.concatenate(src_list) if src_list else np.zeros(0, np.int64)
    split_ids = dec_array(side["split_ids"])
    split_coords = dec_array(side["split_coords"]).reshape(-1, 3, 3)
    if len(split_ids):
        is_split = np.isin(S, split_ids)
        if not np.isin(split_ids, S[is_split]).all():
            fail("Teile fehlen: zerschnittene Dreiecke sind nicht vollstaendig vorhanden.")
        P = np.concatenate([P[~is_split], split_coords])
        S = np.concatenate([S[~is_split], split_ids])
    order = np.argsort(S, kind="stable")
    P, S = P[order], S[order]
    nfaces = side["source"]["faces"]
    if len(S) != nfaces or not (S == np.arange(nfaces)).all():
        fail("Die Teile ergeben nicht alle %d Dreiecke des Originals (Teil fehlt oder doppelt)." % nfaces)
    V, F = weld_triangles(P)
    return Mesh(V, F)


def join_plain(meshes):
    """Ohne Begleitdatei: verschmelzen und Deckelpaare (gleiche Eckpunkte, gegenlaeufig) entfernen. Naeherung: die Dreiecke
    bleiben so fein, wie der Schnitt sie gemacht hat."""
    P = np.concatenate([m.V[m.F] for m in meshes]) if meshes else np.zeros((0, 3, 3))
    V, F = weld_triangles(P)
    if len(F):
        key = np.sort(F, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        # Paar = zwei Dreiecke mit denselben Eckpunkten und entgegengesetztem Umlaufsinn
        rot = np.argmin(F, axis=1)
        canon = np.take_along_axis(F, (rot[:, None] + np.arange(3)) % 3, 1)
        sign = np.where(canon[:, 1] < canon[:, 2], 1, -1)
        drop = np.zeros(len(F), bool)
        order = np.argsort(inv, kind="stable")
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and inv[order[j]] == inv[order[i]]:
                j += 1
            grp = order[i:j]
            if len(grp) >= 2:
                pos_i = [g for g in grp if sign[g] > 0]
                neg_i = [g for g in grp if sign[g] < 0]
                for a, b in zip(pos_i, neg_i):
                    drop[a] = drop[b] = True
            i = j
        F = F[~drop]
    used, inv2 = np.unique(F, return_inverse=True)
    return Mesh(V[used], inv2.reshape(-1, 3))


# ============================================================================= Hausdorff (Eckpunkt zu Flaeche)
def _closest_points_on_triangles(p, a, b, c):
    """Ericson: naechster Punkt auf Dreieck abc zu p (alles (k,3))."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        res = a.copy()
        done = np.zeros(len(p), bool)

        def setv(mask, val):
            nonlocal done
            m = mask & ~done
            res[m] = val[m]
            done |= m
        setv((d1 <= 0) & (d2 <= 0), a)
        setv((d3 >= 0) & (d4 <= d3), b)
        setv((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + (d1 / (d1 - d3))[:, None] * ab)
        setv((d6 >= 0) & (d5 <= d6), c)
        setv((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + (d2 / (d2 - d6))[:, None] * ac)
        setv((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None] * (c - b))
        den = va + vb + vc
        v = vb / den
        w = vc / den
        inner = a + ab * v[:, None] + ac * w[:, None]
        res[~done] = inner[~done]
    return res


def dist_to_mesh(points, V, F, budget=4_000_000):
    """Kleinster Abstand jedes Punkts zu den Dreiecken (Brute Force in Bloecken)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(F) == 0 or len(pts) == 0:
        return np.zeros(len(pts))
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    m = len(F)
    step = max(1, budget // m)
    out = np.empty(len(pts))
    for s in range(0, len(pts), step):
        chunk = pts[s:s + step]
        k = len(chunk)
        p = np.repeat(chunk, m, axis=0)
        cp = _closest_points_on_triangles(p, np.tile(A, (k, 1)), np.tile(B, (k, 1)), np.tile(C, (k, 1)))
        d = np.linalg.norm(cp - p, axis=1).reshape(k, m)
        out[s:s + step] = d.min(axis=1)
    return out


def hausdorff_vertices(V1, F1, V2, F2, limit=3_000_000):
    """Groesster Abstand der Eckpunkte des einen Netzes zur Flaeche des anderen, in beide Richtungen. Bei grossen Netzen
    wird gleichmaessig eine Stichprobe der Eckpunkte genommen (Rueckgabe sagt es)."""
    u1, u2 = np.unique(F1), np.unique(F2)
    sampled = False
    res = 0.0

    def sample(idx, other_faces):
        nonlocal sampled
        k = max(50, limit // max(1, other_faces))
        if len(idx) > k:
            sampled = True
            idx = idx[np.linspace(0, len(idx) - 1, k).astype(np.int64)]
        return idx
    for (Va, ua), (Vb, Fb) in (((V1, u1), (V2, F2)), ((V2, u2), (V1, F1))):
        idx = sample(ua, len(Fb))
        if len(idx):
            res = max(res, float(dist_to_mesh(Va[idx], Vb, Fb).max()))
    return res, sampled


# ============================================================================= Dateiformate: Lesen
def _read_bytes(path):
    p = Path(path)
    if p.is_dir():
        fail("%s ist ein Ordner, keine Netzdatei." % path)
    if not p.exists():
        fail("%s gibt es nicht." % path)
    return p.read_bytes()


def read_stl(data):
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * n:
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
            return [("", rec["v"].astype(np.float64))]
    text = data.decode("utf-8", errors="replace")
    if not text.lstrip().lower().startswith("solid") or "facet" not in text.lower():
        fail("Keine gueltige STL-Datei (weder binaer noch ASCII).")
    vals = re.findall(r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text, re.IGNORECASE)
    if len(vals) % 3:
        fail("STL-Datei ist abgeschnitten (Eckpunkte nicht durch 3 teilbar).")
    try:
        arr = np.array(vals, dtype=np.float64).reshape(-1, 3, 3)
    except ValueError:
        fail("STL-Datei enthaelt unlesbare Zahlen.")
    return [("", arr)]


def _triangulate_polygon(Vs, ids):
    """Polygon (Eckpunktnummern `ids` in Vs) in Dreiecke. Konvex: Faecher; konkav: Ohrenschnitt in der Polygonebene."""
    k = len(ids)
    if k == 3:
        return [(ids[0], ids[1], ids[2])]
    X = np.array([Vs[i] for i in ids], dtype=np.float64)
    X = X - X.mean(axis=0)
    nrm = np.zeros(3)
    for a in range(k):
        b = (a + 1) % k
        nrm += np.cross(X[a], X[b])
    ln = float(np.linalg.norm(nrm))
    fan = [(ids[0], ids[q], ids[q + 1]) for q in range(1, k - 1)]
    if ln == 0 or not math.isfinite(ln):
        return fan
    n = nrm / ln
    ref = np.zeros(3)
    ref[int(np.argmin(np.abs(n)))] = 1.0
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    P2 = [(float(x @ u), float(x @ v)) for x in X]
    cr = [_cross(P2, a, (a + 1) % k, (a + 2) % k) for a in range(k)]
    if min(cr) >= 0:                                            # konvex: Faecher genuegt
        return fan
    try:
        tris, emerg = ear_clip(dict(enumerate(P2)), list(range(k)))
    except MeshcutError:
        return fan
    if emerg or len(tris) != k - 2:
        return fan
    return [(ids[a], ids[b], ids[c]) for a, b, c in tris]


def read_obj(data):
    V = []
    objs = [("", [])]
    for ln in data.decode("utf-8", errors="replace").splitlines():
        q = ln.split()
        if not q:
            continue
        tag = q[0]
        if tag == "v":
            try:
                V.append((float(q[1]), float(q[2]), float(q[3])))
            except (IndexError, ValueError):
                fail("OBJ: unlesbare Zeile '%s'" % ln.strip()[:40])
        elif tag == "o":
            objs.append((ln.strip()[1:].strip(), []))
        elif tag == "f":
            ids = []
            for tok in q[1:]:
                try:
                    k = int(tok.split("/")[0])
                except ValueError:
                    fail("OBJ: unlesbare Flaeche '%s'" % ln.strip()[:40])
                ids.append(k - 1 if k > 0 else len(V) + k)
            if len(ids) < 3:
                fail("OBJ: Flaeche mit weniger als 3 Ecken.")
            if min(ids) < 0 or max(ids) >= len(V):
                fail("OBJ: Flaeche verweist auf einen Eckpunkt, den es nicht gibt.")
            objs[-1][1].extend(_triangulate_polygon(V, ids))
    Va = np.array(V, dtype=np.float64).reshape(-1, 3)
    res = [(n, Va[np.array(f, dtype=np.int64).reshape(-1, 3)]) for n, f in objs if f]
    if not res:
        fail("OBJ ohne Flaechen.")
    return res


THREEMF_UNITS = {"micron": 1e-3, "millimeter": 1.0, "centimeter": 10.0, "inch": 25.4, "foot": 304.8, "meter": 1000.0}


def _xml_local(tag):
    return tag.rsplit("}", 1)[-1]


def read_3mf(data):
    import io
    import xml.etree.ElementTree as ET
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        names = [n for n in z.namelist() if n.lower().endswith(".model")]
        if not names:
            fail("3MF ohne Modelldatei.")
        root = ET.fromstring(z.read(next((n for n in names if n.lower() == "3d/3dmodel.model"), names[0])))
    except zipfile.BadZipFile:
        fail("Keine gueltige 3MF-Datei (kein ZIP).")
    unit = (root.get("unit") or "millimeter").lower()
    if unit not in THREEMF_UNITS:
        fail("3MF: Einheit '%s' unbekannt." % unit)
    scale = THREEMF_UNITS[unit]
    objects = {}
    items = []
    for el in root.iter():
        t = _xml_local(el.tag)
        if t == "object":
            oid = el.get("id")
            mesh = comps = None
            for ch in el:
                if _xml_local(ch.tag) == "mesh":
                    mesh = ch
                elif _xml_local(ch.tag) == "components":
                    comps = ch
            objects[oid] = (el.get("name") or "object_%s" % oid, mesh, comps)
        elif t == "item":
            items.append((el.get("objectid"), el.get("transform")))

    def mat(tr):
        if not tr:
            return None
        v = [float(x) for x in tr.split()]
        if len(v) != 12:
            fail("3MF: ungueltige Transformation.")
        M = np.eye(4)
        M[:3, :3] = np.array(v[:9]).reshape(3, 3)
        M[3, :3] = v[9:]
        return M

    def tris_of(oid, M, depth=0):
        if depth > 8 or oid not in objects:
            fail("3MF: Objekt %s fehlt." % oid)
        _, mesh, comps = objects[oid]
        out = []
        if mesh is not None:
            verts, tris = None, None
            for ch in mesh:
                if _xml_local(ch.tag) == "vertices":
                    verts = np.array([[float(v.get("x")), float(v.get("y")), float(v.get("z"))] for v in ch], dtype=np.float64)
                elif _xml_local(ch.tag) == "triangles":
                    tris = np.array([[int(v.get("v1")), int(v.get("v2")), int(v.get("v3"))] for v in ch], dtype=np.int64)
            if verts is None or tris is None or not len(tris):
                return out
            if tris.min() < 0 or tris.max() >= len(verts):
                fail("3MF: Dreieck verweist auf einen Eckpunkt, den es nicht gibt.")
            P = verts[tris]
            if M is not None:
                P = (np.concatenate([P.reshape(-1, 3), np.ones((P.size // 3, 1))], axis=1) @ M)[:, :3].reshape(-1, 3, 3)
            out.append(P)
        if comps is not None:
            for c in comps:
                Mc = mat(c.get("transform"))
                Mt = Mc if M is None else (M if Mc is None else Mc @ M)
                out.extend(tris_of(c.get("objectid"), Mt, depth + 1))
        return out
    res = []
    todo = items if items else [(k, None) for k in objects]
    for oid, tr in todo:
        arrs = tris_of(oid, mat(tr))
        if arrs:
            allp = np.concatenate(arrs)
            res.append((objects[oid][0], allp * scale if scale != 1.0 else allp))      # alles in Millimeter
    if not res:
        fail("3MF ohne Dreiecke.")
    return res


def read_glb(data):
    if len(data) < 20 or data[:4] != b"glTF":
        fail("Keine gueltige GLB-Datei.")
    pos, js, binary = 12, None, b""
    while pos + 8 <= len(data):
        ln, ty = struct.unpack("<II", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + ln]
        if ty == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif ty == 0x004E4942 and not binary:
            binary = chunk
        pos += 8 + ln + (-ln) % 4
    if js is None:
        fail("GLB ohne JSON-Teil.")
    CT = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
    NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def accessor(i):
        a = js["accessors"][i]
        if "sparse" in a:
            fail("GLB: sparse Accessoren werden nicht unterstuetzt.")
        bv = js["bufferViews"][a["bufferView"]]
        dt = np.dtype(CT[a["componentType"]])
        nc = NC[a["type"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride") or dt.itemsize * nc
        count = a["count"]
        raw = np.frombuffer(binary, dtype=np.uint8, count=(count - 1) * stride + dt.itemsize * nc, offset=off) if count else np.zeros(0, np.uint8)
        out = np.empty((count, nc), dtype=dt)
        for k in range(nc):
            out[:, k] = np.ndarray((count,), dtype=dt, buffer=raw, offset=k * dt.itemsize, strides=(stride,))
        return out

    def node_matrix(nd):
        if "matrix" in nd:
            return np.array(nd["matrix"], dtype=np.float64).reshape(4, 4).T
        T = np.eye(4)
        S = np.eye(4)
        S[0, 0], S[1, 1], S[2, 2] = nd.get("scale", [1, 1, 1])
        x, y, z, w = nd.get("rotation", [0, 0, 0, 1])
        R = np.eye(4)
        R[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        T[:3, 3] = nd.get("translation", [0, 0, 0])
        return T @ R @ S
    res = []

    def walk(ni, parent, depth=0):
        if depth > 32:
            fail("GLB: Knoten zu tief verschachtelt.")
        nd = js["nodes"][ni]
        M = parent @ node_matrix(nd)
        if "mesh" in nd:
            mesh = js["meshes"][nd["mesh"]]
            tris = []
            for pr in mesh["primitives"]:
                mode = pr.get("mode", 4)
                if mode in (5, 6):
                    fail("GLB: Dreiecksstreifen und -faecher (Modus %d) werden nicht unterstuetzt." % mode)
                if mode != 4:
                    continue                                    # Punkte und Linien haben keine Flaeche
                Vv = accessor(pr["attributes"]["POSITION"]).astype(np.float64)
                Ff = accessor(pr["indices"]).reshape(-1).astype(np.int64) if "indices" in pr else np.arange(len(Vv))
                if len(Ff) % 3:
                    fail("GLB: Indexliste nicht durch 3 teilbar.")
                Ff = Ff.reshape(-1, 3)
                if len(Ff) and (Ff.min() < 0 or Ff.max() >= len(Vv)):
                    fail("GLB: Dreieck verweist auf einen Eckpunkt, den es nicht gibt.")
                P = Vv[Ff]
                P = (np.concatenate([P.reshape(-1, 3), np.ones((P.size // 3, 1))], axis=1) @ M.T)[:, :3].reshape(-1, 3, 3)
                if np.linalg.det(M[:3, :3]) < 0:                # Spiegelung: glTF verlangt umgekehrten Umlaufsinn
                    P = P[:, [0, 2, 1]]
                tris.append(P)
            if tris:
                res.append((nd.get("name") or mesh.get("name") or "mesh_%d" % nd["mesh"], np.concatenate(tris)))
        for ch in nd.get("children", []):
            walk(ch, M, depth + 1)
    scene = js.get("scenes", [{}])[js.get("scene", 0)] if js.get("scenes") else None
    roots = scene.get("nodes", []) if scene else [i for i in range(len(js.get("nodes", [])))]
    for r in roots:
        walk(r, np.eye(4))
    if not res:
        fail("GLB ohne Dreiecke.")
    return res


def detect_format(path, data=None):
    ext = Path(path).suffix.lower().lstrip(".")
    if ext in FORMATS:
        return ext
    fail("Dateiendung '.%s' nicht unterstuetzt - moeglich: %s" % (ext, ", ".join("." + f for f in FORMATS)))


def read_meshes(path):
    """Alle Objekte einer Datei als Liste von Mesh (Koordinaten verschmolzen, Reihenfolge des ersten Auftretens)."""
    data = _read_bytes(path)
    fmt = detect_format(path)
    reader = {"stl": read_stl, "obj": read_obj, "3mf": read_3mf, "glb": read_glb}[fmt]
    try:
        objs = reader(data)
    except MeshcutError:
        raise
    except Exception as e:
        fail("%s ist keine lesbare %s-Datei (%s)." % (path, fmt.upper(), (str(e).strip().splitlines() or [type(e).__name__])[0]))
    out = []
    for name, P in objs:
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3, 3)
        if not np.isfinite(P).all():
            fail("%s enthaelt NaN oder unendliche Koordinaten." % path)
        V, F = weld_triangles(P)
        out.append(Mesh(V, F, name))
    return out


def load_mesh(path):
    """Datei -> ein Netz (mehrere Objekte werden zu einem vereint)."""
    ms = read_meshes(path)
    if len(ms) == 1:
        ms[0].name = Path(path).stem
        return ms[0]
    V, F = weld_triangles(np.concatenate([m.V[m.F] for m in ms]))
    return Mesh(V, F, Path(path).stem)


# ============================================================================= Dateiformate: Schreiben
def stl_bytes(V, F):
    n = len(F)
    P = (V[F]).astype(np.float32) if n else np.zeros((0, 3, 3), np.float32)
    nrm = np.zeros((n, 3), np.float32)
    if n:
        c = np.cross(P[:, 1].astype(np.float64) - P[:, 0], P[:, 2].astype(np.float64) - P[:, 0])
        ln = np.linalg.norm(c, axis=1)
        ok = ln > 0
        nrm[ok] = (c[ok] / ln[ok, None]).astype(np.float32)
    rec = np.zeros(n, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = nrm, P
    return b"meshcut".ljust(80, b"\0") + struct.pack("<I", n) + rec.tobytes()


def obj_bytes(V, F, name="part"):
    import io
    buf = io.StringIO()
    buf.write("# meshcut %s\no %s\n" % (__version__, name))
    np.savetxt(buf, V, fmt="v %.17g %.17g %.17g")
    np.savetxt(buf, F + 1, fmt="f %d %d %d")
    return buf.getvalue().encode("utf-8")


def threemf_bytes(objs):
    from xml.sax.saxutils import quoteattr
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>']
    for i, (name, V, F) in enumerate(objs, 1):
        lines.append('<object id="%d" type="model" name=%s><mesh><vertices>' % (i, quoteattr(name)))
        lines.extend('<vertex x="%.17g" y="%.17g" z="%.17g"/>' % tuple(v) for v in V)
        lines.append("</vertices><triangles>")
        lines.extend('<triangle v1="%d" v2="%d" v3="%d"/>' % tuple(f) for f in F.tolist())
        lines.append("</triangles></mesh></object>")
    lines.append("</resources><build>")
    lines.extend('<item objectid="%d"/>' % i for i in range(1, len(objs) + 1))
    lines.append("</build></model>")
    import io
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr("3D/3dmodel.model", "\n".join(lines).encode("utf-8"))
    return bio.getvalue()


def glb_bytes(objs):
    bin_parts, views, accs, meshes, nodes = [], [], [], [], []
    off = 0
    for name, V, F in objs:
        v32 = np.ascontiguousarray(V, dtype="<f4")
        i32 = np.ascontiguousarray(F.reshape(-1), dtype="<u4")
        for arr, tgt in ((v32, 34962), (i32, 34963)):
            raw = arr.tobytes()
            views.append({"buffer": 0, "byteOffset": off, "byteLength": len(raw), "target": tgt})
            pad = (-len(raw)) % 4
            bin_parts.append(raw + b"\0" * pad)
            off += len(raw) + pad
        accs.append({"bufferView": len(views) - 2, "componentType": 5126, "count": len(v32), "type": "VEC3",
                     "min": v32.min(axis=0).tolist() if len(v32) else [0, 0, 0], "max": v32.max(axis=0).tolist() if len(v32) else [0, 0, 0]})
        accs.append({"bufferView": len(views) - 1, "componentType": 5125, "count": len(i32), "type": "SCALAR"})
        meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": len(accs) - 2}, "indices": len(accs) - 1, "mode": 4}]})
        nodes.append({"name": name, "mesh": len(meshes) - 1})
    binary = b"".join(bin_parts)
    js = {"asset": {"version": "2.0", "generator": "meshcut " + __version__}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
          "nodes": nodes, "meshes": meshes, "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(binary)}]}
    jb = json.dumps(js, separators=(",", ":")).encode("utf-8")
    jb += b" " * ((-len(jb)) % 4)
    total = 12 + 8 + len(jb) + 8 + len(binary)
    return (b"glTF" + struct.pack("<II", 2, total) + struct.pack("<II", len(jb), 0x4E4F534A) + jb
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


def mesh_bytes(fmt, V, F, name="mesh"):
    if fmt == "stl":
        return stl_bytes(V, F)
    if fmt == "obj":
        return obj_bytes(V, F, name)
    if fmt == "3mf":
        return threemf_bytes([(name, V, F)])
    if fmt == "glb":
        return glb_bytes([(name, V, F)])
    fail("Format '%s' nicht unterstuetzt." % fmt)


# ============================================================================= Begleitdatei
def enc_array(a):
    a = np.ascontiguousarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape), "b64": base64.b64encode(zlib.compress(a.tobytes(), 6)).decode("ascii")}


MAX_ARRAY_BYTES = 1 << 28                                       # 256 MiB: mehr darf eine Begleitdatei nicht entpacken


def dec_array(d):
    try:
        dt = np.dtype(d["dtype"])
        shape = [int(x) for x in d["shape"]]
        if dt.str not in ("<i8", "<f8") or any(x < 0 for x in shape):
            raise ValueError("dtype")
        want = dt.itemsize * int(np.prod(shape, dtype=np.float64))
        if want > MAX_ARRAY_BYTES:
            raise ValueError("zu gross")
        zo = zlib.decompressobj()
        raw = zo.decompress(base64.b64decode(d["b64"]), want + 1)             # hoechstens so viel, wie die Form verspricht
        if len(raw) != want or zo.unconsumed_tail:
            raise ValueError("Laenge")
        return np.frombuffer(raw, dtype=dt).reshape(shape).copy()
    except Exception:
        fail("Die Begleitdatei %s ist beschaedigt." % SIDECAR)


def check_sidecar(side):
    if not isinstance(side, dict) or side.get("meshcut") != 1 or not isinstance(side.get("parts"), list) or not side["parts"]:
        raise ValueError("kein meshcut-Manifest")
    for k in ("source", "format", "mode", "split_ids", "split_coords"):
        if k not in side:
            raise ValueError("Feld %s fehlt" % k)
    if side["format"] not in FORMATS:
        raise ValueError("Format")
    names = []
    ids = []
    for p in side["parts"]:
        if not (isinstance(p, dict) and isinstance(p.get("id"), int) and isinstance(p.get("file"), str)
                and isinstance(p.get("faces"), int) and isinstance(p.get("cap_faces"), int) and "src" in p
                and isinstance(p.get("hash"), str)):
            raise ValueError("Teileintrag")
        ids.append(p["id"])
        names.append(p["file"])
    fmt = side["format"]
    for n in names:
        if fmt in MULTI_NAME:
            if n != MULTI_NAME[fmt]:
                raise ValueError("Dateiname")
        elif not (PART_NAME.fullmatch(n) and n.endswith("." + fmt)):
            raise ValueError("Dateiname")
    if len(set(ids)) != len(ids):
        raise ValueError("doppelte Teilnummern")
    src = side["source"]
    if not (isinstance(src, dict) and isinstance(src.get("faces"), int) and src.get("precision") in ("float32", "float64") and isinstance(src.get("hash"), str)):
        raise ValueError("source")
    return sorted(set(names))


def load_sidecar(folder):
    if Path(folder).is_dir() and not link_like(folder):
        recover_outdir(folder)
    p = Path(folder) / SIDECAR
    if not p.is_file():
        return None
    try:
        side = json.loads(p.read_text(encoding="utf-8"))
        check_sidecar(side)
    except (ValueError, OSError, UnicodeDecodeError) as e:
        fail("%s ist keine gueltige meshcut-Begleitdatei (%s)." % (p, e))
    return side


def make_sidecar(res, fmt, stats, files):
    precision = "float32" if fmt in FLOAT32_FORMATS else "float64"
    src = res.source
    q = quantize(src.V, precision)
    return {"meshcut": 1, "version": __version__, "mode": res.mode, "format": fmt,
            "source": {"name": src.name, "faces": int(len(src.F)), "vertices": int(len(src.V)), "precision": precision,
                       "hash": triangle_hash(src.V, src.F, precision), "quantized": bool((q != src.V).any()),
                       "volume": stats["volume"], "area": stats["area"], "diag": stats["diag"]},
            "planes": [p.to_json() for p in res.planes],
            "split_ids": enc_array(res.split_ids.astype(np.int64)), "split_coords": enc_array(res.split_coords.astype(np.float64)),
            "parts": [{"id": p.id, "file": files[i][0], "object": files[i][1], "faces": int(len(p.F)), "cap_faces": p.n_cap,
                       "hash": triangle_hash(p.V, p.F, precision), "src": enc_array(p.src.astype(np.int64))} for i, p in enumerate(res.parts)]}


# ============================================================================= Sicher schreiben
def link_like(path):
    st = os.lstat(path)
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & 0x400)


def rmtree_force(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            try:
                os.chmod(os.path.join(root, f), stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
    shutil.rmtree(folder)


JOURNAL = "journal.json"


def _write_journal(aside, state, names, moved):
    tmp = aside / (JOURNAL + ".tmp")
    tmp.write_text(json.dumps({"state": state, "names": names, "moved": moved}), encoding="utf-8")
    os.replace(tmp, aside / JOURNAL)


def recover_outdir(out):
    """Raeumt nach einem harten Abbruch (Stromausfall, kill -9) mitten im Ersetzen auf: war das Ersetzen noch nicht fertig,
    kommt der ganze alte Satz zurueck; war es fertig, wird nur der Rest weggeraeumt."""
    out = Path(out)
    try:
        leftovers = [d for d in out.iterdir() if d.name.startswith(".meshcut_") and d.is_dir()]
    except OSError:
        return
    entries = [d for d in leftovers if (d / JOURNAL).is_file()]
    for d in leftovers:
        if d not in entries:                                         # Zwischenordner eines abgebrochenen Laufs, noch nichts installiert
            rmtree_force(d)
    for aside in entries:
        try:
            j = json.loads((aside / JOURNAL).read_text(encoding="utf-8"))
            names, moved = j["names"], {int(k): v for k, v in j["moved"].items()}
            if j["state"] != "done":
                fresh = [n for n in names if n not in moved.values()]            # Namen ohne Vorgaenger: nur neu installiert
                for n in fresh:
                    if (out / n).is_file():
                        (out / n).unlink()
                for i, n in moved.items():
                    if (aside / str(i)).exists():                                # Original liegt zur Seite -> dort ist nur Neues
                        if (out / n).is_file():
                            (out / n).unlink()
                        os.replace(aside / str(i), out / n)
            rmtree_force(aside)
            print("Hinweis: ein frueherer, abgebrochener Lauf in %s wurde %s." % (out, "abgeschlossen" if j["state"] == "done" else "zurueckgerollt (der alte Satz ist wieder da)"), file=sys.stderr)
        except (OSError, ValueError, KeyError, TypeError) as e:
            print("Hinweis: Aufraeumen von %s nicht moeglich (%s)." % (aside, e), file=sys.stderr)


def publish_files(staging, names, out):
    """Fertige Dateien aus `staging` nach `out` bringen. Bereits vorhandene Ziele wandern erst zur Seite und kommen bei jedem
    Fehler zurueck - entweder liegt der ganze neue Satz da oder der ganze alte. Ein Journal im Seitenordner macht das auch nach
    hartem Abbruch rueckgaengig (recover_outdir)."""
    out = Path(out)
    aside = Path(tempfile.mkdtemp(prefix=".meshcut_", dir=out))
    installed = []
    moved = {i: n for i, n in enumerate(names) if (out / n).exists()}
    try:
        _write_journal(aside, "moving", names, moved)
        for i, n in moved.items():
            os.replace(out / n, aside / str(i))
        for n in names:
            installed.append(n)
            os.replace(Path(staging) / n, out / n)
        _write_journal(aside, "done", names, moved)
    except BaseException:
        stuck = []
        for n in installed:
            try:
                (out / n).unlink(missing_ok=True)
            except OSError as e:
                stuck.append("%s (%s)" % (n, e))
        for f in list(aside.iterdir()):
            if not f.name.isdigit():
                continue
            try:
                os.replace(f, out / names[int(f.name)])
            except OSError as e:
                stuck.append("%s (%s)" % (names[int(f.name)], e))
        if stuck:
            print("Fehler: Nicht alles liess sich zurueckholen - die alten Dateien liegen in %s\n  Problem bei: %s" % (aside, "; ".join(stuck)), file=sys.stderr)
        else:
            rmtree_force(aside)
        raise
    rmtree_force(aside)


def prepare_outdir(out, new_names, inputs):
    out = Path(out)
    if out.exists() and (link_like(out) or not out.is_dir()):
        fail("%s ist ein Link oder kein Ordner - meshcut arbeitet darin nicht." % out)
    if not out.exists():
        if not out.resolve().parent.is_dir():
            fail("Der Ordner fuer %s existiert nicht." % out)
        out.mkdir()
    recover_outdir(out)
    owned = set()
    old = out / SIDECAR
    if old.exists():
        try:
            owned = set(check_sidecar(json.loads(old.read_text(encoding="utf-8")))) | {SIDECAR}
        except (ValueError, OSError, UnicodeDecodeError):
            fail("%s ist keine meshcut-Begleitdatei - bitte einen leeren Ordner nehmen." % old)
    for n in new_names:
        if (out / n).exists() and n not in owned:
            fail("%s liegt schon in %s und stammt nicht von meshcut - bitte einen leeren Ordner nehmen." % (n, out))
        if (out / n).exists() and (link_like(out / n) or not (out / n).is_file()):
            fail("%s ist ein Link oder Ordner." % (out / n))
    for x in inputs:
        rx = Path(x).resolve()
        if rx.parent == out.resolve() and (rx.name in owned or rx.name in new_names):
            fail("%s ist eine der Dateien, die split in %s schreibt - bitte eine andere Eingabe waehlen." % (x, out))
    return owned


def write_split(res, outdir, fmt, inputs=()):
    """Teile + meshcut.json transaktional schreiben. Rueckgabe: Begleitdaten (dict)."""
    ids = [p.id for p in res.parts]
    if fmt in MULTI_NAME:
        files = [(MULTI_NAME[fmt], i) for i in range(len(res.parts))]
        names = [MULTI_NAME[fmt]]
    else:
        files = [("part_%d.%s" % (pid, fmt), None) for pid in ids]
        names = [f for f, _ in files]
    owned = prepare_outdir(outdir, names + [SIDECAR], inputs)
    stats = mesh_stats(res.source.V, res.source.F)
    side = make_sidecar(res, fmt, stats, files)
    out = Path(outdir)
    staging = Path(tempfile.mkdtemp(prefix=".meshcut_", dir=out))
    try:
        if fmt in MULTI_NAME:
            objs = [("part_%d" % p.id, p.V, p.F) for p in res.parts]
            (staging / names[0]).write_bytes(threemf_bytes(objs) if fmt == "3mf" else glb_bytes(objs))
        else:
            for p, (fn, _) in zip(res.parts, files):
                (staging / fn).write_bytes(mesh_bytes(fmt, p.V, p.F, "part_%d" % p.id))
        (staging / SIDECAR).write_text(json.dumps(side, indent=1), encoding="utf-8")
        allnames = names + [SIDECAR]
        # was eine fruehere Ausgabe hinterlassen hat und jetzt nicht mehr dazugehoert, wird mit zur Seite gelegt
        stale = sorted(n for n in owned if n not in allnames and (out / n).is_file())
        publish_files(staging, allnames, out)
        for n in stale:
            try:
                os.chmod(out / n, stat.S_IWRITE | stat.S_IREAD)      # ein schreibgeschuetztes Teil aus frueherem Lauf darf nicht bleiben
                (out / n).unlink()
            except OSError as e:
                print("Hinweis: %s aus einem frueheren Lauf liess sich nicht entfernen (%s)" % (n, e), file=sys.stderr)
    finally:
        rmtree_force(staging)
    return side


def save_atomic(data, path):
    path = Path(path)
    if path.exists() and (link_like(path) or not path.is_file()):
        fail("%s ist ein Ordner oder Link." % path)
    if path.exists() and not os.access(path, os.W_OK):
        fail("%s ist schreibgeschuetzt." % path)
    if not path.resolve().parent.is_dir():
        fail("Der Ordner fuer %s existiert nicht." % path)
    d = Path(tempfile.mkdtemp(prefix=".meshcut_", dir=path.resolve().parent))
    try:
        (d / path.name).write_bytes(data)
        os.replace(d / path.name, path.resolve())
    finally:
        rmtree_force(d)


# ============================================================================= Teile laden / verify
def load_parts(folder):
    """Ordner mit meshcut.json -> (Begleitdaten, Liste (id, V, F) wie in den Dateien)."""
    folder = Path(folder)
    side = load_sidecar(folder)
    if side is None:
        fail("%s enthaelt keine %s." % (folder, SIDECAR))
    fmt = side["format"]
    cache = {}
    out = []
    for meta in side["parts"]:
        fn = folder / meta["file"]
        if not fn.is_file() or link_like(fn):
            fail("Teil %d fehlt (%s)." % (meta["id"], fn))
        if fn not in cache:
            data = fn.read_bytes()
            cache[fn] = read_meshes_bytes(fmt, data, fn)
        ms = cache[fn]
        idx = meta["object"] or 0
        if idx >= len(ms):
            fail("In %s fehlt das Objekt fuer Teil %d." % (fn, meta["id"]))
        out.append((meta["id"], ms[idx].V, ms[idx].F))
    return side, out


def read_meshes_bytes(fmt, data, label):
    reader = {"stl": read_stl, "obj": read_obj, "3mf": read_3mf, "glb": read_glb}[fmt]
    try:
        objs = reader(data)
    except MeshcutError:
        raise
    except Exception as e:
        fail("%s ist keine lesbare %s-Datei (%s)." % (label, fmt.upper(), (str(e).strip().splitlines() or [type(e).__name__])[0]))
    res = []
    for name, P in objs:
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3, 3)
        if not np.isfinite(P).all():
            fail("%s enthaelt NaN oder unendliche Koordinaten." % label)
        # die Dateien eines Teils sind bereits verschmolzen; Reihenfolge der Dreiecke muss erhalten bleiben
        V, F = weld_triangles(P)
        res.append(Mesh(V, F, name))
    return res


def tolerances(precision):
    return (1e-4, 1e-4) if precision == "float32" else (1e-9, 1e-9)


def _check_cap_planes(side, parts, prec):
    """Jedes Deckel-Dreieck muss mit allen drei Ecken auf einer der gespeicherten Schnittebenen liegen."""
    try:
        planes = [(np.array(p["normal"], dtype=np.float64), float(p["offset"])) for p in side["planes"]]
        if not planes or any(n.shape != (3,) or not np.isfinite(n).all() or not math.isfinite(d) for n, d in planes):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return False, "die Schnittebenen in %s sind unbrauchbar" % SIDECAR
    allv = np.concatenate([V for _, V, _ in parts])
    extent = float((allv.max(axis=0) - allv.min(axis=0)).max()) if len(allv) else 0.0
    mag = float(np.abs(allv).max()) if len(allv) else 0.0
    eps = np.finfo(np.float32 if prec == "float32" else np.float64).eps
    tol = (1e-9 if prec == "float64" else 1e-6) * extent + 16 * eps * (mag + max(abs(d) for _, d in planes))
    worst = 0.0
    n_bad = 0
    for meta, (_, V, F) in zip(side["parts"], parts):
        nonc = len(F) - meta["cap_faces"]
        if nonc == len(F):
            continue
        T = V[F[nonc:]]                                                    # (k, 3, 3)
        ok_any = np.zeros(len(T), bool)
        best = np.full(len(T), np.inf)
        for n, d in planes:
            dist = np.abs(T @ n - d).max(axis=1)
            ok_any |= dist <= tol
            best = np.minimum(best, dist)
        n_bad += int((~ok_any).sum())
        if len(best):
            worst = max(worst, float(best.max()))
    if n_bad:
        return False, "%d Deckel-Dreiecke liegen nicht auf einer Schnittebene (groesster Abstand %.3g, erlaubt %.3g)" % (n_bad, worst, tol)
    return True, "alle Deckel-Dreiecke liegen auf den Schnittebenen (groesster Abstand %.3g, erlaubt %.3g)" % (worst, tol)


def verify_folder(folder, source=None):
    """Rundlauf pruefen. Rueckgabe: dict mit `ok` und Einzelpruefungen (Liste von (name, ok, text))."""
    side, parts = load_parts(folder)
    prec = side["source"]["precision"]
    checks = []

    def add(name, ok, text):
        checks.append((name, bool(ok), text))
    rel_v, rel_a = tolerances(prec)
    # 0. die Teile selbst: jede Datei muss genau das enthalten, was split geschrieben hat (Fragmente und Deckel eingeschlossen)
    bad = []
    for meta, (pid, V, F) in zip(side["parts"], parts):
        if triangle_hash(V, F, prec) != meta["hash"]:
            bad.append(pid)
    add("teile", not bad, "alle %d Teile unveraendert (Hash je Teil)" % len(parts) if not bad
        else "Teil %s weicht von dem ab, was split geschrieben hat - die Datei wurde veraendert" % ", ".join(str(b) for b in bad))
    # 0b. die Deckel muessen auf den Schnittebenen liegen
    if side["mode"] != "labels" and any(m["cap_faces"] for m in side["parts"]):
        add("ebenen", *_check_cap_planes(side, parts, prec))
    # 1. Rundlauf ueber die Begleitdatei
    joined = join_with_sidecar(parts, side)
    jh = triangle_hash(joined.V, joined.F, prec)
    add("hash", jh == side["source"]["hash"], "zusammengesetzt %s... %s Hash der Quelle (%s)" % (jh[:12], "=" if jh == side["source"]["hash"] else "!=", prec))
    # 2. Teile einzeln
    vol_sum = area_sum = 0.0
    all_tight = True
    all_even = True
    notes = []
    for meta, (pid, V, F) in zip(side["parts"], parts):
        nonc = len(F) - meta["cap_faces"]
        v, a = volume_area(V, F)
        _, a_nc = volume_area(V, F[:nonc])
        vol_sum += v
        area_sum += a_nc
        st = mesh_stats(V, F, prec)
        if side["mode"] != "labels":
            if not st["watertight"]:
                all_tight = False
                notes.append("Teil %d nicht wasserdicht (%d offen, %d falsch)" % (pid, st["open_edges"], st["bad_edges"]))
            if st["euler"] % 2:
                all_even = False
                notes.append("Teil %d: Euler-Zahl %d ungerade" % (pid, st["euler"]))
    sv, sa, diag = side["source"]["volume"], side["source"]["area"], side["source"]["diag"]
    if side["mode"] == "labels":
        add("flaeche", abs(area_sum - sa) <= rel_a * max(sa, 1e-300), "Flaeche der Teile %.9g, Original %.9g" % (area_sum, sa))
    else:
        has_cap = any(m["cap_faces"] for m in side["parts"])
        add("wasserdicht", all_tight or not has_cap, "jedes Teil wasserdicht" if all_tight else "; ".join(notes))
        add("euler", all_even or not has_cap, "Euler-Zahl je Teil gerade" if all_even else "; ".join(notes))
        if has_cap:
            add("volumen", abs(vol_sum - sv) <= rel_v * max(abs(sv), diag * sa * 1e-3, 1e-300),
                "Summe der Teile %.12g, Original %.12g" % (vol_sum, sv))
        add("flaeche", abs(area_sum - sa) <= rel_a * max(sa, 1e-300), "Flaeche ohne Deckel %.9g, Original %.9g" % (area_sum, sa))
    # 3. naeherungsweises Zusammensetzen ohne Begleitdatei + Hausdorff gegen die Quelle
    if source is not None:
        src_mesh = load_mesh(source)
        sh = triangle_hash(src_mesh.V, src_mesh.F, prec)
        add("quelle", sh == side["source"]["hash"], "Quelldatei %s... %s Hash in der Begleitdatei" % (sh[:12], "=" if sh == side["source"]["hash"] else "!="))
        naive = join_plain([Mesh(V, F) for _, V, F in parts]) if side["mode"] != "labels" else joined
        hd, sampled = hausdorff_vertices(src_mesh.V, src_mesh.F, naive.V, naive.F)
        mag = float(np.abs(src_mesh.V).max()) if len(src_mesh.V) else 0.0
        tol = (1e-9 if prec == "float64" else 1e-6) * max(diag, 1e-300) + (8 * np.finfo(np.float32).eps * mag if prec == "float32" else 0.0)
        add("hausdorff", hd <= tol, "Hausdorff (Eckpunkte, ohne Begleitdatei zusammengesetzt) %.3g, erlaubt %.3g%s" % (hd, tol, " (Stichprobe)" if sampled else ""))
    return {"ok": all(c[1] for c in checks), "checks": checks, "source_hash": side["source"]["hash"], "joined_hash": jh}


# ============================================================================= Befehle
def say(*a):
    print(*a)


def read_label_file(path):
    p = Path(path)
    if not p.is_file():
        fail("%s gibt es nicht." % path)
    ext = p.suffix.lower()
    try:
        if ext == ".npy":
            arr = np.load(p, allow_pickle=False)
        elif ext == ".json":
            arr = np.array(json.loads(p.read_text(encoding="utf-8")))
        elif ext in (".csv", ".txt"):
            arr = np.array([float(x) for x in re.split(r"[,\s;]+", p.read_text(encoding="utf-8").strip()) if x])
        else:
            fail("Zuordnungskarte: .npy, .json oder .csv noetig, nicht '%s'." % ext)
    except MeshcutError:
        raise
    except Exception as e:
        fail("Zuordnungskarte unlesbar (%s)." % (str(e).splitlines() or [type(e).__name__])[0])
    return arr.reshape(-1)


def centroid_labels(mesh, counts):
    c = mesh.V[mesh.F].mean(axis=1)
    lo, hi = mesh.V.min(axis=0), mesh.V.max(axis=0)
    ext = np.where(hi > lo, hi - lo, 1.0)
    cell = np.minimum(((c - lo) / ext * np.array(counts)).astype(np.int64), np.array(counts) - 1)
    return cell[:, 0] + counts[0] * (cell[:, 1] + counts[1] * cell[:, 2])


def cmd_split(a):
    mesh = load_mesh(a.input)
    if a.flip:
        mesh = Mesh(mesh.V, mesh.F[:, [0, 2, 1]], mesh.name)
    modes = [bool(a.plane), bool(a.grid), bool(a.max), bool(a.wedges), bool(a.labels)]
    if sum(modes) != 1:
        fail("Genau eine Art zu zerlegen angeben: --plane, --grid, --max, --wedges oder --labels.")
    lo, hi = mesh.V.min(axis=0), mesh.V.max(axis=0)
    cap = not a.no_cap
    if a.labels:
        res = split_labels(mesh, read_label_file(a.labels))
        if a.no_cap:
            say("Hinweis: --no-cap hat im Zuordnungsmodus keine Wirkung (es gibt keine Schnittflaechen).")
    else:
        table = None
        if a.plane:
            if len(a.plane) > MAX_PLANES:
                fail("Hoechstens %d Ebenen." % MAX_PLANES)
            planes = []
            for k, spec in enumerate(a.plane):
                p = parse_plane(spec)
                p.weight = 1 << k
                planes.append(p)
        elif a.grid:
            counts = parse_counts(a.grid, "--grid")
            check_cell_count(counts)
            planes = grid_planes(lo, hi, counts)
            if not planes:
                fail("--grid %s ergibt nur eine Zelle." % a.grid)
        elif a.max:
            sizes = parse_sizes(a.max)
            ext = hi - lo
            counts = [max(1, math.ceil(ext[i] / sizes[i])) if math.isfinite(sizes[i]) else 1 for i in range(3)]
            check_cell_count(counts)
            planes = grid_planes(lo, hi, counts)
            if not planes:
                fail("Das Netz passt schon in %s - nichts zu schneiden." % a.max)
            say("Raster %dx%dx%d" % tuple(counts))
        else:
            ax = AXES[a.axis]
            if a.center == "auto":
                cen = (lo + hi) / 2
            else:
                try:
                    cc = [float(x) for x in a.center.split(",")]
                except ValueError:
                    cc = []
                if len(cc) != 2:
                    fail("--center: zwei Zahlen noetig, z. B. 10,5 (die beiden Koordinaten quer zur Achse)")
                cen = (lo + hi) / 2
                u, v = (ax + 1) % 3, (ax + 2) % 3
                cen[u], cen[v] = cc
            planes, table = wedge_planes(cen, ax, a.wedges)
        if not (math.isfinite(a.snap) and 0 <= a.snap < 1e-3):
            fail("--snap: Zahl zwischen 0 und 0.001 noetig.")
        res = split_planes(mesh, planes, cap=cap, table=table, mode="wedges" if a.wedges else ("grid" if (a.grid or a.max) else "plane"), snap=a.snap)
        for p in res.parts:
            er = edge_report(p.F, len(p.V))
            if cap and not er["watertight"]:
                say("Hinweis: Teil %d ist nicht wasserdicht (%d offene, %d falsch orientierte Kanten) - die Ebene beruehrt das Netz vermutlich an einer Kante oder Flaeche." % (p.id, er["open"], er["bad"]))
        if res.report["cap_emergencies"]:
            say("Hinweis: %d Deckel-Dreiecke mussten im Notverfahren gebildet werden - bitte mit `verify` pruefen." % res.report["cap_emergencies"])
    side = write_split(res, a.outdir, a.format, inputs=[a.input])
    say("%d Teile + %s in %s geschrieben (%s)." % (len(res.parts), SIDECAR, a.outdir, a.format))
    if a.verify:
        r = verify_folder(a.outdir, source=a.input)
        report_verify(r)
        if not r["ok"]:
            fail("Pruefung fehlgeschlagen: %s." % ", ".join(c[0] for c in r["checks"] if not c[1]))


def report_verify(r):
    for name, ok, text in r["checks"]:
        say("%s %-11s %s" % ("OK    " if ok else "FEHLER", name, text))


def cmd_join(a):
    inputs = [Path(x) for x in a.parts]
    outp = Path(a.output)
    fmt = detect_format(outp)
    if len(inputs) == 1 and inputs[0].is_dir():
        side, parts = load_parts(inputs[0])
        guarded = set(check_sidecar(side)) | {SIDECAR}
        if outp.exists() and any((inputs[0] / n).exists() and os.path.samefile(inputs[0] / n, outp) for n in guarded):
            fail("%s gehoert zu den Teilen in %s - bitte ein anderes Ziel waehlen." % (outp, inputs[0]))
        mesh = join_with_sidecar(parts, side)
        prec = side["source"]["precision"]
        h = triangle_hash(mesh.V, mesh.F, prec)
        say("Mit %s zusammengesetzt: %d Dreiecke, Hash %s %s Quelle." % (SIDECAR, len(mesh.F), h[:12], "=" if h == side["source"]["hash"] else "!="))
        if h != side["source"]["hash"]:
            fail("Der Hash weicht von der Quelle ab - die Teile wurden veraendert.")
    else:
        meshes = []
        for p in inputs:
            if p.is_dir():
                fail("%s ist ein Ordner - entweder genau einen Ordner oder nur Dateien angeben." % p)
            meshes.extend(read_meshes(p))
        if a.require_sidecar:
            fail("--require-sidecar: ohne %s-Ordner nicht moeglich." % SIDECAR)
        mesh = join_plain(meshes)
        say("Ohne %s zusammengesetzt (Naeherung: Deckelpaare entfernt, Dreiecke bleiben fein): %d Dreiecke." % (SIDECAR, len(mesh.F)))
    refuse = [x.resolve() for x in inputs if x.is_file()]
    if outp.resolve() in refuse or (outp.exists() and any(x.is_file() and os.path.samefile(x, outp) for x in inputs)):
        fail("%s ist auch Eingabe - bitte ein anderes Ziel waehlen." % outp)
    save_atomic(mesh_bytes(fmt, mesh.V, mesh.F, outp.stem), outp)
    say("%s geschrieben." % outp)


def cmd_verify(a):
    r = verify_folder(a.parts, source=a.source)
    report_verify(r)
    if not r["ok"]:
        fail("Pruefung fehlgeschlagen: %s." % ", ".join(c[0] for c in r["checks"] if not c[1]))
    say("Alles in Ordnung.")


def cmd_info(a):
    mesh = load_mesh(a.input)
    st = mesh_stats(mesh.V, mesh.F, "float32" if detect_format(a.input) in FLOAT32_FORMATS else "float64")
    say("Datei:        %s" % a.input)
    say("Eckpunkte:    %d   Dreiecke: %d   Kanten: %d" % (st["vertices"], st["faces"], st["edges"]))
    say("Begrenzung:   %s .. %s (Diagonale %.6g)" % (np.round(st["bbox_min"], 6).tolist(), np.round(st["bbox_max"], 6).tolist(), st["diag"]))
    say("Volumen:      %.9g%s   Flaeche: %.9g" % (st["volume"], " (negativ: nach innen orientiert)" if st["volume"] < 0 else "", st["area"]))
    say("Wasserdicht:  %s (%d offene, %d falsch orientierte/mehrfache Kanten)" % ("ja" if st["watertight"] else "nein", st["open_edges"], st["bad_edges"]))
    say("Euler-Zahl:   %d   Huellen: %d" % (st["euler"], st["shells"]))
    say("Hash:         %s" % st["hash"])


def cmd_labels(a):
    mesh = load_mesh(a.input)
    counts = parse_counts(a.grid, "--grid")
    lab = centroid_labels(mesh, counts)
    out = Path(a.output)
    if out.suffix.lower() == ".npy":
        import io
        bio = io.BytesIO()
        np.save(bio, lab.astype(np.int32))
        save_atomic(bio.getvalue(), out)
    elif out.suffix.lower() in (".csv", ".txt"):
        save_atomic(("\n".join(str(int(x)) for x in lab) + "\n").encode(), out)
    elif out.suffix.lower() == ".json":
        save_atomic(json.dumps(lab.tolist()).encode(), out)
    else:
        fail("Zuordnungskarte: .npy, .json oder .csv als Ziel noetig.")
    say("Zuordnungskarte mit %d Werten und %d Teilen geschrieben: %s" % (len(lab), len(np.unique(lab)), out))


# ============================================================================= Kommandozeile
class _ApiParser(argparse.ArgumentParser):
    def error(self, message):
        raise MeshcutError(message)


def build_parser(parser_class=argparse.ArgumentParser):
    p = parser_class(prog="meshcut", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version="meshcut " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split", help="Netz zerlegen")
    s.add_argument("input"); s.add_argument("outdir")
    s.add_argument("--plane", action="append", help="Schnittebene, mehrfach moeglich: x=12.5 | z=-3 | a,b,c,d")
    s.add_argument("--grid", help="Gitter, Zellen in x mal y mal z, z. B. 2x2x3")
    s.add_argument("--max", help="Druckraum in x mal y mal z, z. B. 220x220x250: gleichmaessig so viele Schnitte, dass jedes Teil hineinpasst")
    s.add_argument("--wedges", type=int, help="Keile um eine Achse (gerade Zahl, z. B. 8)")
    s.add_argument("--axis", choices=("x", "y", "z"), default="z", help="Achse der Keile")
    s.add_argument("--center", default="auto", help="Lage der Achse: zwei Zahlen quer zur Achse oder 'auto' (Mitte des Netzes)")
    s.add_argument("--labels", help="Zuordnungskarte (.npy/.json/.csv): ein Teilnummer pro Dreieck, Geometrie bleibt unveraendert")
    s.add_argument("--format", choices=FORMATS, default="stl", help="Dateiformat der Teile (3mf und glb: alle Teile in einer Datei)")
    s.add_argument("--no-cap", action="store_true", help="ohne Deckel schneiden (fuer offene Netze)")
    s.add_argument("--snap", type=float, default=1e-12, help="Eckpunkte naeher als dieser Anteil der Koordinatengroesse an der Ebene gelten als auf der Ebene (Rundungsrauschen); 0 = exakter Vorzeichentest")
    s.add_argument("--flip", action="store_true", help="Umlaufsinn der Dreiecke umdrehen (Netz ist nach innen orientiert)")
    s.add_argument("--verify", action="store_true", help="danach die gespeicherten Dateien einlesen und den Rundlauf pruefen")
    s.set_defaults(fn=cmd_split)

    s = sub.add_parser("join", help="Teile zusammensetzen")
    s.add_argument("parts", nargs="+", help="Ordner mit meshcut.json (bitgenau) oder einzelne Dateien (Naeherung)")
    s.add_argument("output")
    s.add_argument("--require-sidecar", action="store_true", help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_join)

    s = sub.add_parser("verify", help="Rundlauf eines Ordners pruefen")
    s.add_argument("parts")
    s.add_argument("--source", help="Originaldatei fuer den Vergleich (Hash, Hausdorff)")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("info", help="Kennzahlen eines Netzes")
    s.add_argument("input")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("labels", help="Zuordnungskarte fuer ein Gitter schreiben")
    s.add_argument("input"); s.add_argument("output")
    s.add_argument("--grid", required=True, help="Zellen in x mal y mal z, z. B. 2x2x2")
    s.set_defaults(fn=cmd_labels)
    return p


def oserror_message(e):
    name = getattr(e, "filename", None) or "einer Datei"
    if isinstance(e, FileNotFoundError):
        return "%s gibt es nicht" % name
    if isinstance(e, PermissionError):
        return "%s ist schreibgeschuetzt oder in Benutzung" % name
    if isinstance(e, (FileExistsError, NotADirectoryError, IsADirectoryError)):
        return "%s: Datei und Ordner verwechselt" % name
    return None


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    try:
        a.fn(a)
    except KeyboardInterrupt:
        raise
    except MeshcutError as e:
        sys.exit("Fehler: %s" % e)
    except SystemExit:
        raise
    except BaseException as e:
        if os.environ.get("MESHCUT_DEBUG"):
            raise
        sys.exit("Fehler: %s (MESHCUT_DEBUG=1 zeigt den ganzen Ablauf)" % (oserror_message(e) if isinstance(e, OSError) and oserror_message(e)
                                                                       else (str(e).strip().splitlines() or [type(e).__name__])[0]))


# ============================================================================= Python-Schnittstelle
def _flags(opts):
    argv = []
    for key, val in opts.items():
        if key in ("help", "h", "version"):
            raise MeshcutError("Option '%s' gibt es in der Python-Schnittstelle nicht." % key)
        flag = "--" + key.replace("_", "-")
        if val is None or val is False:
            continue
        if val is True:
            argv.append(flag)
        elif isinstance(val, (list, tuple)) and key == "plane":
            if len(val) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in val):
                argv.append("%s=%s" % (flag, ",".join(repr(float(v)) for v in val)))        # eine Ebene a,b,c,d
            else:
                argv.extend("%s=%s" % (flag, v) for v in val)
        else:
            if isinstance(val, (list, tuple)):
                val = ",".join(str(v) for v in val)
            argv.append("%s=%s" % (flag, val))
    return argv


def _run(cmd, positional, opts):
    import contextlib
    import io
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            a = build_parser(_ApiParser).parse_args([cmd] + [str(x) for x in positional] + _flags(opts))
            a.fn(a)
    except SystemExit as e:
        if e.code in (None, 0):
            return out.getvalue()
        raise MeshcutError(str(e.code).removeprefix("Fehler: ")) from None
    except (KeyboardInterrupt, MeshcutError):
        raise
    except BaseException as e:
        if isinstance(e, OSError) and oserror_message(e):
            raise MeshcutError(oserror_message(e)) from e
        raise MeshcutError((str(e).strip().splitlines() or [type(e).__name__])[0]) from e
    return out.getvalue()


def split(input, outdir, **options):
    """Netz zerlegen (Optionen wie auf der Kommandozeile: plane, grid, max, wedges, axis, center, labels, format, no_cap, flip,
    verify). Rueckgabe: Inhalt von meshcut.json als dict."""
    _run("split", [input, outdir], options)
    try:
        return json.loads((Path(outdir) / SIDECAR).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise MeshcutError("%s liess sich nach dem Schreiben nicht lesen (%s)" % (SIDECAR, e)) from e


def join(parts, output):
    """Teile zusammensetzen: Ordner mit meshcut.json (bitgenau) oder Liste von Dateien (Naeherung). Rueckgabe: Zielpfad."""
    items = list(parts) if isinstance(parts, (list, tuple)) else [parts]
    _run("join", items + [output], {})
    return Path(output)


def verify(parts, source=None):
    """Rundlauf pruefen. Rueckgabe: {'ok': bool, 'checks': [(name, ok, text)], 'source_hash', 'joined_hash'}; wirft nicht bei Abweichung."""
    import contextlib
    import io
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return verify_folder(parts, source)
    except MeshcutError:
        raise
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        raise MeshcutError((str(e).strip().splitlines() or [type(e).__name__])[0]) from e


def info(path):
    """Kennzahlen eines Netzes als dict."""
    try:
        m = load_mesh(path)
        return mesh_stats(m.V, m.F, "float32" if detect_format(path) in FLOAT32_FORMATS else "float64")
    except MeshcutError:
        raise
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        if isinstance(e, OSError) and oserror_message(e):
            raise MeshcutError(oserror_message(e)) from e
        raise MeshcutError((str(e).strip().splitlines() or [type(e).__name__])[0]) from e


def labels(input, output, grid):
    _run("labels", [input, output], {"grid": grid})
    return Path(output)


__all__ = ["__version__", "MeshcutError", "Mesh", "Plane", "split", "join", "verify", "info", "labels", "load_mesh",
           "split_planes", "split_labels", "triangle_hash", "mesh_stats", "parse_plane"]


if __name__ == "__main__":
    sys.exit(main())
