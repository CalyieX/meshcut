"""meshcut edit - Teile eines Netzes nummerieren, entfernen und ersetzen, mit allem, was das Netz traegt.

Gedacht fuer KI-Assistenten (Leitidee im README): der Assistent nennt nur das Ziel ("Teil 7 weg", "ersetze die Kugel durch einen
Zylinder"), meshcut misst am Original, entscheidet selbst und prueft am Ende, dass alles andere unveraendert ist.

OBJ wird vollstaendig mitgefuehrt (Eckpunkte, UV, Normalen, Material, Objekt-/Gruppennamen, Glaettungsgruppen); andere Formate
tragen nur Geometrie. Der exakte Schnitt (kkmeshcut.py) wird hier nicht angefasst.
"""
import codecs
import colorsys
import json
import math
import os
import re
import shutil
import struct
import uuid
import zlib
from pathlib import Path

import numpy as np

import kkmeshcut as mc
from kkmeshcut import MeshcutError, fail

KINDS = ("teil", "objekt", "gruppe", "material")
MAX_EDIT_FACES = 500_000                                       # Flaechen, die bearbeitet werden (jede ist ein Python-Objekt): groessere Netze vorher mit split teilen
MAX_REGIONS = 500                                                # mehr Elemente werden nicht nummeriert (die groessten bleiben)
MAX_LABELS = 60                                                  # so viele Nummern stehen im Uebersichtsbild
MAX_RENDER_TRIANGLES = 40_000
SCALE_RANGE = (0.01, 100.0)                                      # --scale: darunter/darueber entstuende ein unbrauchbarer Ersatz
SIZE_RANGE = (64, 4096)                                          # --size des Uebersichtsbildes in Pixeln                                    # mehr Dreiecke werden fuers Bild gleichmaessig ausgeduennt
CURVED_SPHERE_MAX_RISE = 0.37                                    # Anteil der Diagonale des Randes: eine Halbkugel hat die Hoehe R bei einer Diagonale von rund 2,83 R (0.354), dazu Spielraum fuer den gezackten Rand
STAR_TOL = 1e-9                                                  # Rundungsspielraum der Winkelreihenfolge des Randes (Bogenmass)
CURVED_SPHERE_FIT = 0.0002                                       # Kugelfit: Abweichung von Rand und erstem Ring (Anteil der Diagonale); ein Ellipsoid trifft eine Kugel nur grob und bekommt keine
BENT_RING = 0.003                                                # liegt der erste Nachbarring weiter als dies (Anteil der Diagonale) neben der Randebene, ist die Umgebung gewoelbt
CURVED_RIM_FLAT = 0.1                                            # der Rand einer Kappe schwankt hoechstens so stark (Anteil des Durchmessers) um seine Ebene; ein Dreiecksnetz zackt ihn
CURVED_MAX_RISE = 0.15                                            # Mitte des Paraboloids hoechstens so weit (Anteil des Durchmessers) ueber dem Rand: sonst ragt eine Spitze ins Leere, und flach ist besser
CURVED_MIN = 0.01                                                # Loch folgt der Wölbung, wenn die Mitte mehr als dieser Anteil des Durchmessers von der Randebene abweicht
CURVED_FIT = 0.02                                                # ... und der Paraboloid die Umgebung mit höchstens diesem Fehler (Anteil des Durchmessers) trifft
UV_FIT = 0.1                                                     # UV der gewoelbten Fuellung aus der Lage, wenn der Randfit hoechstens diesen Anteil der UV-Spanne danebenliegt
BAD_SHAPE = 0.15                                                 # swap auto: ab dieser Abweichung (Anteil der Diagonale) sagt meshcut, dass keine Form gut passt
CURVED_RINGS = (2 / 3, 1 / 3)                                    # Zwischenringe der gewölbten Füllung (Anteil des Abstands zur Mitte)
HOLE_PLANAR = 0.02                                               # Loch gilt als eben, wenn kein Randpunkt weiter als dieser Anteil des Durchmessers von der Ebene liegt


# ============================================================================= Szene (Netz mit allen Attributen)
class Face:
    __slots__ = ("c", "mat", "obj", "grp", "smooth")

    def __init__(self, c, mat=None, obj="", grp="", smooth=""):
        self.c, self.mat, self.obj, self.grp, self.smooth = c, mat, obj, grp, smooth      # c: Liste (v, vt, vn), -1 = fehlt


class Scene:
    """Ein Netz mit Polygonen und Attributen. v/vt/vn halten den Originaltext, damit unberuehrte Teile bitgleich zurueckkommen."""

    def __init__(self):
        self.V = np.zeros((0, 3))
        self.vt = []
        self.vn = []
        self.vtext = []
        self.faces = []
        self.mtllib = []
        self.fmt = "obj"
        self.dropped = 0                                         # Linien/Punkte, die nicht uebernommen werden
        self.escaped = False                                     # Eingabe war kein UTF-8: einzelne Bytes stehen als U+F780..U+F7FF im Text

    # ---- Zugriff
    def has_uv(self):
        return any(c[1] >= 0 for f in self.faces for c in f.c)

    def has_normals(self):
        return any(c[2] >= 0 for f in self.faces for c in f.c)

    def triangles(self, face_ids=None):
        """Fan-Dreiecke (a, b, c) ueber Eckpunktnummern, mit Herkunft (Flaechennummer)."""
        out, src = [], []
        for i in (range(len(self.faces)) if face_ids is None else face_ids):
            c = self.faces[i].c
            for a, b, d in self._polygon_triangles(c):
                out.append((c[a][0], c[b][0], c[d][0]))
                src.append(i)
        return np.array(out, dtype=np.int64).reshape(-1, 3), np.array(src, dtype=np.int64)

    def _polygon_triangles(self, c):
        """Zerlegung einer Flaeche in Dreiecke als Ecken-Positionen (a, b, c) innerhalb der Flaeche. Dreiecke und konvexe Vierecke als Faecher;
        alles andere per Ohrenschnitt in der Flaechenebene (ein Faecher legt bei konkaven Polygonen Dreiecke ausserhalb der Flaeche)."""
        n = len(c)
        if n <= 3:
            return [(0, 1, 2)] if n == 3 else []
        P = self.V[[x[0] for x in c]]
        if n == 4 and mc._convex_quad(P):
            return [(0, 1, 2), (0, 2, 3)]
        cen = P.mean(axis=0)
        nv = np.cross(P - cen, np.roll(P, -1, axis=0) - cen).sum(axis=0)
        nl = float(np.linalg.norm(nv))
        if nl > 0:
            t = _projected_ear_clip(P, cen, nv / nl)
            if t is not None:
                return [tuple(x) for x in t]
        return [(0, k, k + 1) for k in range(1, n - 1)]

    def n_triangles(self, face_ids=None):
        return sum(len(self.faces[i].c) - 2 for i in (range(len(self.faces)) if face_ids is None else face_ids))

    def add_vertex(self, xyz, text=None):
        self.V = np.vstack([self.V, np.asarray(xyz, dtype=np.float64).reshape(1, 3)])
        self.vtext.append(text or " ".join(repr(float(x)) for x in xyz))
        return len(self.V) - 1

    def add_vt(self, uv):
        self.vt.append(" ".join(repr(float(x)) for x in uv))
        return len(self.vt) - 1

    def add_vn(self, n):
        self.vn.append(" ".join(repr(float(x)) for x in n))
        return len(self.vn) - 1

    def copy(self):
        s = Scene()
        s.V, s.vt, s.vn, s.vtext = self.V.copy(), list(self.vt), list(self.vn), list(self.vtext)
        s.faces = list(self.faces)
        s.mtllib, s.fmt, s.dropped, s.escaped = list(self.mtllib), self.fmt, self.dropped, self.escaped
        return s

    def signature(self, f):
        """Wert einer Flaeche unabhaengig von Nummern: Text von Eckpunkt/UV/Normale je Ecke plus Material, Objekt, Gruppe, Glaettung."""
        return (tuple((self.vtext[v], self.vt[t] if t >= 0 else "", self.vn[n] if n >= 0 else "") for v, t, n in f.c),
                self.raw_name(f.mat), self.raw_name(f.obj), self.raw_name(f.grp), f.smooth)

    def raw_name(self, name):
        """Ein Name als die Bytes, die in der Datei stehen: so sind Namen vor und nach dem Schreiben vergleichbar, auch wenn die Ausgabe (anders als die Eingabe) reines UTF-8 ist."""
        return _encode_text(name, self.escaped) if name else name


ESCAPE_MARK = "\uf700"
ESCAPE_BASE = 0xF700                                               # Bytes ab 0x80 ausserhalb von UTF-8 werden zu U+F780..U+F7FF (druckbar, JSON-sicher);
_ESCAPE_TABLE = {0xDC00 + b: chr(ESCAPE_BASE + b) for b in range(0x80, 0x100)}      # echte Zeichen U+F700..U+F7FF bekommen dann U+F700 vorangestellt
_ESCAPE_PAIR = re.compile("\uf700([\uf700-\uf7ff])|([\uf780-\uf7ff])")


def _decode_text(data):
    """OBJ-Text: UTF-8 (mit oder ohne BOM). Bytes, die kein UTF-8 sind (Latin-1-Namen aus alten Exportern), bleiben erhalten und gehen unveraendert
    wieder hinaus, statt zu U+FFFD zu verschmelzen (umkehrbar, auch wenn der Text echte Zeichen aus dem Bereich U+F700..U+F7FF enthaelt).
    Rueckgabe: (Text, ob solche Bytes vorkamen)."""
    try:
        return data.decode("utf-8-sig"), False
    except UnicodeDecodeError:
        raw = data.decode("utf-8-sig", errors="surrogateescape")
        raw = re.sub("[\uf700-\uf7ff]", lambda m: ESCAPE_MARK + m.group(0), raw)
        return raw.translate(_ESCAPE_TABLE), True


def _encode_text(text, escaped):
    if not escaped:
        return text.encode("utf-8")
    out, pos = [], 0
    for m in _ESCAPE_PAIR.finditer(text):
        out.append(text[pos:m.start()].encode("utf-8"))
        out.append(m.group(1).encode("utf-8") if m.group(1) else bytes([ord(m.group(2)) - ESCAPE_BASE]))
        pos = m.end()
    out.append(text[pos:].encode("utf-8"))
    return b"".join(out)


def _plain(name):
    """Name zum Anzeigen und Vergleichen: eingeschmuggelte Bytes als Latin-1-Zeichen (so liest sie fast jeder alte Exporter), Markierungen entfernt."""
    name = _ESCAPE_PAIR.sub(lambda m: m.group(1) if m.group(1) else chr(ord(m.group(2)) - ESCAPE_BASE), name)
    return name


_LINE_BREAK = re.compile("\r\n|\r|\n")                              # nur diese trennen OBJ-Zeilen (nicht U+2028, U+0085, \x0b, \x0c wie bei splitlines)


def parse_obj(data):
    sc = Scene()
    mc_budget = mc._ReadBudget("OBJ")
    verts, v_text = [], []
    cur = {"mat": None, "obj": "", "grp": "", "smooth": ""}
    n_tri = 0
    text, sc.escaped = _decode_text(data)
    for ln in _LINE_BREAK.split(text):
        q = ln.split()
        if not q:
            continue
        tag = q[0]
        rest = ln.strip(" \t")[len(tag):].strip(" \t")
        if tag == "v":
            try:
                verts.append((float(q[1]), float(q[2]), float(q[3])))
            except (IndexError, ValueError):
                fail("OBJ: unlesbare Zeile '%s'" % ln.strip()[:40])
            v_text.append(rest)
        elif tag == "vt":
            sc.vt.append(rest)
        elif tag == "vn":
            sc.vn.append(rest)
        elif tag == "f":
            corners = []
            for tok in q[1:]:
                parts = tok.split("/")
                try:
                    idx = [int(p) if p else 0 for p in parts[:3]] + [0] * (3 - len(parts[:3]))
                except ValueError:
                    fail("OBJ: unlesbare Flaeche '%s'" % ln.strip()[:40])
                lim = (len(verts), len(sc.vt), len(sc.vn))
                c = []
                for k in range(3):
                    i = idx[k]
                    if i == 0:
                        if k == 0:
                            fail("OBJ: Flaeche ohne Eckpunkt '%s'" % ln.strip()[:40])
                        c.append(-1)
                        continue
                    i = i - 1 if i > 0 else lim[k] + i
                    if not 0 <= i < lim[k]:
                        fail("OBJ: Flaeche verweist auf einen Eintrag, den es nicht gibt '%s'" % ln.strip()[:40])
                    c.append(i)
                corners.append(tuple(c))
            if len(corners) < 3:
                fail("OBJ: Flaeche mit weniger als drei Ecken '%s'" % ln.strip()[:40])
            if len(corners) > mc.MAX_POLYGON:
                fail("Eine Flaeche hat %d Ecken - mehr als %d werden nicht bearbeitet." % (len(corners), mc.MAX_POLYGON))
            n_tri += len(corners) - 2
            mc_budget.triangles(len(corners) - 2)
            if len(sc.faces) >= MAX_EDIT_FACES:
                fail("Mehr als %d Flaechen werden nicht bearbeitet - das Netz vorher mit `meshcut split` teilen." % MAX_EDIT_FACES)
            sc.faces.append(Face(corners, cur["mat"], cur["obj"], cur["grp"], cur["smooth"]))
        elif tag == "o":
            cur["obj"] = rest
        elif tag == "g":
            cur["grp"] = rest
        elif tag == "usemtl":
            cur["mat"] = None if rest in ("", "off") else rest          # "usemtl off" heisst: kein Material
        elif tag == "s":
            cur["smooth"] = rest
        elif tag == "mtllib":
            if rest:
                sc.mtllib.append(rest)
        elif tag in ("l", "p"):
            sc.dropped += 1
    sc.V = np.array(verts, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(sc.V).all():
        fail("Das Netz enthaelt NaN oder unendliche Koordinaten.")
    sc.vtext = v_text
    if not sc.faces:
        fail("Die Datei enthaelt keine Flaechen.")
    return sc


def scene_from_mesh(mesh, fmt):
    sc = Scene()
    sc.fmt = fmt
    sc.V = mesh.V.copy()
    sc.vtext = [" ".join(repr(float(x)) for x in v) for v in mesh.V]
    sc.faces = [Face([(int(a), -1, -1) for a in f]) for f in mesh.F]
    return sc


def read_scene(path):
    p = Path(path)
    fmt = mc.detect_format(p)
    if fmt == "obj":
        data = mc._read_bytes(p)
        return parse_obj(data)
    mesh = mc.load_mesh(p)
    if len(mesh.F) > MAX_EDIT_FACES:
        fail("Mehr als %d Flaechen werden nicht bearbeitet - das Netz vorher mit `meshcut split` teilen." % MAX_EDIT_FACES)
    sc = scene_from_mesh(mesh, fmt)
    if not sc.faces:
        fail("%s enthaelt keine Dreiecke." % p.name)
    return sc


def compact(sc):
    """Nicht mehr gebrauchte v/vt/vn weglassen, Reihenfolge der uebrigen bleibt. Rueckgabe: neue Szene."""
    used = [set(), set(), set()]
    for f in sc.faces:
        for c in f.c:
            for k in range(3):
                if c[k] >= 0:
                    used[k].add(c[k])
    maps = []
    for k, n in enumerate((len(sc.V), len(sc.vt), len(sc.vn))):
        m = {}
        for i in sorted(used[k]):
            m[i] = len(m)
        maps.append(m)
    out = Scene()
    out.fmt, out.mtllib, out.dropped, out.escaped = sc.fmt, list(sc.mtllib), sc.dropped, sc.escaped
    keep = sorted(used[0])
    out.V = sc.V[keep] if keep else np.zeros((0, 3))
    out.vtext = [sc.vtext[i] for i in keep]
    out.vt = [sc.vt[i] for i in sorted(used[1])]
    out.vn = [sc.vn[i] for i in sorted(used[2])]
    for f in sc.faces:
        out.faces.append(Face([tuple(maps[k][c[k]] if c[k] >= 0 else -1 for k in range(3)) for c in f.c], f.mat, f.obj, f.grp, f.smooth))
    return out


def obj_text(sc):
    sc = compact(sc)
    lines = ["# meshcut edit"]
    lines += ["mtllib " + m for m in sc.mtllib]
    lines += ["v " + t for t in sc.vtext]
    lines += ["vt " + t for t in sc.vt]
    lines += ["vn " + t for t in sc.vn]
    cur = {"obj": "", "grp": "", "mat": None, "smooth": ""}
    for f in sc.faces:
        if f.obj != cur["obj"]:
            lines.append("o " + f.obj if f.obj else "o")
            cur["obj"] = f.obj
        if f.grp != cur["grp"]:
            lines.append("g " + f.grp if f.grp else "g")
            cur["grp"] = f.grp
        if f.mat != cur["mat"]:
            lines.append("usemtl " + (f.mat if f.mat is not None else "off"))   # ohne diese Zeile erbte die Flaeche das Material davor
            cur["mat"] = f.mat
        if f.smooth != cur["smooth"]:
            lines.append("s " + (f.smooth or "off"))
            cur["smooth"] = f.smooth
        toks = []
        for v, t, n in f.c:
            s = str(v + 1)
            if t >= 0 and n >= 0:
                s += "/%d/%d" % (t + 1, n + 1)
            elif t >= 0:
                s += "/%d" % (t + 1)
            elif n >= 0:
                s += "//%d" % (n + 1)
            toks.append(s)
        lines.append("f " + " ".join(toks))
    return _encode_text("\n".join(lines) + "\n", sc.escaped)


def scene_bytes(sc, fmt):
    if fmt == "obj":
        return obj_text(sc)
    c = compact(sc)
    V = c.V
    F, _ = c.triangles()
    name = "edit"
    if fmt in ("3mf", "glb"):
        return (mc.threemf_bytes if fmt == "3mf" else mc.glb_bytes)([(name, V, F)])
    return mc.mesh_bytes(fmt, V, F, name)


# ============================================================================= Zusammenhang, Kennzahlen
def position_ids(sc):
    if not len(sc.V):
        return np.zeros(0, dtype=np.int64)
    _, inv = np.unique(sc.V, axis=0, return_inverse=True)
    return inv.reshape(-1).astype(np.int64)


def component_labels(sc, pid):
    """Zusammenhangskomponenten ueber gemeinsame Positionen (UV- und Normalen-Nähte trennen nichts)."""
    parent = list(range(int(pid.max()) + 1 if len(pid) else 0))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for f in sc.faces:
        a = find(int(pid[f.c[0][0]]))
        for c in f.c[1:]:
            b = find(int(pid[c[0]]))
            if a != b:
                parent[b] = a
    return np.array([find(int(pid[f.c[0][0]])) for f in sc.faces], dtype=np.int64)


def _fan_normals(sc, face_ids):
    """Flaechenvektor (Betrag = doppelte Flaeche) je Flaeche (Newell)."""
    out = np.zeros((len(face_ids), 3))
    for j, i in enumerate(face_ids):
        c = sc.faces[i].c
        P = sc.V[[x[0] for x in c]]
        out[j] = np.cross(P, np.roll(P, -1, axis=0)).sum(axis=0)
    return out


def edge_counts(sc, pid, face_ids=None):
    cnt = {}
    for i in (range(len(sc.faces)) if face_ids is None else face_ids):
        c = sc.faces[i].c
        for k in range(len(c)):
            a, b = int(pid[c[k][0]]), int(pid[c[(k + 1) % len(c)][0]])
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            cnt[key] = cnt.get(key, 0) + 1
    return cnt


def region_stats(sc, pid, face_ids):
    fid = np.asarray(face_ids, dtype=np.int64)
    nv = _fan_normals(sc, fid)
    area_f = np.linalg.norm(nv, axis=1) / 2
    verts = sorted({c[0] for i in fid for c in sc.faces[i].c})
    P = sc.V[verts]
    lo, hi = P.min(axis=0), P.max(axis=0)
    big = int(fid[int(np.argmax(area_f))])
    point = sc.V[[c[0] for c in sc.faces[big].c]].mean(axis=0)            # Mitte der groessten Flaeche: liegt auf der Oberflaeche
    cnt = edge_counts(sc, pid, fid)
    closed = bool(cnt) and all(v == 2 for v in cnt.values())
    vol = None
    if closed:
        T, _ = sc.triangles(fid)
        A, B, C = sc.V[T[:, 0]], sc.V[T[:, 1]], sc.V[T[:, 2]]
        vol = abs(float(np.einsum("ij,ij->i", A, np.cross(B, C)).sum()) / 6)
    mats = {}
    for j, i in enumerate(fid):
        m = sc.faces[i].mat or ""
        mats[m] = mats.get(m, 0.0) + float(area_f[j])
    return {"faces": int(len(fid)), "triangles": int(sc.n_triangles(fid)), "area": float(area_f.sum()), "closed": closed,
            "volume": vol, "bbox": {"min": lo.tolist(), "max": hi.tolist()}, "center": ((lo + hi) / 2).tolist(), "point": point.tolist(),
            "materials": [_plain(m) if sc.escaped else m for m, _ in sorted(mats.items(), key=lambda kv: -kv[1]) if m]}


def find_regions(sc):
    """Alle unterscheidbaren Elemente: zusammenhaengende Teile, Objekte, Gruppen, Materialien. Nummern ab 1, in dieser Reihenfolge;
    ein Element, das genau dieselben Flaechen hat wie ein frueheres, kommt nicht doppelt vor."""
    shown = _plain if sc.escaped else str                          # Anzeigename: eingeschmuggelte Bytes als Latin-1-Zeichen
    pid = position_ids(sc)
    n = len(sc.faces)
    lab = component_labels(sc, pid)
    groups = []
    order = {}
    for i, l in enumerate(lab):
        order.setdefault(int(l), []).append(i)
    for faces in order.values():
        groups.append(("teil", "", faces))
    for kind, key in (("objekt", lambda f: f.obj), ("gruppe", lambda f: f.grp), ("material", lambda f: f.mat or "")):
        d = {}
        for i, f in enumerate(sc.faces):
            k = key(f)
            if k:
                d.setdefault(k, []).append(i)
        for name, faces in d.items():
            groups.append((kind, name, faces))
    cand, seen, also = [], {}, {}
    for kind, name, faces in groups:
        if len(faces) == n:                                       # das ganze Netz ist nichts zum Auswaehlen
            continue
        key = tuple(faces)
        if key in seen:
            if name:                                              # dieselben Flaechen unter anderem Namen: der Name zeigt auf das fruehere Element
                also.setdefault(seen[key], []).append({"kind": kind, "name": shown(name)})
            continue
        seen[key] = len(cand)
        cand.append((kind, name, faces))
    if len(cand) > MAX_REGIONS:                                   # die groessten bleiben, in ihrer Reihenfolge
        keep = sorted(sorted(range(len(cand)), key=lambda i: -len(cand[i][2]))[:MAX_REGIONS])
        cand = [cand[i] for i in keep]
    regions = [dict(id=i + 1, kind=k, name=shown(nm), face_ids=f, also=also.get(i, [])) for i, (k, nm, f) in enumerate(cand)]
    omitted = len(seen) - len(cand)
    smallest = np.zeros(n, dtype=np.int64)                        # je Flaeche das kleinste Element, das sie enthaelt (das waehlt --at)
    for r in sorted(regions, key=lambda r: (-len(r["face_ids"]), -r["id"])):      # bei gleicher Groesse gewinnt die kleinere Nummer (wie choose_region)
        smallest[r["face_ids"]] = r["id"]
    for r in regions:
        r.update(region_stats(sc, pid, r["face_ids"]))
        if not r["name"]:
            r["name"] = "%s %d" % (r["kind"].capitalize(), r["id"])
        fid = np.asarray(r["face_ids"], dtype=np.int64)
        area = np.linalg.norm(_fan_normals(sc, fid), axis=1)
        mine = np.nonzero(smallest[fid] == r["id"])[0]
        j = int(mine[np.argmax(area[mine])]) if len(mine) else int(np.argmax(area))
        r["point"] = sc.V[[c[0] for c in sc.faces[int(fid[j])].c]].mean(axis=0).tolist()
        r["at_selects_this"] = bool(len(mine))
    find_regions.omitted = omitted
    return regions


# ============================================================================= Uebersichtsbild
_FONT = {"0": "01110100011001110101110011000101110", "1": "00100011000010000100001000010001110", "2": "01110100010000100010001000100011111",
         "3": "11110000010000101110000010000111110", "4": "00010001100101010010111110001000010", "5": "11111100001111000001000011000101110",
         "6": "00110010001000011110100011000101110", "7": "11111000010001000100010000100001000", "8": "01110100011000101110100011000101110",
         "9": "01110100011000101111000010001001100"}


def _color(i, n):
    h = (i * 0.61803398875) % 1.0
    return np.array(colorsys.hsv_to_rgb(h, 0.55, 0.95)) * 255


def png_bytes(img):
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def _draw_text(img, text, x, y, scale=3):
    w = len(text) * 6 * scale + 2 * scale
    h = 7 * scale + 2 * scale
    x0, y0 = max(0, min(img.shape[1] - w, x - w // 2)), max(0, min(img.shape[0] - h, y - h // 2))
    img[y0:y0 + h, x0:x0 + w] = (0, 0, 0)
    for k, ch in enumerate(text):
        bits = _FONT[ch]
        for r in range(7):
            for c in range(5):
                if bits[r * 5 + c] == "1":
                    px, py = x0 + scale + (k * 6 + c) * scale, y0 + scale + r * scale
                    img[py:py + scale, px:px + scale] = (255, 255, 255)


def render_overview(sc, regions, size=720):
    """Schraegansicht mit Z-Puffer, jede Flaeche in der Farbe ihres kleinsten Elements, Nummern an den Punkten der Elemente."""
    T, src = sc.triangles()
    if len(T) > MAX_RENDER_TRIANGLES:
        keep = np.linspace(0, len(T) - 1, MAX_RENDER_TRIANGLES).astype(np.int64)
        T, src = T[keep], src[keep]
    colorid = np.zeros(len(sc.faces), dtype=np.int64)
    for r in sorted(regions, key=lambda r: -r["faces"]):          # kleinere Elemente spaeter: ueberschreiben die groesseren
        colorid[r["face_ids"]] = r["id"]
    az, el = math.radians(-35), math.radians(28)
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(el), -math.sin(el)], [0, math.sin(el), math.cos(el)]])
    R = Rx @ Rz
    Q = sc.V @ R.T
    lo, hi = Q[:, :2].min(axis=0), Q[:, :2].max(axis=0)
    scale = (size * 0.9) / (float((hi - lo).max()) or 1.0)
    off = (size - (hi - lo) * scale) / 2
    X = (Q[:, 0] - lo[0]) * scale + off[0]
    Y = size - ((Q[:, 1] - lo[1]) * scale + off[1])
    Z = Q[:, 2]
    img = np.full((size, size, 3), 245, dtype=np.uint8)
    zbuf = np.full((size, size), -np.inf)
    fbuf = np.full((size, size), -1, dtype=np.int64)
    n_col = max(len(regions), 1)
    for t, f in zip(T, src):
        a, b, c = t
        xs, ys = X[[a, b, c]], Y[[a, b, c]]
        x0, x1 = int(max(0, math.floor(xs.min()))), int(min(size - 1, math.ceil(xs.max())))
        y0, y1 = int(max(0, math.floor(ys.min()))), int(min(size - 1, math.ceil(ys.max())))
        if x1 < x0 or y1 < y0:
            continue
        den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(den) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        l1 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
        l2 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        if not inside.any():
            continue
        z = l1 * Z[a] + l2 * Z[b] + l3 * Z[c]
        win = zbuf[y0:y1 + 1, x0:x1 + 1]
        upd = inside & (z > win)
        if not upd.any():
            continue
        nrm = np.cross(Q[b] - Q[a], Q[c] - Q[a])
        ln = np.linalg.norm(nrm)
        shade = 0.35 + 0.65 * abs(float(nrm[2]) / ln) if ln > 0 else 0.5
        cid = colorid[f]
        col = (_color(cid, n_col) if cid else np.array([200.0, 200.0, 200.0])) * shade
        win[upd] = z[upd]
        fbuf[y0:y1 + 1, x0:x1 + 1][upd] = f
        img[y0:y1 + 1, x0:x1 + 1][upd] = col.astype(np.uint8)
    visible, placed = {}, []
    box = 7 * 3 + 6
    for r in regions:
        ys, xs = np.nonzero(np.isin(fbuf, np.asarray(r["face_ids"], dtype=np.int64)))
        visible[r["id"]] = bool(len(xs))
        if not len(xs) or len(placed) >= MAX_LABELS:
            continue
        cx, cy = xs.mean(), ys.mean()
        j = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))        # sichtbares Pixel nahe der Mitte
        x, y = int(xs[j]), int(ys[j])
        while any(abs(x - px) < box * 1.4 and abs(y - py) < box for px, py in placed) and y < size - box:
            y += box
        placed.append((x, y))
        _draw_text(img, str(r["id"]), x, y)
    return png_bytes(img), visible


# ============================================================================= Auswahl
def parse_point(text):
    try:
        v = [float(x) for x in str(text).replace(";", ",").split(",")]
    except ValueError:
        v = []
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        fail("Punkt '%s' nicht verstanden - Beispiel: 1.5,0,2" % str(text)[:30])
    return np.array(v)


def nearest_face(sc, point):
    """Flaeche mit dem kleinsten Abstand zum Punkt. Rueckgabe: (Flaechennummer, Abstand)."""
    T, src = sc.triangles()
    p = np.repeat(np.asarray(point, dtype=np.float64).reshape(1, 3), len(T), axis=0)
    cp = mc._closest_points_on_triangles(p, sc.V[T[:, 0]], sc.V[T[:, 1]], sc.V[T[:, 2]])
    d = np.linalg.norm(cp - p, axis=1)
    j = int(np.argmin(d))
    return int(src[j]), float(d[j])


def choose_region(sc, regions, region=None, at=None, name=None):
    given = [x for x in (region, at, name) if x not in (None, "")]
    if len(given) != 1:
        fail("Genau eine Auswahl angeben: --region NUMMER (aus regions.json), --at x,y,z oder --name NAME.")
    if not regions:
        fail("Das Netz ist ein einziges, einheitliches Stueck - es gibt nichts auszuwaehlen.")
    if region not in (None, ""):
        try:
            if isinstance(region, bool):
                raise ValueError
            rid = int(region)
        except (TypeError, ValueError):
            fail("Die Nummer '%s' ist keine ganze Zahl." % str(region)[:20])
        if not 1 <= rid <= len(regions):
            fail("Es gibt keine Region %d - gueltig sind 1 bis %d (siehe `meshcut regions`)." % (rid, len(regions)))
        return regions[rid - 1]
    if name not in (None, ""):
        want = str(name).strip(" \t").lower()
        hit = [r for r in regions if r["name"].lower() == want or any(a["name"].lower() == want for a in r["also"])]
        if not hit:
            names = []
            for r in regions:
                names += [r["name"]] + [a["name"] for a in r["also"]]
            fail("Kein Element heisst '%s' - vorhanden: %s%s" % (str(name)[:40], ", ".join(names[:15]), _omitted_note()))
        if len(hit) > 1:                                          # Namen, die sich nur in der Gross-/Kleinschreibung unterscheiden: die genaue Schreibweise gewinnt
            exact = [r for r in hit if r["name"] == str(name).strip(" 	") or any(a["name"] == str(name).strip(" 	") for a in r["also"])]
            hit = exact if len(exact) == 1 else hit
        if len(hit) > 1:
            fail("Der Name '%s' passt auf mehrere Elemente (%s) - mit --region NUMMER waehlen." % (str(name)[:40], ", ".join(str(r["id"]) for r in hit[:8])))
        return hit[0]
    p = parse_point(at)
    fi, dist = nearest_face(sc, p)
    cand = [r for r in regions if fi in set(r["face_ids"])]
    if not cand:
        fail("Am Punkt %s liegt kein auswaehlbares Element.%s" % (str(at)[:30], _omitted_note()))
    return min(cand, key=lambda r: (r["faces"], r["id"]))


def _omitted_note():
    om = getattr(find_regions, "omitted", 0)
    return " (%d kleinere Elemente sind nicht nummeriert, hoechstens %d - ein kleineres Netz waehlen oder vorher mit split teilen)" % (om, MAX_REGIONS) if om else ""


# ============================================================================= Loch schliessen
def _hole_loops(sc, pid, removed):
    """Raender der Restflaechen, an denen entfernte Flaechen sassen: Liste von Schleifen aus (pid, v, vt, vn, Flaechennummer),
    in Umlaufrichtung der Restflaechen."""
    rem = set(removed)
    removed_edges = set()
    for i in rem:
        c = sc.faces[i].c
        for k in range(len(c)):
            a, b = int(pid[c[k][0]]), int(pid[c[(k + 1) % len(c)][0]])
            removed_edges.add((a, b) if a < b else (b, a))
    cnt, directed = {}, []
    for i, f in enumerate(sc.faces):
        if i in rem:
            continue
        c = f.c
        for k in range(len(c)):
            a, b = int(pid[c[k][0]]), int(pid[c[(k + 1) % len(c)][0]])
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            cnt[key] = cnt.get(key, 0) + 1
            directed.append((a, b, i, k))
    edges = [e for e in directed if ((e[0], e[1]) in removed_edges or (e[1], e[0]) in removed_edges)
             and cnt[(e[0], e[1]) if e[0] < e[1] else (e[1], e[0])] == 1]
    nxt = {}
    for e in edges:
        nxt.setdefault(e[0], []).append(e)
    loops, used = [], set()
    for e0 in edges:
        if (e0[2], e0[3]) in used:
            continue
        loop, at, e = [], {}, e0                                  # at: Startecke -> Stelle im Weg (ein Weg, der eine Ecke wieder trifft, schliesst dort eine Schleife)
        while True:
            used.add((e[2], e[3]))
            c = sc.faces[e[2]].c[e[3]]
            at[e[0]] = len(loop)
            loop.append((e[0], c[0], c[1], c[2], e[2]))
            nv = e[1]
            if nv in at:                                          # Schleife geschlossen (bei einer Beruehrungsecke zweier Loecher: das innere Loch fuer sich)
                k = at[nv]
                if len(loop) - k >= 3:
                    loops.append(loop[k:])
                for x in loop[k:]:
                    at.pop(x[0], None)
                loop = loop[:k]
                if not loop:
                    break
            cands = [x for x in nxt.get(nv, []) if (x[2], x[3]) not in used]
            if not cands:
                break
            e = cands[0]
    return loops


def _second_ring(sc, rloop):
    """Eckpunkte der Restflaechen, die an den ersten Nachbarring grenzen (ohne die entfernten Flaechen)."""
    ctx = getattr(sc, "hole_ctx", None)
    if ctx is None:
        return np.zeros((0, 3))
    pid, removed = ctx
    near = {int(pid[c[0]]) for x in rloop for c in sc.faces[x[4]].c}
    seen, pts = set(), []
    for i, f in enumerate(sc.faces):
        if i in removed or not any(int(pid[c[0]]) in near for c in f.c):
            continue
        for c in f.c:
            q = int(pid[c[0]])
            if q not in near and q not in seen:
                seen.add(q)
                pts.append(sc.V[c[0]])
    return np.array(pts) if pts else np.zeros((0, 3))


def _sphere_height(a, b, w, nrim, diam, nnear=None):
    """Kugel durch Rand und ersten Nachbarring (kleinste Quadrate): die Fuellung folgt einer Kappe genau, wo ein Paraboloid sie ueberschiesst.
    a, b, w: Lage der Punkte in Durchmessern (in der Randebene und darueber); die ersten nrim Punkte sind der Rand.
    Rueckgabe: Funktion h(a, b) in Originaleinheiten oder None, wenn die Punkte keiner Kugel folgen. Laengen sind Anteile der Diagonale des Randes (bei einem Kreis rund das 1,41-fache seines Durchmessers)."""
    M = np.stack([a, b, w, np.ones(len(a))], axis=1)
    y = -(a * a + b * b + w * w)
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    ca, cb, cw = -coef[0] / 2, -coef[1] / 2, -coef[2] / 2
    r2 = ca * ca + cb * cb + cw * cw - coef[3]
    if not np.isfinite(r2) or r2 <= 0:
        return None
    rad = math.sqrt(r2)
    dist = np.sqrt((a - ca) ** 2 + (b - cb) ** 2 + (w - cw) ** 2)
    if float(np.sqrt(np.mean((dist - rad) ** 2))) > CURVED_SPHERE_FIT or rad < 0.33:
        return None                                              # keine Kugel, oder kleiner als eine Halbkugel ueber dem Loch
    if float(np.ptp(w[:nrim])) > CURVED_RIM_FLAT:
        return None                                              # der Rand einer Kappe liegt (fast) in einer Ebene; ein verwundener Rand ist keine Kappe
    # Die Kappe liegt auf der Seite der Randebene, von der die Nachbarflaechen wegfuehren (sie liegen auf der stehenden Kugelschale, nicht ueber dem Loch).
    # Das Vorzeichen allein aus der Lage des Randes zum Kugelmittelpunkt zu nehmen, kippt bei Kappen ueber der Halbkugel auf die falsche Schale.
    nnear = nnear or len(w)                                      # Nachbarn zaehlen nur der erste Ring, nicht der zweite
    away = float(w[:nrim].mean()) - float(w[nrim:nnear].mean()) if nnear > nrim else float(w[:nrim].mean()) - cw
    sign = 1.0 if away > 0 else -1.0

    def h(x, y_):
        x, y_ = x / diam, y_ / diam
        return diam * (cw + sign * math.sqrt(max(rad * rad - (x - ca) ** 2 - (y_ - cb) ** 2, 0.0)))
    return h


def _curved_fill(sc, rloop, Pr, cen, n):
    """Wölbungsfolgende Füllung: ein Paraboloid h(u,v) wird an den Rand und den ersten Nachbarring gefittet (kleinste Quadrate); weicht seine
    Mitte spürbar von der Randebene ab und trifft er die Umgebung gut, entstehen zwei Zwischenringe und eine Mitte auf dieser Fläche.
    Rückgabe: Liste von Dreiecken aus Ecken-Indizes (Ring, Nr.) oder None, wenn die Füllung flach bleiben soll."""
    diam = float(np.linalg.norm(Pr.max(axis=0) - Pr.min(axis=0)))
    ref_axis = np.zeros(3)
    ref_axis[int(np.argmin(np.abs(n)))] = 1.0
    u = np.cross(n, ref_axis)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    rel = Pr - cen
    uv = np.stack([rel @ u, rel @ v], axis=1)
    ang = np.unwrap(np.arctan2(uv[:, 1], uv[:, 0]))
    star = bool(len(Pr) >= 6 and np.all(np.diff(ang) > -STAR_TOL) and float(ang[-1] - ang[0]) < 2 * math.pi + STAR_TOL)     # sonst wuerden sich Zwischenringe ueberlappen; zwei Randpunkte auf demselben Strahl (Zickzack-Rand eines Dreiecksnetzes) sind erlaubt
    if len(Pr) < 3:
        return None
    pts = [Pr]
    for x in rloop:                                               # erster Nachbarring: die uebrigen Ecken der anliegenden Restflaechen
        pts.append(sc.V[[c[0] for c in sc.faces[x[4]].c]])
    P = np.vstack(pts)
    r = (P - cen) / diam                                          # Laengen in Durchmessern: der Rangtest haengt so nicht vom Massstab ab
    A = np.stack([np.ones(len(P)), r @ u, r @ v, (r @ u) ** 2, (r @ u) * (r @ v), (r @ v) ** 2], axis=1)
    w = r @ n
    if np.linalg.matrix_rank(A) < 6:
        return None
    far = _second_ring(sc, rloop)                                 # ein Ellipsoid trifft Rand und ersten Ring noch wie eine Kugel, einen zweiten Ring nicht mehr
    if len(far):
        rf = (far - cen) / diam
        sph = _sphere_height(np.concatenate([r @ u, rf @ u]), np.concatenate([r @ v, rf @ v]), np.concatenate([w, rf @ n]), len(Pr), diam,
                             nnear=len(P))
    else:
        sph = _sphere_height(r @ u, r @ v, w, len(Pr), diam)
    if sph is not None:
        hb = np.array([sph(a, b) for a, b in uv])
        rise = abs(float(sph(0.0, 0.0)) - float(hb.mean()))
        if rise > CURVED_SPHERE_MAX_RISE * diam:
            fail("Das Loch liegt auf einer Kugel und ist groesser als eine Halbkugel; eine Fuellung waere geraten. "
                 "Ein kleineres Element waehlen oder die Kappe mit split abtrennen.")
        if CURVED_MIN * diam <= rise:
            return {"u": u, "v": v, "uv": uv, "h": sph, "hb": hb, "star": star}
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    if float(np.sqrt(np.mean((A @ coef - w) ** 2))) > CURVED_FIT:
        return None

    def h(a, b):                                                  # Hoehe in Originaleinheiten an der Stelle (a, b) in der Randebene
        a, b = a / diam, b / diam
        return diam * (coef[0] + coef[1] * a + coef[2] * b + coef[3] * a * a + coef[4] * a * b + coef[5] * b * b)
    hb = np.array([h(a, b) for a, b in uv])
    rise = abs(float(h(0.0, 0.0)) - float(hb.mean()))
    if rise < CURVED_MIN * diam or rise > CURVED_MAX_RISE * diam:
        return None
    return {"u": u, "v": v, "uv": uv, "h": h, "hb": hb, "star": star}


def _projected_ear_clip(Pr, cen, n):
    """Ohrenschnitt des Randes, projiziert auf die Randebene (Normale n). Rueckgabe: Dreiecke (Randindizes) oder None, wenn die Projektion
    sich ueberschlaegt (Ohrenschnitt scheitert, oder ein Dreieck zeigt gegen n)."""
    ref_axis = np.zeros(3)
    ref_axis[int(np.argmin(np.abs(n)))] = 1.0
    u = np.cross(n, ref_axis)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    P2 = {i: (float((p - cen) @ u), float((p - cen) @ v)) for i, p in enumerate(Pr)}
    try:
        t, emerg = mc.ear_clip(P2, list(range(len(Pr))))
    except MeshcutError:
        return None
    if t is None or emerg or len(t) != len(Pr) - 2:
        return None
    for a, b, c in t:
        if float(np.cross(Pr[b] - Pr[a], Pr[c] - Pr[a]) @ n) <= 0:
            return None
    return t


def _fan_is_consistent(Pr, cen, n):
    """Ein Faecher vom Mittelpunkt ist nur gut, wenn alle seine Dreiecke zur Randebene gleich herum zeigen (Rand sternfoermig um die Mitte)."""
    for k in range(len(Pr)):
        if float(np.cross(Pr[k] - cen, Pr[(k + 1) % len(Pr)] - cen) @ n) <= 0:
            return False
    return True


def _uv_of(text):
    """(u, v) aus einem vt-Eintrag. Ein Eintrag mit nur u zaehlt als (u, 0); leer, nicht numerisch oder nicht endlich: None."""
    try:
        t = [float(x) for x in text.split()[:2]]
    except ValueError:
        return None
    if not t or not all(math.isfinite(x) for x in t):
        return None
    return [t[0], t[1] if len(t) > 1 else 0.0]


def _rim_uv(sc, rloop):
    """UV der Randecken als Feld (n, 2) oder None, wenn eine Ecke keine brauchbare UV hat."""
    if any(x[2] < 0 for x in rloop):
        return None
    uv = [_uv_of(sc.vt[x[2]]) for x in rloop]
    return None if any(u is None for u in uv) else np.array(uv)


def _unwrap_uv_seam(sc, rloop):
    """Liegt das Loch auf einer UV-Naht (am Rand springt u oder v von fast 1 auf fast 0), bekommen die Randecken der neuen Flaechen fortlaufende UV-Werte
    (z. B. 1,02 statt 0,02; die Textur wiederholt sich), sonst wuerde die Fuellung die ganze Textur in ein Dreieck spannen. Die vorhandenen Flaechen bleiben
    unberuehrt: es entstehen neue vt-Eintraege nur fuer die neuen Ecken. Umschliesst der Rand die ganze Textur (Pol), bleibt es bei den Werten."""
    arr = _rim_uv(sc, rloop)
    if arr is None:
        return rloop
    out = arr.copy()
    for c in (0, 1):
        col = arr[:, c]
        w, shift = col.copy(), 0
        for k in range(1, len(col)):
            d = col[k] - col[k - 1]
            shift += -1 if d > 0.5 else (1 if d < -0.5 else 0)
            w[k] = col[k] + shift
        d = col[0] - col[-1]
        shift += -1 if d > 0.5 else (1 if d < -0.5 else 0)
        if shift == 0:                                            # sonst laeuft der Rand einmal um die Textur: nicht entwirrbar
            out[:, c] = w
    if np.array_equal(out, arr):
        return rloop
    return [(x[0], x[1], sc.add_vt(out[k]), x[3], x[4]) for k, x in enumerate(rloop)]


def close_holes(sc, pid, removed):
    """Schliesst die Loecher, die das Entfernen hinterlaesst. Rueckgabe: (neue Flaechen, Anzahl geschlossener Loecher,
    Anzahl der Loecher mit nicht ebenem Rand, die flach gefuellt wurden)."""
    loops = _hole_loops(sc, pid, removed)
    new, closed, uneven, bent = [], 0, 0, 0
    sc.hole_ctx = (pid, set(removed))
    for loop in loops:
        rloop = loop[::-1]                                        # Umlaufrichtung der neuen Flaechen ist die Gegenrichtung
        rloop = _unwrap_uv_seam(sc, rloop)
        Pr = sc.V[[x[1] for x in rloop]]
        cen = Pr.mean(axis=0)
        n = np.cross(Pr - cen, np.roll(Pr, -1, axis=0) - cen).sum(axis=0)
        nl = float(np.linalg.norm(n))
        diam = float(np.linalg.norm(Pr.max(axis=0) - Pr.min(axis=0)))
        faces_adj = [x[4] for x in loop]
        mats = {}
        for fi in faces_adj:
            mats[sc.faces[fi].mat] = mats.get(sc.faces[fi].mat, 0) + 1
        ref = sc.faces[max(faces_adj, key=lambda fi: mats[sc.faces[fi].mat])]
        has_n = sc.has_normals()
        tris, flat, curved = None, False, None
        if nl > 0 and diam > 0:
            n = n / nl
            curved = _curved_fill(sc, rloop, Pr, cen, n)
            flat = float(np.abs((Pr - cen) @ n).max()) <= HOLE_PLANAR * diam
            if curved is None or not curved["star"]:             # keine Woelbung gefunden: die Fuellung wird flach, und das Volumen schrumpft, wenn die Umgebung gewoelbt ist
                ring = np.vstack([sc.V[[c[0] for c in sc.faces[x[4]].c]] for x in rloop])
                bent += int(float(np.abs((ring - cen) @ n).max()) > BENT_RING * diam)
            tris = _projected_ear_clip(Pr, cen, n)                  # in der Randebene triangulieren, auch bei gewoelbtem oder konkavem Rand
            fan_ok = _fan_is_consistent(Pr, cen, n)
            if tris is not None and fan_ok:                         # Pruefer: Ohrenschnitt gegen Faecher vom Mittelpunkt
                qe = min(_quality(Pr[list(x)]) for x in tris)
                qf = min(_quality(np.array([Pr[k], Pr[(k + 1) % len(Pr)], cen])) for k in range(len(Pr)))
                if qe < SLIVER and qf > 3 * qe:
                    tris = None
            if tris is None and not fan_ok and not (curved is not None and curved["star"]):
                fail("Das Loch ist zu verwickelt, um es sauber zu schliessen (Rand mit %d Punkten, weder eben noch sternfoermig). "
                     "Ein kleineres Element waehlen oder das Netz vorher reparieren." % len(Pr))
        if curved is not None and curved["star"]:
            new += _fill_with_rings(sc, rloop, Pr, cen, n, curved, ref, has_n)
        elif tris is not None:
            uneven += 0 if flat else 1
            vn_flat = sc.add_vn(n) if (flat and has_n) else -1
            for a_, b_, c_ in tris:
                vn = vn_flat
                if has_n and not flat:                              # gewoelbter Rand: jedes Dreieck bekommt seine eigene Normale
                    tn = np.cross(Pr[b_] - Pr[a_], Pr[c_] - Pr[a_])
                    tl = float(np.linalg.norm(tn))
                    vn = sc.add_vn(tn / tl) if tl > 0 else -1
                new.append(Face([(rloop[k][1], rloop[k][2], vn) for k in (a_, b_, c_)], ref.mat, ref.obj, ref.grp, ""))
        else:
            ci = sc.add_vertex(cen)
            arr = _rim_uv(sc, rloop)
            cvt = -1
            if arr is not None:
                cvt = sc.add_vt(arr.mean(axis=0))
            for k in range(len(rloop)):
                a_, b_ = rloop[k], rloop[(k + 1) % len(rloop)]
                tn = np.cross(sc.V[b_[1]] - sc.V[a_[1]], sc.V[ci] - sc.V[a_[1]])
                tl = float(np.linalg.norm(tn))
                vn = sc.add_vn(tn / tl) if (has_n and tl > 0) else -1
                new.append(Face([(a_[1], a_[2], vn), (b_[1], b_[2], vn), (ci, cvt, vn)], ref.mat, ref.obj, ref.grp, ""))
        closed += 1
    sc.bent_holes = bent
    return new, closed, uneven


def _fill_with_rings(sc, rloop, Pr, cen, n, cv, ref, has_n):
    """Baut die gewölbte Füllung: Randring -> zwei Zwischenringe -> Mitte, Höhen aus dem gefitteten Paraboloid."""
    m = len(rloop)
    arr = _rim_uv(sc, rloop)
    have_uv = arr is not None
    if have_uv:
        uvc = arr.mean(axis=0)
    diam = float(np.linalg.norm(Pr.max(axis=0) - Pr.min(axis=0)))
    uv_map = None
    if have_uv:                                                   # UV aus der Lage: affine Abbildung Lage -> UV, am Rand gefittet; ohne glatten Rand bleibt es bei linear
        B = np.hstack([(Pr - cen) / diam, np.ones((m, 1))])
        coef_uv, *_ = np.linalg.lstsq(B, arr, rcond=None)
        span = float(np.ptp(arr, axis=0).max()) or 1.0
        if np.linalg.matrix_rank(B) == 4 and float(np.abs(B @ coef_uv - arr).max()) <= UV_FIT * span:
            uv_map = lambda p: np.append((p - cen) / diam, 1.0) @ coef_uv
    rings = [[(x[1], x[2]) for x in rloop]]
    for f in CURVED_RINGS:
        ring = []
        for k in range(m):
            a, b = f * cv["uv"][k, 0], f * cv["uv"][k, 1]
            pos = cen + a * cv["u"] + b * cv["v"] + float(cv["h"](a, b)) * n
            uvk = (uv_map(pos) if uv_map else uvc + f * (arr[k] - uvc)) if have_uv else None
            ring.append((sc.add_vertex(pos), sc.add_vt(uvk) if have_uv else -1))
        rings.append(ring)
    cpos = cen + float(cv["h"](0.0, 0.0)) * n
    centre = (sc.add_vertex(cpos), sc.add_vt(uv_map(cpos) if uv_map else uvc) if have_uv else -1)
    out = []

    def tri(a, b, c):
        tn = np.cross(sc.V[b[0]] - sc.V[a[0]], sc.V[c[0]] - sc.V[a[0]])
        tl = float(np.linalg.norm(tn))
        vn = sc.add_vn(tn / tl) if (has_n and tl > 0) else -1
        out.append(Face([(a[0], a[1], vn), (b[0], b[1], vn), (c[0], c[1], vn)], ref.mat, ref.obj, ref.grp, ""))
    for r in range(len(rings) - 1):
        for k in range(m):
            k2 = (k + 1) % m
            tri(rings[r][k], rings[r][k2], rings[r + 1][k2])
            tri(rings[r][k], rings[r + 1][k2], rings[r + 1][k])
    last = rings[-1]
    for k in range(m):
        tri(last[k], last[(k + 1) % m], centre)
    return out


# ============================================================================= Grundkoerper
def _ellipse_perimeter(a, b):
    h = ((a - b) / (a + b)) ** 2 if a + b > 0 else 0.0
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def _shape_areas(ext):
    a, b, c = ext
    box = 2 * (a * b + b * c + a * c)
    out = {"box": box}
    p = 1.6075
    out["sphere"] = 4 * math.pi * (((a / 2) ** p * (b / 2) ** p + (a / 2) ** p * (c / 2) ** p + (b / 2) ** p * (c / 2) ** p) / 3) ** (1 / p)
    best = None
    for ax in range(3):
        o = [x for k, x in enumerate(ext) if k != ax]
        area = _ellipse_perimeter(o[0] / 2, o[1] / 2) * ext[ax] + 2 * math.pi * (o[0] / 2) * (o[1] / 2)
        if best is None or area < best[0]:
            best = (area, ax)
    out["cylinder"] = best[0]
    return out


def _extent(d):
    """Ausdehnung je Achse, nach unten relativ zur groessten begrenzt (kein fester Wert in Einheiten: das Netz kann in Nanometern oder Kilometern sein)."""
    d = np.asarray(d, dtype=np.float64)
    m = float(d.max())
    return np.maximum(d, 1e-9 * m if m > 0 else 1.0)


def _min_half(half):
    """Halbe Ausdehnung des Ersatzes: ein flaches Element ergibt keinen Ersatz mit Dicke null (mindestens 2,5 % der groessten Achse)."""
    m = float(half.max())
    return np.maximum(half, 0.025 * m if m > 0 else 1.0)


def choose_shape(region, shape="auto", axis="auto"):
    lo, hi = np.array(region["bbox"]["min"]), np.array(region["bbox"]["max"])
    ext = _extent(hi - lo)
    areas = _shape_areas(ext)
    if shape in (None, "auto"):
        shape = min(areas, key=lambda k: abs(math.log(areas[k] / max(region["area"], 1e-300))))
    if shape not in ("box", "sphere", "cylinder"):
        fail("Form '%s' gibt es nicht - moeglich: auto, box, sphere, cylinder." % str(shape)[:20])
    if axis in (None, "auto"):
        # Achse mit dem kleinsten Querschnittsunterschied: die Achse, deren Ausdehnung sich am staerksten von den beiden anderen unterscheidet
        dev = [abs(math.log(ext[k] / math.sqrt(ext[(k + 1) % 3] * ext[(k + 2) % 3]))) for k in range(3)]
        ax = int(np.argmax(dev))
    elif axis in ("x", "y", "z"):
        ax = "xyz".index(axis)
    else:
        fail("Achse '%s' gibt es nicht - moeglich: auto, x, y, z." % str(axis)[:20])
    return shape, ax


def _quality(P):
    """Dreiecksgüte 0..1 (1 = gleichseitig, 0 = flach)."""
    a, b, c = P
    e2 = float(((b - a) ** 2).sum() + ((c - b) ** 2).sum() + ((a - c) ** 2).sum())
    return 0.0 if e2 <= 0 else min(1.0, 4 * math.sqrt(3) * float(np.linalg.norm(np.cross(b - a, c - a))) / 2 / e2)


SCORE_SAMPLES = 300
SLIVER = 0.05                                                    # Dreiecksgüte, unter der ein Ohrenschnitt-Deckel als Splitter gilt
SCORE_TRIANGLES = 4000


def rank_shapes(sc, region, scale=1.0, axis="auto"):
    """Rechnet alle Kandidaten (Quader, Kugel, Zylinder je Achse), vergleicht jeden mit dem Original (Abstand der Eckpunkte in beide
    Richtungen, relativ zur Diagonale) und sortiert: das beste zuerst. Rueckgabe: Liste (Wert, Form, Achse)."""
    lo, hi = np.array(region["bbox"]["min"]), np.array(region["bbox"]["max"])
    centre = (lo + hi) / 2
    half = (hi - lo) / 2 * float(scale)
    half = _min_half(half)
    diag = float(np.linalg.norm(2 * half)) or 1.0
    T, _ = sc.triangles(region["face_ids"])
    if len(T) > SCORE_TRIANGLES:
        T = T[np.linspace(0, len(T) - 1, SCORE_TRIANGLES).astype(np.int64)]
    rv = np.unique(T)
    if len(rv) > SCORE_SAMPLES:
        rv = rv[np.linspace(0, len(rv) - 1, SCORE_SAMPLES).astype(np.int64)]
    out = []
    cands = [("box", 0), ("sphere", 0)] + [("cylinder", a) for a in range(3) if axis in (None, "auto") or "xyz".index(axis) == a]
    for kind, ax in cands:
        tmp = Scene()
        faces = make_primitive(tmp, kind, ax, centre - half, centre + half, None, (0, 0, 1, 1), False, False, "")
        PT = np.array([[c[0] for c in f.c][:3] for f in faces], dtype=np.int64)
        pv = np.unique(PT)
        if len(pv) > SCORE_SAMPLES:
            pv = pv[np.linspace(0, len(pv) - 1, SCORE_SAMPLES).astype(np.int64)]
        d1 = mc.dist_to_mesh(sc.V[rv], tmp.V, PT)                 # Original -> Kandidat
        d2 = mc.dist_to_mesh(tmp.V[pv], sc.V, T)                  # Kandidat -> Original
        out.append(((float(d1.mean()) + float(d2.mean())) / 2 / diag + 0.5 * max(float(d1.max()), float(d2.max())) / diag, kind, ax))
    return sorted(out, key=lambda x: x[0])


def make_primitive(sc, shape, axis, lo, hi, material, uv_box, with_normals, with_uv, smooth_name, segments=32, obj="", grp=""):
    """Fuegt einen Grundkoerper in die Szene ein (Eckpunkte, UV, Normalen) und gibt seine Flaechen zurueck."""
    c, ext = (lo + hi) / 2, _extent(hi - lo)
    faces = []
    u0, v0, u1, v1 = uv_box

    def uv(a, b):
        return sc.add_vt((u0 + (u1 - u0) * a, v0 + (v1 - v0) * b)) if with_uv else -1

    def nrm(n):
        n = np.asarray(n, dtype=np.float64)
        ln = float(np.linalg.norm(n))
        return sc.add_vn(n / ln) if (with_normals and ln > 0) else -1

    def F(corners, smooth):
        faces.append(Face(corners, material, obj, grp, smooth))
    if shape == "box":
        idx = {}
        for sx in (0, 1):
            for sy in (0, 1):
                for sz in (0, 1):
                    idx[(sx, sy, sz)] = sc.add_vertex([float(hi[0] if sx else lo[0]), float(hi[1] if sy else lo[1]), float(hi[2] if sz else lo[2])])
        quads = [((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]), ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
                 ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]), ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
                 ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]), ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)])]
        for n, q in quads:
            vn = nrm(n)
            t = [uv(0, 0), uv(1, 0), uv(1, 1), uv(0, 1)]
            for a, b, d in ((0, 1, 2), (0, 2, 3)):
                F([(idx[q[k]], t[k], vn) for k in (a, b, d)], "")
        return faces
    ax = axis
    a1, a2 = (ax + 1) % 3, (ax + 2) % 3
    r1, r2, h = ext[a1] / 2, ext[a2] / 2, ext[ax]

    def pos(ang, t):
        p = np.zeros(3)
        p[a1], p[a2], p[ax] = c[a1] + r1 * math.cos(ang), c[a2] + r2 * math.sin(ang), lo[ax] + h * t
        return p

    def radial(ang):
        n = np.zeros(3)
        n[a1], n[a2] = math.cos(ang) / r1, math.sin(ang) / r2
        return n
    n = segments
    if shape == "cylinder":
        ring = [[sc.add_vertex(pos(2 * math.pi * i / n, t)) for i in range(n)] for t in (0, 1)]
        vns = [nrm(radial(2 * math.pi * i / n)) for i in range(n)]
        up, down = np.zeros(3), np.zeros(3)
        up[ax], down[ax] = 1, -1
        vnu, vnd = nrm(up), nrm(down)
        us = [uv(i / n, 0) for i in range(n + 1)]
        ut = [uv(i / n, 1) for i in range(n + 1)]
        for i in range(n):
            j = (i + 1) % n
            F([(ring[0][i], us[i], vns[i]), (ring[0][j], us[i + 1], vns[j]), (ring[1][j], ut[i + 1], vns[j])], smooth_name)
            F([(ring[0][i], us[i], vns[i]), (ring[1][j], ut[i + 1], vns[j]), (ring[1][i], ut[i], vns[i])], smooth_name)
        for t, vn, rev in ((1, vnu, False), (0, vnd, True)):
            cv = sc.add_vertex(np.where(np.arange(3) == ax, lo[ax] + h * t, c))
            cuv = uv(0.5, 0.5)
            for i in range(n):
                j = (i + 1) % n
                a, b = (ring[t][i], ring[t][j]) if not rev else (ring[t][j], ring[t][i])
                ua = uv(0.5 + 0.5 * math.cos(2 * math.pi * (i if not rev else j) / n), 0.5 + 0.5 * math.sin(2 * math.pi * (i if not rev else j) / n))
                ub = uv(0.5 + 0.5 * math.cos(2 * math.pi * (j if not rev else i) / n), 0.5 + 0.5 * math.sin(2 * math.pi * (j if not rev else i) / n))
                F([(a, ua, vn), (b, ub, vn), (cv, cuv, vn)], "")
        return faces
    # Kugel / Ellipsoid: Ringe von Pol zu Pol
    rings = max(8, n // 2)
    south = sc.add_vertex(np.where(np.arange(3) == ax, lo[ax], c))
    north = sc.add_vertex(np.where(np.arange(3) == ax, hi[ax], c))
    grid = []
    for k in range(1, rings):
        t = k / rings
        phi = math.pi * t
        row = []
        for i in range(n):
            ang = 2 * math.pi * i / n
            p = np.zeros(3)
            p[a1], p[a2], p[ax] = c[a1] + r1 * math.sin(phi) * math.cos(ang), c[a2] + r2 * math.sin(phi) * math.sin(ang), c[ax] - (h / 2) * math.cos(phi)
            row.append(sc.add_vertex(p))
        grid.append(row)

    def en(phi, ang):
        v = np.zeros(3)
        v[a1], v[a2], v[ax] = math.sin(phi) * math.cos(ang) / r1, math.sin(phi) * math.sin(ang) / r2, -math.cos(phi) / (h / 2)
        return nrm(v)

    def corner(vi, k, i):
        t = k / rings
        phi = math.pi * t
        return (vi, uv(i / n, t), en(phi, 2 * math.pi * i / n))
    for i in range(n):
        j = (i + 1) % n
        F([(south, uv((i + 0.5) / n, 0), en(0, 0)), (grid[0][j], uv((i + 1) / n, 1 / rings), en(math.pi / rings, 2 * math.pi * j / n)),
           (grid[0][i], uv(i / n, 1 / rings), en(math.pi / rings, 2 * math.pi * i / n))], smooth_name)
        F([(north, uv((i + 0.5) / n, 1), en(math.pi, 0)), (grid[-1][i], uv(i / n, (rings - 1) / rings), en(math.pi * (rings - 1) / rings, 2 * math.pi * i / n)),
           (grid[-1][j], uv((i + 1) / n, (rings - 1) / rings), en(math.pi * (rings - 1) / rings, 2 * math.pi * j / n))], smooth_name)
        for k in range(len(grid) - 1):
            a, b, d, e = grid[k][i], grid[k][j], grid[k + 1][j], grid[k + 1][i]
            ca, cb, cd, ce = corner(a, k + 1, i), corner(b, k + 1, i + 1), corner(d, k + 2, i + 1), corner(e, k + 2, i)
            F([ca, cb, cd], smooth_name)
            F([ca, cd, ce], smooth_name)
    return faces


# ============================================================================= Befehle
def _out_path(input, out):
    src = Path(input)
    dst = Path(out) if out else src.with_name(src.stem + "_edit" + src.suffix)
    if dst.exists() and dst.resolve() == src.resolve():
        fail("Die Ausgabe darf nicht die Eingabedatei sein.")
    try:
        mc.detect_format(dst)
    except MeshcutError:
        fail("Die Ausgabedatei braucht eine Endung wie .obj, .stl, .ply, .3mf oder .glb.")
    _check_target(dst)
    return dst


def _check_target(dst):
    """Vor der Berechnung: das Ziel darf kein Ordner oder Link und nicht schreibgeschuetzt sein, sein Ordner muss es geben (wie save_atomic)."""
    dst = Path(dst)
    if dst.exists() and (mc.link_like(dst) or not dst.is_file()):
        fail("%s ist ein Ordner oder Link." % dst)
    if dst.exists() and not os.access(dst, os.W_OK):
        fail("%s ist schreibgeschuetzt." % dst)
    if not dst.resolve().parent.is_dir():
        fail("Der Ordner fuer %s existiert nicht." % dst)


def _boundary_edges(sc, pid):
    return sum(1 for v in edge_counts(sc, pid).values() if v == 1)


def _copy_mtl(src_scene, src_path, dst_path):
    note = []
    names = []
    for entry in src_scene.mtllib:                                   # "mtllib a.mtl b.mtl" nennt zwei Dateien; ein Name mit Leerzeichen bleibt ganz, wenn es ihn gibt
        entry = _plain(entry) if src_scene.escaped else entry
        names += [entry] if _plain_file(Path(src_path).resolve().parent / entry) else entry.split()
    for m in names:
        if Path(m).name != m or m in ("", ".", ".."):                # nur ein Dateiname neben der OBJ: nichts aus der Datei darf Pfade bestimmen
            continue
        s, d = Path(src_path).resolve().parent / m, Path(dst_path).resolve().parent / m
        if _plain_file(s) and not d.exists():
            try:
                shutil.copyfile(s, d)
                note.append(m)
                for t in _mtl_textures(s):
                    ts, td = s.parent / t, d.parent / t
                    if _plain_file(ts) and not td.exists():
                        shutil.copyfile(ts, td)
                        note.append(t)
            except OSError:
                pass
    return note


MTL_MAP_KEYS = ("bump", "disp", "decal", "refl")


def _plain_file(p):
    """Eine gewoehnliche Datei, kein Link (auch nicht auf dem Weg dorthin): so fuehrt kein Link aus dem Eingabeordner hinaus."""
    try:
        return p.is_file() and not mc.link_like(p) and p.resolve().parent == p.parent.resolve()
    except (OSError, ValueError):
        return False


def _mtl_textures(path, limit=1_000_000):
    """Dateinamen der Texturen einer .mtl (map_Kd, bump ...): nur schlichte Namen neben der Datei, nichts mit Pfad."""
    try:
        text, escaped = _decode_text(Path(path).read_bytes()[:limit])
    except OSError:
        return []
    out = []
    for ln in _LINE_BREAK.split(text):
        tok = ln.split()
        if len(tok) >= 2 and (tok[0].lower().startswith("map_") or tok[0].lower() in MTL_MAP_KEYS):
            name = _plain(tok[-1]) if escaped else tok[-1]
            if Path(name).name == name and name not in ("", ".", "..") and "/" not in name and "\\" not in name and name not in out:
                out.append(name)
    return out


def directed_duplicates(sc):
    """Wie oft dieselbe gerichtete Kante (von Position a nach Position b) mehrfach vorkommt: Flaechen, die sich gegenlaeufig ueberlappen
    oder falsch herum liegen, erzeugen solche Doppel. Ein sauberes, ueberall gleich orientiertes Netz hat keine."""
    pid = position_ids(sc)
    seen, dup = set(), 0
    for f in sc.faces:
        c = f.c
        for k in range(len(c)):
            a, b = int(pid[c[k][0]]), int(pid[c[(k + 1) % len(c)][0]])
            if a == b:
                continue
            if (a, b) in seen:
                dup += 1
            seen.add((a, b))
    return dup


GEOMETRY_TOLERANCE = 1e-4                                          # relativ; STL und GLB speichern float32
FLOAT32_EPS = 1.2e-7                                               # Rundung einer float32-Koordinate, relativ zu ihrem Betrag (Modelle fern vom Ursprung)


def _check_geometry(sc_new, back, dst):
    """Nicht-OBJ-Formate: Flaeche (aus den Polygonen der Szene, unabhaengig von der Dreieckszerlegung) und Rahmen der geschriebenen Datei muessen
    zur Szene passen. Faengt verschobene, skalierte oder falsch zerlegte Ausgaben, auch wo kein Text zu vergleichen ist."""
    want_area = float(np.linalg.norm(_fan_normals(sc_new, range(len(sc_new.faces))), axis=1).sum() / 2)
    T, _ = back.triangles()
    got_area = float(np.linalg.norm(np.cross(back.V[T[:, 1]] - back.V[T[:, 0]], back.V[T[:, 2]] - back.V[T[:, 0]]), axis=1).sum() / 2)
    used = np.unique([c[0] for f in sc_new.faces for c in f.c])
    lo, hi = sc_new.V[used].min(axis=0), sc_new.V[used].max(axis=0)
    blo, bhi = back.V.min(axis=0), back.V.max(axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    far = float(max(np.abs(lo).max(), np.abs(hi).max()))                       # Abstand zum Ursprung: float32 rundet dort grober
    slack = 8 * FLOAT32_EPS * far / diag
    if abs(got_area - want_area) > (GEOMETRY_TOLERANCE + 4 * slack) * max(want_area, 1e-300) or max(float(np.abs(blo - lo).max()), float(np.abs(bhi - hi).max())) > (GEOMETRY_TOLERANCE + slack) * diag:
        fail("Selbstpruefung fehlgeschlagen: Flaeche oder Ausdehnung der geschriebenen Datei weichen von der Szene ab (%s)." % dst)


def _check_orientation(src_scene, back, dst):
    if directed_duplicates(back) > directed_duplicates(src_scene):
        fail("Selbstpruefung fehlgeschlagen: die Fuellung liegt teilweise falsch herum oder ueberlappt sich (%s). Ein kleineres Element waehlen." % dst)


def _verify_written(sc_new, src_scene, kept_ids, dst, fmt):
    """Liest die geschriebene Datei zurueck. OBJ: behaltene Flaechen am Anfang gleich (Eckpunkt, UV, Normale, Material), neue Flaechen mit dem
    beabsichtigten Material und endlichen Koordinaten. Andere Formate: Dreieckszahl und endliche Koordinaten. Rueckgabe: (unveraenderte, neue Dreiecke)."""
    n_kept = src_scene.n_triangles(kept_ids)
    n_new = sc_new.n_triangles(range(len(kept_ids), len(sc_new.faces)))
    if fmt != "obj":
        back = read_scene(dst)
        if not np.isfinite(back.V).all() or back.n_triangles() != n_kept + n_new:
            fail("Selbstpruefung fehlgeschlagen: die geschriebene Datei hat nicht die erwarteten Dreiecke (%s)." % dst)
        _check_geometry(sc_new, back, dst)
        _check_orientation(src_scene, back, dst)
        return None
    back = parse_obj(Path(dst).read_bytes())
    want = [src_scene.signature(src_scene.faces[i]) for i in kept_ids]
    got = [back.signature(f) for f in back.faces[:len(kept_ids)]]
    if got != want or len(back.faces) != len(sc_new.faces):
        fail("Selbstpruefung fehlgeschlagen: die unberuehrten Flaechen weichen im geschriebenen Ergebnis vom Original ab (%s)." % dst)
    fresh_back, fresh = back.faces[len(kept_ids):], sc_new.faces[len(kept_ids):]
    if not np.isfinite(back.V).all() or [back.raw_name(f.mat) for f in fresh_back] != [sc_new.raw_name(f.mat) for f in fresh] or sum(len(f.c) - 2 for f in fresh_back) != n_new:
        fail("Selbstpruefung fehlgeschlagen: die neuen Flaechen sind in der Datei nicht wie vorgesehen (Material, Zahlen) (%s)." % dst)
    if not all(_text_finite(back.vt[c[1]]) for f in fresh_back for c in f.c if c[1] >= 0) or not all(
            _text_finite(back.vn[c[2]]) for f in fresh_back for c in f.c if c[2] >= 0):
        fail("Selbstpruefung fehlgeschlagen: die neuen Flaechen haben unendliche oder ungueltige UV-/Normalenwerte, meist weil die Eingabe sie am Lochrand schon hat (%s)." % dst)
    _check_orientation(src_scene, back, dst)
    return n_kept, n_new


def _text_finite(s):
    try:
        return all(math.isfinite(float(t)) for t in s.split())
    except ValueError:
        return False


def _write_and_check(sc_new, src_scene, kept_ids, dst):
    """Schreibt das Ergebnis neben das Ziel, prueft es durch Zuruecklesen (siehe _verify_written) und setzt es erst danach an die Stelle des Ziels:
    ein Fehlschlag laesst ein frueheres gutes Ergebnis unberuehrt (und hinterlaesst nichts)."""
    dst = Path(dst)
    fmt = mc.detect_format(dst)
    _check_target(dst)
    stage = dst.with_name(".meshcut-%s%s" % (uuid.uuid4().hex[:10], dst.suffix))       # kurz und einmalig: keine fremde Datei, keine Kollision, keine Pfadlaenge
    mc.save_atomic(scene_bytes(sc_new, fmt), stage)
    try:
        report = _verify_written(sc_new, src_scene, kept_ids, stage, fmt)
        try:
            os.replace(stage, dst)
        except OSError:
            fail("%s ist schreibgeschuetzt oder in Benutzung." % dst)
        return report
    except BaseException:
        try:
            stage.unlink()
        except OSError:
            pass
        raise


def _raw_material(sc, region):
    """Das vorherrschende Material des Elements, so geschrieben wie in der Szene (die Liste der Elemente zeigt es als Anzeigename)."""
    if not region["materials"]:
        return None
    want = region["materials"][0]
    for i in region["face_ids"]:
        m = sc.faces[i].mat
        if m and ((_plain(m) if sc.escaped else m) == want):
            return m
    return want


def edit_scene(sc, region, mode, shape="auto", axis="auto", scale=1.0, material=None, close=True):
    """Gemeinsamer Kern von remove und swap. Rueckgabe: (neue Szene, behaltene Flaechennummern, Bericht)."""
    pid = position_ids(sc)
    removed = region["face_ids"]
    rem = set(removed)
    kept = [i for i in range(len(sc.faces)) if i not in rem]
    if not kept:
        fail("Danach bliebe nichts uebrig - das waere das ganze Netz.")
    out = sc.copy()
    open_before = _boundary_edges(sc, pid)
    new_faces, holes, uneven, prim_note, hints = [], 0, 0, "", []
    lo = np.array(region["bbox"]["min"])
    hi = np.array(region["bbox"]["max"])
    prim = None
    ranking = []
    if mode == "swap":
        if shape in (None, "auto"):                                  # Pruefer: mehrere Kandidaten rechnen, selbst bewerten, den besten nehmen
            ranking = rank_shapes(sc, region, 1.0, axis)                   # die Form wird am Original gewaehlt; --scale gilt erst fuer den Ersatz
            kind, ax = ranking[0][1], ranking[0][2]
        else:
            kind, ax = choose_shape(region, shape, axis)
        centre = (lo + hi) / 2
        half = (hi - lo) / 2 * float(scale)
        half = _min_half(half)                                           # ein flaches Element (z. B. eine Platte) ergibt keinen Ersatz mit Dicke null
        plo, phi_ = centre - half, centre + half
        mat = _typed(sc, material) if material else _raw_material(sc, region)
        uvs = [u for i in removed for c in sc.faces[i].c if c[1] >= 0 for u in [_uv_of(sc.vt[c[1]])] if u is not None]
        if uvs:
            arr = np.array(uvs)
            uvbox = (float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max()))
        else:
            uvbox = (0.0, 0.0, 1.0, 1.0)
        prim = (kind, ax, plo, phi_, mat, uvbox)
    if close:                                                    # auch beim Ersetzen: erst das Loch zu, dann sitzt der Ersatz darauf
        extra, holes, uneven = close_holes(out, pid, removed)
        new_faces += extra
    if prim is not None:
        kind, ax, plo, phi_, mat, uvbox = prim
        smooth = "1" if kind != "box" else ""
        first = sc.faces[removed[0]]
        new_faces += make_primitive(out, kind, ax, plo, phi_, mat, uvbox, out.has_normals(), out.has_uv(), smooth, obj=first.obj, grp=first.grp)
        prim_note = "%s (%s)" % ({"box": "Quader", "sphere": "Kugel", "cylinder": "Zylinder"}[kind], "xyz"[ax] if kind == "cylinder" else "ohne Achse")
    if uneven:
        hints.append("%d Loch/Loecher mit nicht ebenem Rand wurden flach gefuellt (Ohrenschnitt aus Randpunkten): die Fuellung folgt der Woelbung der Umgebung nicht." % uneven)
    if getattr(out, "bent_holes", 0):
        hints.append("%d Loch/Loecher liegen in einer gewoelbten Umgebung, fuer die keine passende Woelbung gefunden wurde (keine Kugel): sie wurden flach gefuellt, das Volumen schrumpft." % out.bent_holes)
    if prim is not None and prim[4] and prim[4] not in {f.mat for f in sc.faces}:
        shown_m = _plain(prim[4]) if sc.escaped else prim[4]
        hints.append("Das Material '%s' kommt in der Datei sonst nicht vor (Tippfehler?): es muss in der .mtl stehen, sonst bleibt der Ersatz ohne Material." % shown_m[:40])
    if ranking and ranking[0][0] > BAD_SHAPE:
        hints.append("Kein Kandidat passt gut (beste Abweichung %.2f der Diagonale): der Ersatz ist nur eine grobe Naeherung." % ranking[0][0])
    out.faces = [sc.faces[i] for i in kept] + new_faces
    open_after = _boundary_edges(out, position_ids(out))
    report = {"entfernt": len(removed), "entfernte_dreiecke": int(sc.n_triangles(removed)), "neu": len(new_faces),
              "neue_dreiecke": int(sum(len(f.c) - 2 for f in new_faces)), "loecher_geschlossen": holes,
              "ersatz": prim_note, "kandidaten": [(round(v, 4), k, a) for v, k, a in ranking], "offene_kanten_vorher": open_before, "offene_kanten_nachher": open_after, "hinweise": hints}
    return out, kept, report


def _check_numbers(a):
    sc = getattr(a, "scale", None)
    if sc is not None and not (isinstance(sc, (int, float)) and not isinstance(sc, bool) and math.isfinite(sc) and SCALE_RANGE[0] <= sc <= SCALE_RANGE[1]):
        fail("--scale muss eine Zahl zwischen %g und %g sein." % SCALE_RANGE)
    sz = getattr(a, "size", None)
    if sz is not None and not (isinstance(sz, int) and not isinstance(sz, bool) and SIZE_RANGE[0] <= sz <= SIZE_RANGE[1]):
        fail("--size muss eine ganze Zahl zwischen %d und %d sein." % SIZE_RANGE)


def _check_material(m):
    if m is not None and m.strip(" \t").lower() == "off":
        fail("Material 'off' ist in OBJ der Wert fuer 'kein Material'; einen anderen Namen waehlen (oder --material weglassen, wenn der Ersatz keines braucht).")
    if m is not None and (not m.strip(" \t") or m != m.strip(" \t") or len(m) > 128
                          or any(ord(ch) < 32 or ord(ch) == 127 or ch in "\u0085\u2028\u2029" for ch in m)):
        fail("Material: ein Name ohne Zeilenumbruch, Steuerzeichen und Leerzeichen am Rand, hoechstens 128 Zeichen.")


def _typed(sc, text):
    """Ein getippter Name, so wie ihn die Szene schreibt: in einer Szene mit Nicht-UTF-8-Bytes werden Zeichen bis U+00FF wieder zu einem einzelnen Byte
    (sonst stuende neben dem Latin-1-Material der Datei ein UTF-8-Name, der zur .mtl nicht passt)."""
    if text is None or not sc.escaped:
        return text
    return "".join(chr(ESCAPE_BASE + ord(c)) if 0x80 <= ord(c) <= 0xFF else (ESCAPE_MARK + c if 0xF700 <= ord(c) <= 0xF7FF else c) for c in text)


def _run_edit(a, mode):
    _check_material(getattr(a, "material", None))
    _check_numbers(a)
    sc = read_scene(a.input)
    regions = find_regions(sc)
    r = choose_region(sc, regions, a.region, a.at, a.name)
    dst = _out_path(a.input, a.out)
    new, kept, rep = edit_scene(sc, r, mode, getattr(a, "shape", "auto"), getattr(a, "axis", "auto"), getattr(a, "scale", 1.0),
                                getattr(a, "material", None))
    same = _write_and_check(new, sc, kept, dst)
    mtl = _copy_mtl(sc, a.input, dst) if mc.detect_format(dst) == "obj" else []
    total = sc.n_triangles()
    mc.say("%s %d (%s, %d Dreiecke) %s." % ("Entfernt: Region" if mode == "remove" else "Ersetzt: Region", r["id"], r["name"], rep["entfernte_dreiecke"],
                                            "entfernt" if mode == "remove" else "durch " + rep["ersatz"]))
    if rep["kandidaten"]:
        names = {"box": "Quader", "sphere": "Kugel", "cylinder": "Zylinder"}
        mc.say("Kandidaten verglichen (Abweichung zum Original, kleiner ist besser): " + ", ".join(
            "%s%s %.3f" % (names[k], " " + "xyz"[a] if k == "cylinder" else "", v) for v, k, a in rep["kandidaten"]))
    if rep["loecher_geschlossen"]:
        mc.say("Loch geschlossen (%d Rand%s), Material und UV der Umgebung uebernommen." % (rep["loecher_geschlossen"], "" if rep["loecher_geschlossen"] == 1 else "er"))
    if same is not None:
        mc.say("Pruefung (aus der geschriebenen Datei): von %d Dreiecken %d entfernt, %d neu dazugekommen; die uebrigen %d sind unveraendert (Eckpunkt, UV, Normale, Material verglichen)."
               % (total, total - same[0], same[1], same[0]))
    else:
        mc.say("Pruefung: von %d Dreiecken %d entfernt, %d neu dazugekommen; %s wird hier nur mit Geometrie geschrieben - Farben, Materialien und UV der Eingabe gehen verloren." % (
            total, rep["entfernte_dreiecke"], rep["neue_dreiecke"], mc.detect_format(dst).upper()))
    for h in rep["hinweise"]:
        mc.say("Hinweis: " + h)
    if rep["offene_kanten_vorher"] != rep["offene_kanten_nachher"]:
        mc.say("Hinweis: offene Kanten vorher %d, nachher %d." % (rep["offene_kanten_vorher"], rep["offene_kanten_nachher"]))
    if sc.dropped:
        mc.say("Hinweis: %d Linien/Punkte der Eingabe werden nicht uebernommen." % sc.dropped)
    if mtl:
        mc.say("Materialdatei mitkopiert: %s" % ", ".join(mtl))
    mc.say("Geschrieben: %s" % dst)


def cmd_regions(a):
    _check_numbers(a)
    sc = read_scene(a.input)
    regions = find_regions(sc)
    out = Path(a.outdir)
    if out.exists() and (mc.link_like(out) or not out.is_dir()):
        fail("%s ist ein Link oder kein Ordner." % out)
    if not out.exists():
        if not out.resolve().parent.is_dir():
            fail("Der Ordner fuer %s existiert nicht." % out)
        out.mkdir()
    for f in ("regions.png", "regions.json"):
        p = out / f
        if p.exists() and (mc.link_like(p) or not p.is_file()):
            fail("%s ist ein Link oder Ordner." % p)
    png, visible = render_overview(sc, regions, a.size)
    for r in regions:
        r["visible"] = visible[r["id"]]
    js = {"input": Path(a.input).name, "triangles": sc.n_triangles(), "has_uv": sc.has_uv(), "has_normals": sc.has_normals(),
          "kinds": list(KINDS), "labeled": min(len(regions), MAX_LABELS), "omitted": getattr(find_regions, "omitted", 0),
          "regions": [{k: v for k, v in r.items() if k != "face_ids"} for r in regions]}
    mc.save_atomic(png, out / "regions.png")
    mc.save_atomic(json.dumps(js, indent=1).encode("utf-8"), out / "regions.json")
    if not regions:
        mc.say("Das Netz ist ein einziges, einheitliches Stueck - nichts zu nummerieren.")
    for r in regions:
        mc.say("%3d  %-8s %-24s %7d Dreiecke  Punkt %s" % (r["id"], r["kind"], r["name"][:24], r["triangles"], ",".join("%.6g" % x for x in r["point"])))
    if getattr(find_regions, "omitted", 0):
        mc.say("Hinweis: %d weitere, kleinere Elemente sind nicht nummeriert (hoechstens %d)." % (find_regions.omitted, MAX_REGIONS))
    if len(regions) > MAX_LABELS:
        mc.say("Hinweis: im Bild stehen nur die ersten %d Nummern." % MAX_LABELS)
    mc.say("Geschrieben: %s, %s" % (out / "regions.png", out / "regions.json"))


def cmd_remove(a):
    _run_edit(a, "remove")


def cmd_swap(a):
    _run_edit(a, "swap")


def register(sub):
    """Haengt die Befehle an den Parser von kkmeshcut (siehe build_parser)."""
    s = sub.add_parser("regions", help="Elemente eines Netzes nummerieren (Uebersichtsbild + JSON), damit ein Assistent 'Teil 7' waehlen kann")
    s.add_argument("input"); s.add_argument("outdir")
    s.add_argument("--size", type=int, default=720, help="Kantenlaenge des Bildes in Pixeln")
    s.set_defaults(fn=cmd_regions)

    def selection(s):
        s.add_argument("input")
        s.add_argument("--region", help="Nummer aus regions.json")
        s.add_argument("--at", help="Punkt x,y,z auf dem Element (das kleinste Element dort wird gewaehlt)")
        s.add_argument("--name", help="Name eines Objekts, einer Gruppe oder eines Materials")
        s.add_argument("--out", help="Ausgabedatei (Vorgabe: <Eingabe>_edit.<Endung>; die Eingabe wird nie ueberschrieben)")

    s = sub.add_parser("remove", help="ein Element entfernen und das Loch sauber schliessen (Material, UV, Normalen der Umgebung bleiben)")
    selection(s)
    s.set_defaults(fn=cmd_remove)

    s = sub.add_parser("swap", help="ein Element durch einen Grundkoerper gleicher Groesse ersetzen (Material und Glaettung automatisch)")
    selection(s)
    s.add_argument("--shape", choices=("auto", "box", "sphere", "cylinder"), default="auto", help="Form des Ersatzes (auto: waehlt nach der Oberflaeche des Originals)")
    s.add_argument("--axis", choices=("auto", "x", "y", "z"), default="auto", help="Achse des Zylinders (auto: aus den Massen des Originals)")
    s.add_argument("--scale", type=float, default=1.0, help="Ersatz um diesen Faktor vergroessern oder verkleinern (Mitte bleibt)")
    s.add_argument("--material", help="Material des Ersatzes (Vorgabe: das Material des Originals)")
    s.set_defaults(fn=cmd_swap)
