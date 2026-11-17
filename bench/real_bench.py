"""Pruefstand mit ECHTEN Fremdmodellen (Schritt 5 des Ein-Prompt-Plans).

Wahrheit = das unveraenderte Original. Aus jedem Modell wird ein zusammenhaengender Fleck gestanzt (Breitensuche von einem festen Startdreieck, 1 %, 3 %
und 6 % der Dreiecke), als Objekt 'patch' markiert und mit `meshcut remove` entfernt. Gemessen wird gegen das Original, nicht gegen die Meinung des Werkzeugs.

Die Modelle werden NICHT mit dem Repo ausgeliefert: sie stammen aus https://github.com/alecjacobson/common-3d-test-models (Herkunft und Lizenz je Modell
stehen dort in der README; fuer Forschung/Tests frei verfuegbar) und werden beim ersten Lauf in --cache geladen.

Schwellen und Ziel stehen VOR dem ersten Lauf fest (bench/OK_SCHWELLEN.md, Abschnitt 'Fremdmodelle') und sind dieselben wie in edit_bench.py:
  gut        = Abweichung der Fuellung <= 2 % der Diagonale UND mittlerer Normalenwinkel <= 30 Grad UND keine zusaetzlichen offenen Kanten gegenueber dem Original
  bestanden  = gut ODER das Werkzeug hat 'unsicher'/'nicht ok' gemeldet (ein gemeldeter Fehler zaehlt, ein verschwiegener nicht)
  Ziel       = >= 85 % 'ok' UND kein verschwiegener Fehler
Aufruf: python bench/real_bench.py [--cache ORDNER] [--write]    (Exit 1, wenn ein Fehler verschwiegen wurde)
"""
import argparse
import json
import math
import sys
import tempfile
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import edit_bench as eb  # noqa: E402
import kkmeshcut as mc  # noqa: E402
import kkmeshcut_edit as ed  # noqa: E402

BASE = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/"
MODELS = ["alligator", "beetle", "cheburashka", "cow", "fandisk", "homer", "rocker-arm", "spot", "suzanne", "teapot", "woody"]
FRACTIONS = (0.01, 0.03, 0.06)
TH_DEV, TH_NORMAL_DEG, TARGET_OK = 0.02, 30.0, 0.85


def fetch(name, cache):
    p = Path(cache) / (name + ".obj")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BASE + name + ".obj", timeout=60) as r:
            p.write_bytes(r.read())
    return p


def grow_patch(T, pid, n, seed):
    """Zusammenhaengender Fleck aus n Dreiecken (Breitensuche ueber gemeinsame Kanten von einem Startdreieck), der KEINE vorhandene offene Kante
    beruehrt und dessen Rand eine einzige Schleife ist (sonst ist es kein 'Loch in einer Flaeche', sondern ein Stueck vom Rand des Modells).
    Der erste Lauf (real_run1) hatte das nicht beachtet: Flecken am offenen Rand zaehlten als Loecher und das Werkzeug lehnte sie ehrlich ab."""
    edges = {}
    for i, t in enumerate(T):
        for k in range(3):
            a, b = int(pid[t[k]]), int(pid[t[(k + 1) % 3]])
            edges.setdefault((min(a, b), max(a, b)), []).append(i)
    border = {v for (a, b), fl in edges.items() if len(fl) != 2 for v in (a, b)}                # offene oder mehrfach belegte Kanten
    rng = np.random.default_rng(seed)
    for _ in range(400):
        start = int(rng.integers(len(T)))
        seen, order, q = {start}, [start], deque([start])
        while q and len(order) < n:
            i = q.popleft()
            t = T[i]
            for k in range(3):
                a, b = int(pid[t[k]]), int(pid[t[(k + 1) % 3]])
                for j in edges[(min(a, b), max(a, b))]:
                    if j not in seen and len(order) < n:
                        seen.add(j)
                        order.append(j)
                        q.append(j)
        if len(order) < n:
            continue
        verts = {int(pid[x]) for i in order for x in T[i]}
        if verts & border:
            continue
        rim = {}                                                                                   # Randkanten des Flecks: jede Ecke genau zweimal = eine Schleife
        for (a, b), fl in edges.items():
            if sum(1 for j in fl if j in seen) == 1:
                rim[a] = rim.get(a, 0) + 1
                rim[b] = rim.get(b, 0) + 1
        if rim and all(v == 2 for v in rim.values()):
            return set(order)
    return None


def punched_obj(V, T, patch):
    rest = [i for i in range(len(T)) if i not in patch]
    lines = ["v %.9g %.9g %.9g" % tuple(p) for p in V]
    lines.append("o rest")
    lines += ["f %d %d %d" % tuple(int(x) + 1 for x in T[i]) for i in rest]
    lines.append("o patch")
    lines += ["f %d %d %d" % tuple(int(x) + 1 for x in T[i]) for i in sorted(patch)]
    return ("\n".join(lines) + "\n").encode(), len(rest)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def run_case(name, src, frac, work, idx):
    sc0 = ed.read_scene(src)
    T0, _ = sc0.triangles()
    pid0 = ed.position_ids(sc0)
    n = max(8, int(frac * len(T0)))
    patch = grow_patch(T0, pid0, n, 1000 + idx)
    if patch is None:
        return {"modell": name, "anteil": frac, "dreiecke": len(T0), "verdict": "kein Fleck", "grund": "kein Fleck ohne Beruehrung des offenen Rands", "gut": None, "verschwiegen": False}
    data, n_rest = punched_obj(sc0.V, T0, patch)
    p = work / ("%s_%g.obj" % (name, frac))
    p.write_bytes(data)
    out = work / ("%s_%g_out.obj" % (name, frac))
    try:
        r = mc.remove(p, name="patch", out=out)
    except mc.MeshcutError as e:
        return {"modell": name, "anteil": frac, "dreiecke": len(T0), "verdict": "abgelehnt", "grund": str(e)[:100], "gut": None, "verschwiegen": False}
    sc1 = ed.read_scene(out)
    fill = eb.new_triangles(sc1, n_rest)
    diag = float(np.linalg.norm(sc0.V.max(axis=0) - sc0.V.min(axis=0)))
    if not len(fill):
        dev, ang = 0.0, 0.0
    else:
        samp = np.vstack([sc1.V[fill].reshape(-1, 3), sc1.V[fill].mean(axis=1)])
        dev = float(mc.dist_to_mesh(samp, sc0.V, T0).max()) / diag
        cen_t, cen_f = sc0.V[T0].mean(axis=1), sc1.V[fill].mean(axis=1)
        near = np.argmin(((cen_f[:, None, :] - cen_t[None, :, :]) ** 2).sum(axis=2), axis=1)
        cosang = np.einsum("ij,ij->i", eb.tri_normals(sc1.V, fill), eb.tri_normals(sc0.V, T0)[near])
        ang = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))).mean())
    open0 = ed._boundary_edges(sc0, pid0)
    open1 = ed._boundary_edges(sc1, ed.position_ids(sc1))
    good = dev <= TH_DEV and ang <= TH_NORMAL_DEG and open1 <= open0
    return {"modell": name, "anteil": frac, "dreiecke": len(T0), "gestanzt": n, "verdict": r["verdict"], "grund": r["reason"], "abweichung": round(dev, 4),
            "normale_grad": round(ang, 1), "offene_kanten_mehr": open1 - open0, "gut": good, "verschwiegen": r["verdict"] == "ok" and not good}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "meshcut_fremd"))
    ap.add_argument("--write", action="store_true", help="Ergebnis nach bench/REAL_RESULTS.md schreiben")
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for mi, name in enumerate(MODELS):
            src = fetch(name, a.cache)
            for fi, frac in enumerate(FRACTIONS):
                row = run_case(name, src, frac, work, mi * 10 + fi)
                rows.append(row)
                print("%-12s %4.0f%%  %-9s dev %-7s n %-6s %s" % (name, 100 * frac, row["verdict"], row.get("abweichung", "-"), row.get("normale_grad", "-"),
                                                                  ("verschwiegen!" if row["verschwiegen"] else (row["grund"] or ""))[:90]), flush=True)
    n = len(rows)
    ok = sum(1 for r in rows if r["verdict"] == "ok")
    ok_good = sum(1 for r in rows if r["verdict"] == "ok" and r["gut"])
    hidden = [r for r in rows if r["verschwiegen"]]
    passed = n - len(hidden)
    lo, hi = wilson(ok_good, n)
    plo, phi_ = wilson(passed, n)
    text = ("Fremdmodelle: %d Faelle (%d Modelle x %d Fleckgroessen)\n"
            "  'ok' gemeldet und wirklich gut: %d von %d = %.0f %% (95 %%-Intervall %.0f-%.0f %%), Ziel >= %.0f %%\n"
            "  davon 'ok' gemeldet insgesamt: %d; unsicher/nicht ok/abgelehnt: %d\n"
            "  verschwiegene Fehler ('ok', aber nicht gut): %d\n"
            "  bestanden (gut oder gemeldet): %d von %d = %.0f %% (Intervall %.0f-%.0f %%)\n"
            % (n, len(MODELS), len(FRACTIONS), ok_good, n, 100 * ok_good / n, 100 * lo, 100 * hi, 100 * TARGET_OK, ok, n - ok, len(hidden), passed, n, 100 * passed / n,
               100 * plo, 100 * phi_))
    print("\n" + text)
    if a.write:
        lines = ["# Fremdmodelle: Messergebnis (bench/real_bench.py)", "", "```", text.rstrip(), "```", "",
                 "| Modell | Fleck | Dreiecke | Urteil | Abweichung | Normale | gut | Grund |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append("| %s | %.0f %% | %s | %s | %s | %s | %s | %s |" % (r["modell"], 100 * r["anteil"], r["dreiecke"], r["verdict"], r.get("abweichung", "-"),
                                                                       r.get("normale_grad", "-"), {True: "ja", False: "nein", None: "-"}[r["gut"]], (r["grund"] or "")[:110]))
        (Path(__file__).resolve().parent / "REAL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (Path(__file__).resolve().parent / "real_results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 1 if hidden else 0


if __name__ == "__main__":
    sys.exit(main())
