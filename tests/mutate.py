"""Mutationstest fuer meshcut: jede Aenderung in kkmeshcut.py muss von den Tests bemerkt werden (pytest scheitert).

    python tests/mutate.py                 # alle Mutanten (jeder braucht einen Testlauf, das dauert)
    python tests/mutate.py Rollback Hash   # nur Mutanten, deren Name eines dieser Woerter enthaelt
"""
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
NO_WINDOW = 0x08000000 if os.name == "nt" else 0

MUTANTS = [
    ("kantenpunkt t=0.5", "t = s[klo] / (s[klo] - s[khi])", "t = 0.5 + 0 * s[klo]"),
    ("koplanar vertauscht", "bit[zi] = np.where(nrm @ plane.n > 0, 0, 1)", "bit[zi] = np.where(nrm @ plane.n > 0, 1, 0)"),
    ("snap aus", "snap=1e-12):\n    \"\"\"Zerlegt", "snap=0.0):\n    \"\"\"Zerlegt"),
    ("flaechenlose Dreiecke immer pos", "_settle_flat_faces(F, zi[flat], bit, crossing, st.label)", "bit[zi[flat]] = 1"),
    ("deckel nicht gespiegelt", "cap_f.append((x, z, y)); cap_bit.append(0)", "cap_f.append((x, y, z)); cap_bit.append(0)"),
    ("deckel falsch herum", "tris, emerg = cap_triangles(list(zip(ba.tolist(), bb.tolist())), V2, plane.n, flip=True)", "tris, emerg = cap_triangles(list(zip(ba.tolist(), bb.tolist())), V2, plane.n, flip=False)"),
    ("loecher ignoriert", "poly = bridge_holes(P, ol, owner[i]) if owner[i] else list(ol)", "poly = list(ol)"),
    ("faecher nicht von kleinster nummer", "r0 = poly.index(min(poly))", "r0 = 0"),
    ("ohr: Rest-flach-Pruefung weg", "if remaining > 3 and area_left[0] - t / 2.0 <= 1e-12 * area0:", "if False:"),
    ("hash ohne Umlaufsinn", "rots = [ids, ids[:, [1, 2, 0]], ids[:, [2, 0, 1]]]", "rots = [np.sort(ids, axis=1)]"),
    ("hash ohne Rundung", "Vq = quantize(V, precision)\n        u, inv", "Vq = np.asarray(V, dtype=np.float64)\n        u, inv"),
    ("hash precision ignoriert", "h.update(b\"meshcut-tris-1:\" + precision.encode() + b\":\")", "h.update(b\"meshcut-tris-1:\")"),
    ("join ohne Original-Dreiecke", "P = np.concatenate([P[~is_split], split_coords])", "P = np.concatenate([P[~is_split], P[is_split][:len(split_coords)]])"),
    ("join ohne Vollstaendigkeitspruefung", "if len(S) != nfaces or not (S == np.arange(nfaces)).all():", "if False:"),
    ("join Deckel behalten", "tri_list.append(V[F[:nonc]])", "tri_list.append(V[F]); src = np.concatenate([src, -np.ones(ncap, np.int64)])"),
    ("join Dateiveraenderung nicht erkannt", "if len(F) != meta[\"faces\"]:", "if False:"),
    ("verify volumen ohne Pruefung", "add(\"volumen\", abs(vol_sum - sv) <= rel_v * max(abs(sv), diag * sa * 1e-3, 1e-300),", "add(\"volumen\", True,"),
    ("verify hash ohne Pruefung", "add(\"hash\", jh == side[\"source\"][\"hash\"],", "add(\"hash\", True,"),
    ("wasserdicht nicht geprueft", "add(\"wasserdicht\", all_tight or not has_cap,", "add(\"wasserdicht\", True,"),
    ("offene Kanten nicht gezaehlt", "open_ = int((~has_rev).sum())", "open_ = 0"),
    ("falsch orientierte Kanten nicht gezaehlt", "dup = int((cnt > 1).sum())", "dup = 0"),
    ("negatives Volumen erlaubt", "if vol < 0 and not flip:", "if False:"),
    ("wasserdicht-Pruefung vor Schnitt weg", "    er = edge_report(mesh.F, len(mesh.V))\n    if not er[\"watertight\"]:", "    er = edge_report(mesh.F, len(mesh.V))\n    if False:"),
    ("Fremddatei ueberschreiben", "if (out / n).exists() and n not in owned:", "if False:"),
    ("Rollback fehlt", "        for n in installed:\n            try:\n                (out / n).unlink(missing_ok=True)", "        for n in []:\n            try:\n                (out / n).unlink(missing_ok=True)"),
    ("Seitenwiederherstellung fehlt", "                os.replace(f, out / names[int(f.name)])", "                pass"),
    ("Eingabe im Ausgabeordner erlaubt", "if rx.parent == out.resolve() and (rx.name in owned or rx.name in new_names):", "if False:"),
    ("Link erlaubt", "if out.exists() and (link_like(out) or not out.is_dir()):", "if out.exists() and (False or not out.is_dir()):"),
    ("alte Teile bleiben", "stale = sorted(n for n in owned if n not in allnames and (out / n).is_file())", "stale = []"),
    ("Keil-Tabelle falsch", "if math.sin(phi - k * delta) > 0:", "if math.sin(phi - k * delta) < 0:"),
    ("Gitter Gewichte falsch", "step *= n", "step *= 1"),
    ("Ebene negativ", "d = float(offset) / abs(c)", "d = float(offset) / c"),
    ("pinch nicht gezaehlt", "chi = nv + pinch - er[\"edges\"] + len(F)", "chi = nv - er[\"edges\"] + len(F)"),
    ("labels: Laenge nicht geprueft", "if labels.shape != (len(mesh.F),):", "if False:"),
    ("labels: negativ erlaubt", "if len(labels) and labels.min() < 0:", "if False:"),
    ("max: aufrunden fehlt", "counts = [max(1, math.ceil(ext[i] / sizes[i])) if math.isfinite(sizes[i]) else 1 for i in range(3)]", "counts = [max(1, int(ext[i] / sizes[i])) if math.isfinite(sizes[i]) else 1 for i in range(3)]"),
    ("OBJ Index nicht geprueft", "if min(ids) < 0 or max(ids) >= len(V):", "if False:"),
    ("STL abgeschnitten akzeptiert", "if len(vals) % 3:", "if False:"),
    ("NaN akzeptiert", "if not np.isfinite(P).all():\n            fail(\"%s enthaelt NaN oder unendliche Koordinaten.\" % path)", "if False:\n            fail(\"x\")"),
    ("Sidecar-Dateiname nicht geprueft", "elif not (PART_NAME.fullmatch(n) and n.endswith(\".\" + fmt)):\n            raise ValueError(\"Dateiname\")", "elif False:\n            raise ValueError(\"Dateiname\")"),
    ("Deckel: Nachbarlinks-Regel falsch", "if ang > best:", "if ang < best:"),
    ("API: SystemExit durchgelassen", "raise MeshcutError(str(e.code).removeprefix(\"Fehler: \")) from None", "raise"),
    ("API: stdout nicht umgeleitet", "with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):\n            a = build_parser(_ApiParser)", "if True:\n            a = build_parser(_ApiParser)"),
    ("Hausdorff Richtung eins", "for (Va, ua), (Vb, Fb) in (((V1, u1), (V2, F2)), ((V2, u2), (V1, F1))):", "for (Va, ua), (Vb, Fb) in (((V1, u1), (V2, F2)),):"),
    ("Join ohne Paarentfernung", "                for a, b in zip(pos_i, neg_i):\n                    drop[a] = drop[b] = True", "                pass"),
    ("Ebene: Gewicht Default", "p.weight = 1 << k", "p.weight = 1"),
    ("snap-Skala", "<= snap * extent +", "<= snap * 1e-9 +"),
    ("Eckpunkt-Rotation in Fragment", "if gi * gj < 0:\n                e = emap[min(i, j) * nV + max(i, j)]", "if gi * gj < 0:\n                e = emap[min(i, j) * nV + min(i, j)] if False else emap[min(i, j) * nV + max(i, j)]"),
    ("teile-hash nicht geprueft", 'add("teile", not bad,', 'add("teile", True,'),
    ("ebenen nicht geprueft", 'add("ebenen", *_check_cap_planes(side, parts, prec))', 'add("ebenen", True, "x")'),
    ("flache Fuellstuecke nicht verknuepft", "_alias_flat_edges(V, Fc, Sc, emap, nV)\n", "pass\n"),
    ("deckel nicht zentriert", "X = X - X.mean(axis=0)", "pass"),
    ("Wiederherstellung in prepare fehlt", "recover_outdir(out)\n    owned = set()", "owned = set()"),
    ("Wiederherstellung in load fehlt", "        recover_outdir(folder)\n    p = Path(folder) / SIDECAR", "        pass\n    p = Path(folder) / SIDECAR"),
    ("Journal: done wird zurueckgerollt", 'if j["state"] != "done":', 'if True:'),
    ("Journal nicht geschrieben", '_write_journal(aside, "moving", names, moved)', 'pass'),
    ("join-Ordnerschutz samefile", "if outp.exists() and any((inputs[0] / n).exists() and os.path.samefile(inputs[0] / n, outp) for n in guarded):", "if False:"),
    ("3MF Einheit ignoriert", "scale = THREEMF_UNITS[unit]", "scale = 1.0"),
    ("3MF Einheit unbekannt erlaubt", "if unit not in THREEMF_UNITS:", "if False:"),
    ("GLB Spiegelung", "if np.linalg.det(M[:3, :3]) < 0:", "if False:"),
    ("dec_array Groesse", "if want > MAX_ARRAY_BYTES:", "if False:"),
    ("dec_array Entpackgrenze", 'raw = zo.decompress(base64.b64decode(d["b64"]), want + 1)', 'raw = zo.decompress(base64.b64decode(d["b64"]))'),
    ("dec_array dtype", 'if dt.str not in ("<i8", "<f8") or any(x < 0 for x in shape):', 'if False:'),
    ("OBJ konkav nur Faecher", "    if min(cr) >= 0:", "    if True:"),
    ("Zellen-Limit fehlt", "    if cells > MAX_PARTS:", "    if False:"),
    ("Gitter komprimiert", 'mode="wedges" if a.wedges else ("grid" if (a.grid or a.max) else "plane")', 'mode="wedges" if a.wedges else "plane"'),
    ("doppelte Eckpunkte-Fragmente behalten", "if poly[0] == poly[q] or poly[q] == poly[q + 1] or poly[0] == poly[q + 1]:", "if False:"),
    ("API info nicht gekapselt", "raise MeshcutError((str(e).strip().splitlines() or [type(e).__name__])[0]) from e\n\n\ndef labels", "raise\n\n\ndef labels"),
    ("ASCII-STL gross", "text, re.IGNORECASE)", "text)"),
    ("OBJ Einrueckung", "q = ln.split()\n        if not q:\n            continue", "q = ln.split()\n        if not q or ln[0] in ' \t':\n            continue"),
    ("Ebene als Tupel", "if len(val) == 4 and all(", "if False and all("),
    ("help erlaubt", 'if key in ("help", "h", "version"):', 'if False:'),
    ("Pruefmeldung ohne Namen", 'fail("Pruefung fehlgeschlagen: %s." % ", ".join(c[0] for c in r["checks"] if not c[1]))', 'fail("Pruefung fehlgeschlagen.")'),
    ("Snap ohne Rechenungenauigkeit", "+ 8 * np.finfo(np.float64).eps * mag * (snap > 0)", "+ 0"),
    ("Hausdorff float32-Zuschlag", '+ (8 * np.finfo(np.float32).eps * mag if prec == "float32" else 0.0)', ""),
    ("Ebenen-Toleranz zu weit", "tol = (1e-9 if prec == \"float64\" else 1e-6) * extent + 16 * eps * (mag + max(abs(d) for _, d in planes))", "tol = 1e9"),
    ("GLB Streifen still", "if mode in (5, 6):", "if False:"),
    ("Hinweis nicht wasserdicht", "if cap and not er[\"watertight\"]:", "if False:"),
    ("Namen nicht geprueft", "if not isinstance(nm, dict) or not all(isinstance(k, str)", "if False and not all(isinstance(k, str)"),
    ("Namen doppelt erlaubt", "if len({v.lower() for v in nm.values()}) != len(nm):", "if False:"),
    ("Namen beim Schreiben nicht geprueft", "if res.names and not (set(res.names) <= set(ids)", "if False and not (set(res.names) <= set(ids)"),
    ("Objektnamen nicht gesetzt", '(res.names or {}).get(p.id) or "part_%d" % p.id', '"part_%d" % p.id'),
    ("PLY Index nicht geprueft", "if faces.size and (faces.min() < 0 or faces.max() >= len(V)):", "if False:"),
    ("PLY ascii abgeschnitten", "if n < 0 or ti + 1 + n > len(toks):", "if False:"),
    ("PLY big endian", 'order = "<" if fmt == "binary_little_endian" else ">"', 'order = "<"'),
    ("PLY Format unbekannt", 'if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):', "if False:"),
    ("Zwischenordner bleiben liegen", "        if d not in entries:", "        if False:"),
]


def run(mutant):
    name, old, new = mutant
    src = (REPO / "kkmeshcut.py").read_text(encoding="utf-8")
    if old not in src:
        return name, "MUSTER FEHLT", ""
    work = Path(tempfile.mkdtemp(prefix="mut_"))
    try:
        shutil.copytree(REPO / "tests", work / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(REPO / "examples", work / "examples")
        (work / "docs").mkdir()
        mutated = src.replace(old, new, 1)
        for extra in ("kkmeshcut_ai.py", "kkmeshcut_serve.py"):
            shutil.copy(REPO / extra, work / extra)
        (work / "kkmeshcut.py").write_text(mutated, encoding="utf-8")
        (work / "docs" / "kkmeshcut.py").write_text(mutated, encoding="utf-8")
        (work / "docs" / "index.html").write_text((REPO / "docs" / "index.html").read_text(encoding="utf-8"), encoding="utf-8")
        r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "tests"], cwd=work, capture_output=True, text=True, timeout=1500, creationflags=NO_WINDOW)
        tail = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED") or ln.startswith("ERROR")][:1]
        return name, ("GETOETET" if r.returncode != 0 else "UEBERLEBT"), (tail[0][:110] if tail else "")
    except subprocess.TimeoutExpired:
        return name, "ZEITUEBERSCHREITUNG (zaehlt als getoetet)", ""
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    only = sys.argv[1:]
    todo = [m for m in MUTANTS if not only or any(o in m[0] for o in only)]
    results = []
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        for name, state, info in ex.map(run, todo):
            print("%-40s %-12s %s" % (name, state, info), flush=True)
            results.append(state)
    alive = [t for t, s in zip(todo, results) if s in ("UEBERLEBT", "MUSTER FEHLT")]
    print("\n%d Mutanten, %d getoetet, %d ueberlebt/ungueltig" % (len(todo), len(todo) - len(alive), len(alive)))
    for t in alive:
        print("  ->", t[0])
