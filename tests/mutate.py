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
    ("serve: Sandbox ohne Pruefung", "p.relative_to(self.root)", "pass", "kkmeshcut_serve.py"),
    ("serve: Schluessel nicht geprueft", 'return got.startswith("Bearer ") and hmac.compare_digest(got[7:].encode(), token.encode())', "return True", "kkmeshcut_serve.py"),
    ("serve: Host nicht geprueft", 'return host in ("127.0.0.1", "localhost", "::1")', "return True", "kkmeshcut_serve.py"),
    ("serve: Content-Type nicht geprueft", '.strip().lower() != "application/json":', '.strip().lower() != "application/json" and False:', "kkmeshcut_serve.py"),
    ("serve: Body-Grenze fehlt", "if n < 0 or n > MAX_BODY:", "if False:", "kkmeshcut_serve.py"),
    ("serve: Timeout fehlt", "H.timeout = timeout", "pass", "kkmeshcut_serve.py"),
    ("serve: fremde Adresse ohne Schluessel", 'if host not in ("127.0.0.1", "localhost", "::1") and not token:', "if False:", "kkmeshcut_serve.py"),
    ("ai: Weiterleitung erlaubt", 'raise urllib.error.HTTPError(req.full_url, code, "Weiterleitungen sind nicht erlaubt", headers, fp)', "return super().redirect_request(req, fp, code, msg, headers, newurl)", "kkmeshcut_ai.py"),
    ("ai: http fuer fremde Hosts", 'if u.scheme != "https" and not (u.scheme == "http" and local):', "if False:", "kkmeshcut_ai.py"),
    ("ai: Kommazahlen in Karte", "if not np.isfinite(lab).all() or (lab != np.floor(lab)).any():", "if False:", "kkmeshcut_ai.py"),
    ("ai: doppelte Namen", "if len(set(low)) != len(low):", "if False:", "kkmeshcut_ai.py"),
    ("ai: Zuordnung nicht vollstaendig geprueft", "if sorted(out) != ids or sorted(out.values()) != sorted(names):", "if False:", "kkmeshcut_ai.py"),
    ("ai: Teil 0 nicht oben", "key.append((-round(float(c[2]), 9),", "key.append((round(float(c[2]), 9),", "kkmeshcut_ai.py"),
    ("ai: stderr durchgereicht", 'fail("Der Segmentierungsbefehl endete mit Code %d - zum Nachsehen den Befehl von Hand ausfuehren." % r.returncode)', 'fail("Code %d: %s" % (r.returncode, r.stderr.strip()))', "kkmeshcut_ai.py"),
    ("ai: Antwort zu gross erlaubt", "if len(body) > MAX_RESPONSE_BYTES:", "if False:", "kkmeshcut_ai.py"),
    ("journal: Eintraege nicht geprueft", "or not all(0 <= i < len(names) and v == names[i] for i, v in moved.items())):", "):"),
    ("journal: Namen nicht geprueft", "or not all(isinstance(n, str) and (PART_NAME.fullmatch(n) or n in MULTI_NAME.values() or n == SIDECAR) for n in names)", ""),
    ("ai: Kindprogramm erbt stdin", "shell=False, stdin=subprocess.DEVNULL)", "shell=False)", "kkmeshcut_ai.py"),
    ('r2 serve: Schluessel nur Praefix', 'hmac.compare_digest(got[7:].encode(), token.encode())', 'got[7:].startswith(token[:3])', "kkmeshcut_serve.py"),
    ('r2 serve: Schluessel nur Laenge', 'hmac.compare_digest(got[7:].encode(), token.encode())', 'len(got[7:]) == len(token)', "kkmeshcut_serve.py"),
    ('r2 serve: GET ohne Schluessel', 'def do_GET(self):\n            if not self._authorized():', 'def do_GET(self):\n            if False:', "kkmeshcut_serve.py"),
    ('r2 serve: Typ bool nicht geprueft', 'if not isinstance(v, bool):', 'if False:', "kkmeshcut_serve.py"),
    ('r2 serve: Protokollversion immer neueste', 'want if want in MCP_VERSIONS else MCP_VERSIONS[0]', 'MCP_VERSIONS[0]', "kkmeshcut_serve.py"),
    ('r2 serve: Werkzeugfehler beendet Dienst', '        except Exception as e:                                   # nie den Dienst beenden', '        except KeyboardInterrupt as e:', "kkmeshcut_serve.py"),
    ('r2 serve: LOCK weg', '    with LOCK:\n        return getattr(tools, name)(args)', '    return getattr(tools, name)(args)', "kkmeshcut_serve.py"),
    ('r2 serve: Nullbyte erlaubt', 'or "\\0" in value:', ':', "kkmeshcut_serve.py"),
    ('r2 serve: chunked erlaubt', 'if self.headers.get("Transfer-Encoding"):', 'if False:', "kkmeshcut_serve.py"),
    ('r2 serve: verschachteltes JSON beendet Anfrage', '            except Exception:                                    # ValueError, UnicodeDecodeError, RecursionError', '            except (ValueError, UnicodeDecodeError):', "kkmeshcut_serve.py"),
    ('r2 serve: MCP-Zeile ohne Schutz', '            except Exception as e:                               # eine kaputte Nachricht beendet nie den Dienst', '            except KeyboardInterrupt as e:', "kkmeshcut_serve.py"),
    ('r2 serve: MCP nicht UTF-8', 'stream.reconfigure(encoding="utf-8", errors="replace"', 'stream.reconfigure(errors="replace"', "kkmeshcut_serve.py"),
    ('r2 ai: http fuer fremde Hosts (Name)', 'local = (u.hostname or "") in ("localhost", "127.0.0.1", "::1")', 'local = True', "kkmeshcut_ai.py"),
    ('r2 ai: Adresse ohne Host', '    if not u.hostname:\n        fail("%s hat keine Adresse." % what)', '    pass', "kkmeshcut_ai.py"),
    ('r2 ai: Teilezahl weicht ab', 'if parts and found != parts:', 'if False:', "kkmeshcut_ai.py"),
    ('r2 ai: nur ein Teil erlaubt', 'if found < 2:', 'if False:', "kkmeshcut_ai.py"),
    ('r2 ai: Antwortgroesse', 'if len(body) > MAX_RESPONSE_BYTES:', 'if False:', "kkmeshcut_ai.py"),
    ('r2 ai: Karte Kommazahlen', '        if not np.isfinite(lab).all() or (lab != np.floor(lab)).any():', '        if False:', "kkmeshcut_ai.py"),
    ('r2 ai: Karte negativ', 'if len(lab) and lab.min() < 0:', 'if False:', "kkmeshcut_ai.py"),
    ('r2 ai: Karte Laenge', 'if lab.shape != (nfaces,):', 'if False:', "kkmeshcut_ai.py"),
    ('r2 ai: kmeans Startpunkt', 'centers = [X[int(np.argmin(((X - mean) ** 2).sum(axis=1)))]]', 'centers = [X[int(np.argmax(((X - mean) ** 2).sum(axis=1)))]]', "kkmeshcut_ai.py"),
    ('r2 ai: Flaechenanteil', 'round(float(area[m].sum() / total), 3)', '0', "kkmeshcut_ai.py"),
    ('r2 ai: Befehl mit Shell', 'shell=False, stdin=subprocess.DEVNULL)', 'shell=True, stdin=subprocess.DEVNULL)', "kkmeshcut_ai.py"),
    ('r2 ai: Befehl ohne Zeitgrenze', 'capture_output=True, text=True, timeout=timeout,', 'capture_output=True, text=True,', "kkmeshcut_ai.py"),
    ('r2 ai: LLM-Schluessel nach fremder Adresse', 'if os.environ.get("MESHCUT_LLM_URL", "").strip():           # ausdruecklich', 'if False:           # ausdruecklich', "kkmeshcut_ai.py"),
    ('r2 ai: components Teilezahl', 'if parts and parts != k:', 'if False:', "kkmeshcut_ai.py"),
    ('r2 kern: PLY Listentyp', 'if t[2] not in PLY_TYPES or t[3] not in PLY_TYPES:', 'if False:'),
    ('r2 kern: PLY negative Anzahl', 'if any(c < 0 for _, c, _ in elements):', 'if False:'),
    ('r2 kern: PLY binaer abgeschnitten', 'if pos + need > len(data):', 'if False:'),
    ('r2 kern: PLY Schnellweg ohne Pruefung', 'if (arr["n"] == n0).all():', 'if True:'),
    ('r2 kern: PLY Index bei kurzen Flaechen ungeprueft', 'if ids and (min(ids) < 0 or max(ids) >= len(V)):', 'if len(ids) >= 3 and (min(ids) < 0 or max(ids) >= len(V)):'),
    ('r2 kern: Karte mit Pickle', 'np.load(p, allow_pickle=False)', 'np.load(p, allow_pickle=True)'),
    ('r2 kern: Dreiecksreihenfolge nicht hergestellt', '    P, S = P[order], S[order]\n    nfaces', '    S = S[order]\n    nfaces'),
    ('r2 kern: Reihenfolge-Hash nicht geprueft (verify)', 'add("reihenfolge", oh == side["source"]["order_hash"],', 'add("reihenfolge", True,'),
    ('r2 kern: Reihenfolge-Hash nicht geprueft (join)', 'if sequence_hash(mesh.V, mesh.F, prec) != side["source"]["order_hash"]:', 'if False:'),
    ('r3 kern: flaechenlose Dreiecke nicht in split_ids', 'return np.nonzero(~whole)[0].astype(np.int64)', 'return np.nonzero(cnt > 1)[0].astype(np.int64)'),
    ('r3 kern: lesende Befehle raeumen sofort', 'READ_RECOVER_AGE = 600.0', 'READ_RECOVER_AGE = 0.0'),
    ('r3 kern: Alter der Reste ignoriert', 'if time.time() - d.stat().st_mtime >= min_age]', 'if True]'),
    ('r3 kern: Format beim Schreiben ungeprueft', '    if fmt not in FORMATS:\n        fail("Format \'%s\' gibt es nicht', '    if False:\n        fail("Format \'%s\' gibt es nicht'),
    ('r3 kern: Quellreihenfolge ungeprueft', 'add("quelle-reihenfolge", sequence_hash(src_mesh.V, src_mesh.F, prec) == side["source"]["order_hash"],', 'add("quelle-reihenfolge", True,'),
    ('r3 kern: order_hash nicht Pflicht', 'and isinstance(src.get("order_hash"), str)):', '):'),
    ('r3 kern: Sequenz-Hash ohne Praezision', 'h.update(b"meshcut-seq-1:" + precision.encode() + b":")', 'h.update(b"meshcut-seq-1:")'),
    ('r3 kern: Besuche nicht gezaehlt', 'self.visits -= 1', 'self.visits -= 0'),
    ('r3 kern: Dreiecke nicht gezaehlt', 'self.tris -= int(n)', 'self.tris -= 0'),
    ('r3 kern: 3MF entpackt unbegrenzt', 'if len(xml) > MAX_XML_BYTES:', 'if False:'),
    ('r3 kern: Polygon unbegrenzt', 'if k > MAX_POLYGON:', 'if False:'),
    ('r3 kern: kein fsync', 'os.fsync(fh.fileno())', 'pass'),
    ('r3 serve: Format nicht geprueft', 'if v not in mc.FORMATS:', 'if False:', "kkmeshcut_serve.py"),
    ('r3 serve: verify-Quelle leer uebersprungen', 'source=None if src is None else self.sb.path(src, "source")', 'source=self.sb.path(src, "source") if src else None', "kkmeshcut_serve.py"),
    ('r3 serve: labels ohne Sandbox', 'opts[key] = self.sb.path(v, "labels") if key == "labels" else v', 'opts[key] = v', "kkmeshcut_serve.py"),
    ('r4 kern: 3MF Eckpunkte je Besuch neu', '        if oid not in parsed:\n            verts, tris = None, None', '        if True:\n            verts, tris = None, None'),
    ('r4 kern: GLB Primitive je Besuch neu', 'if key not in prim_cache:', 'if True:'),
    ('r4 kern: Zwischenordner ohne Marker geraeumt', 'and ((d / MARKER).is_file() or (d / JOURNAL).is_file())]', ']'),
    ('r4 kern: Links beim Aufraeumen gefolgt', 'd.is_dir() and not link_like(d)]', 'd.is_dir()]'),
    ('r4 kern: Hausdorff mit flaechenlosen Quelldreiecken', 'sF = src_mesh.F[_face_areas(src_mesh.V, src_mesh.F) > 0]', 'sF = src_mesh.F'),
    ('r4 kern: leeres Netz durchgelassen', '    if len(mesh.F) == 0:\n        fail("%s enthaelt keine Dreiecke." % Path(path).name)', '    if False:\n        fail("%s enthaelt keine Dreiecke." % Path(path).name)'),
    ('r4 kern: leeres join durchgelassen', '        if len(mesh.F) == 0:\n            fail("Die Eingabedateien enthalten keine Dreiecke.")', '        if False:\n            fail("Die Eingabedateien enthalten keine Dreiecke.")'),
    ('r4 kern: Polygon-Gesamtarbeit ungezaehlt', 'self.left -= k * k', 'self.left -= 0'),
    ('r4 kern: Marker nicht geschrieben', '    (d / MARKER).write_text("meshcut", encoding="utf-8")\n', '    pass\n'),
    ('r4 serve: MCP-Zeilenlaenge unbegrenzt', 'if len(line) > MAX_BODY and not line.endswith(NL):', 'if False:', "kkmeshcut_serve.py"),
    ('r4 serve: Host [::1] ohne Port', 'host = host[1:host.find("]")] if host.startswith("[") and "]" in host else', "host = ((host.rsplit(':', 1)[0].strip('[]')) if True else host) if False else", "kkmeshcut_serve.py"),
]


def run(mutant):
    name, old, new = mutant[:3]
    target = mutant[3] if len(mutant) > 3 else "kkmeshcut.py"                  # ohne Angabe: der Kern
    src = (REPO / target).read_text(encoding="utf-8")
    if old not in src:
        return name, "MUSTER FEHLT", ""
    work = Path(tempfile.mkdtemp(prefix="mut_"))
    try:
        shutil.copytree(REPO / "tests", work / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(REPO / "examples", work / "examples")
        (work / "docs").mkdir()
        mutated = src.replace(old, new, 1)
        for extra in ("kkmeshcut.py", "kkmeshcut_ai.py", "kkmeshcut_serve.py"):
            shutil.copy(REPO / extra, work / extra)
        (work / target).write_text(mutated, encoding="utf-8")
        shutil.copy(work / "kkmeshcut.py", work / "docs" / "kkmeshcut.py")
        (work / "docs" / "index.html").write_text((REPO / "docs" / "index.html").read_text(encoding="utf-8"), encoding="utf-8")
        r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--tb=line", "tests"], cwd=work, capture_output=True, text=True, timeout=1500, creationflags=NO_WINDOW, stdin=subprocess.DEVNULL)
        lines = r.stdout.splitlines()
        tail = [ln for ln in lines if ln.startswith("FAILED") or ln.startswith("ERROR")][:1]
        why = [ln for ln in lines if ln.startswith(("/", "E:", "C:", "D:")) and ": " in ln][:1]            # Zeile aus --tb=line
        if why:
            tail = [(tail[0] if tail else "") + " | " + why[0][-160:]]
        # nur Code 1 heisst: ein Test ist gescheitert. Code 2 und hoeher (Syntaxfehler, Import, keine Tests) ist ein ungueltiger Mutant.
        state = {0: "UEBERLEBT", 1: "GETOETET"}.get(r.returncode, "UNGUELTIG (pytest Code %d)" % r.returncode)
        return name, state, (tail[0][:330] if tail else "")
    except subprocess.TimeoutExpired:
        return name, "ZEITUEBERSCHREITUNG", ""
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
    alive = [t for t, s in zip(todo, results) if s not in ("GETOETET",)]
    print("\n%d Mutanten, %d getoetet, %d ueberlebt/ungueltig" % (len(todo), len(todo) - len(alive), len(alive)))
    for t in alive:
        print("  ->", t[0])
