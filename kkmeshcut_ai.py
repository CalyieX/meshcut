"""meshcut KI-Schicht: Teile finden und benennen - das Ergebnis ist immer eine Zuordnungskarte Dreieck -> Teil.

Die Karte geht durch denselben Weg wie jede andere (`split_labels`): Geometrie bleibt unveraendert, `join` und `verify` greifen
bitgenau. Ob eine Karte von einem eingebauten Verfahren, einem lokalen Modell oder einem gehosteten Dienst kommt, aendert daran
nichts - deshalb sind alle Quellen austauschbare Adapter mit derselben Schnittstelle (`Segmenter.labels`).

Schluessel und Adressen kommen nur aus Umgebungsvariablen, nie aus Dateien oder aus der Kommandozeile:

    MESHCUT_SEGMENTER_CMD   Befehl eines lokalen Modells (Platzhalter {input} {output} {parts}), z. B. PartField/SAMesh-Skript
    MESHCUT_SEGMENT_URL     gehosteter Dienst (https; http nur fuer localhost), MESHCUT_SEGMENT_KEY = Bearer-Schluessel
    MESHCUT_LLM_URL         OpenAI-kompatibler Chat-Endpunkt, MESHCUT_LLM_KEY, MESHCUT_LLM_MODEL
    ANTHROPIC_API_KEY       Anthropic Messages API (MESHCUT_LLM_MODEL waehlt das Modell, MESHCUT_ANTHROPIC_URL die Adresse)
"""
import json
import os
import re
import shlex
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

import kkmeshcut as mc
from kkmeshcut import fail

MAX_RESPONSE_BYTES = 256 << 20
MAX_NAME_LEN = 40
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


# ============================================================================= Hilfen
def triangle_centroids(mesh):
    return mesh.V[mesh.F].mean(axis=1)


def shell_labels(mesh):
    """Zusammenhaengende Huellen als Zuordnungskarte (Etikett = laufende Nummer, nach Dreiecksnummer des ersten Dreiecks)."""
    F = np.asarray(mesh.F, dtype=np.int64)
    n = len(mesh.V)
    comp = np.arange(n, dtype=np.int64)
    used = np.zeros(n, bool)
    used[F.reshape(-1)] = True
    while True:
        m = comp[F].min(axis=1)
        new = comp.copy()
        for k in range(3):
            np.minimum.at(new, F[:, k], m)
        new = new[new]                                           # Zeiger verkuerzen
        if (new == comp).all():
            break
        comp = new
    raw = comp[F[:, 0]]
    _, first = np.unique(raw, return_index=True)
    order = {int(raw[i]): r for r, i in enumerate(sorted(first.tolist()))}
    return np.array([order[int(c)] for c in raw], dtype=np.int64)


def order_by_height(mesh, labels):
    """Etiketten neu nummerieren: Teil 0 ist das hoechste (z), dann abwaerts; Gleichstand nach x, y. Damit sind Namen stabil."""
    cen = triangle_centroids(mesh)
    ids = np.unique(labels)
    key = []
    for i in ids:
        c = cen[labels == i].mean(axis=0)
        key.append((-round(float(c[2]), 9), round(float(c[0]), 9), round(float(c[1]), 9), int(i)))
    rank = {k[3]: r for r, k in enumerate(sorted(key))}
    return np.array([rank[int(x)] for x in labels], dtype=np.int64)


# ============================================================================= Adapter: Teile finden
class Segmenter:
    """Schnittstelle: `labels(mesh, parts)` -> ganzzahlige Zuordnungskarte, ein Wert pro Dreieck. `parts` ist die gewuenschte
    Anzahl (oder None); ein Verfahren darf davon abweichen, wenn es nicht anders geht."""
    name = ""

    def labels(self, mesh, parts=None):
        raise NotImplementedError


class ComponentsSegmenter(Segmenter):
    """Eigenstaendige Huellen (Gehaeuse, lose Teile eines Szenen-Exports)."""
    name = "components"

    def labels(self, mesh, parts=None):
        lab = shell_labels(mesh)
        k = int(lab.max()) + 1 if len(lab) else 0
        if parts and parts != k:
            fail("components: das Netz hat %d getrennte Huellen, nicht %d." % (k, parts))
        return order_by_height(mesh, lab)


def start_centers(X, w, k):
    """Startpunkte fuer k-Means: weitester-Punkt-Verfahren, beginnend am Punkt, der dem gewichteten Gesamtschwerpunkt am naechsten liegt."""
    mean = (X * w[:, None]).sum(axis=0)
    centers = [X[int(np.argmin(((X - mean) ** 2).sum(axis=1)))]]
    dist = ((X - centers[0]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        centers.append(X[nxt])
        dist = np.minimum(dist, ((X - X[nxt]) ** 2).sum(axis=1))
    return np.array(centers)


class KMeansSegmenter(Segmenter):
    """Raeumliche Gruppen: k-Means auf den Dreiecksschwerpunkten (feste Startpunkte, ohne Zufall - gleiche Eingabe, gleiche Karte).
    Einfach, ohne Modell und ohne Netz; fuer Formen mit klaren raeumlichen Teilen (Figur: Kopf/Rumpf/Beine) ein brauchbarer Anfang."""
    name = "kmeans"

    def labels(self, mesh, parts=None):
        k = 2 if parts is None else int(parts)                  # 0 ist keine Teilezahl und wird abgelehnt, nicht still zu 2
        if k < 2 or k > mc.MAX_PARTS:
            fail("kmeans: Teilezahl zwischen 2 und %d noetig." % mc.MAX_PARTS)
        X = triangle_centroids(mesh)
        area = np.linalg.norm(np.cross(mesh.V[mesh.F[:, 1]] - mesh.V[mesh.F[:, 0]], mesh.V[mesh.F[:, 2]] - mesh.V[mesh.F[:, 0]]), axis=1) / 2
        w = area / area.sum() if area.sum() > 0 else np.full(len(X), 1.0 / len(X))
        if len(np.unique(X, axis=0)) < k:
            fail("kmeans: zu wenige verschiedene Dreiecke fuer %d Teile." % k)
        C = start_centers(X, w, k)
        lab = np.zeros(len(X), np.int64)
        for _ in range(100):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            new = d.argmin(axis=1)
            if (new == lab).all() and _ > 0:
                break
            lab = new
            for j in range(k):
                m = lab == j
                if m.any():
                    C[j] = (X[m] * w[m, None]).sum(axis=0) / w[m].sum() if w[m].sum() > 0 else X[m].mean(axis=0)
        if len(np.unique(lab)) < 2:
            fail("kmeans: ergibt nur ein Teil.")
        return order_by_height(mesh, lab)


def split_command(tpl):
    """Befehlszeile in Teile zerlegen. Unter Windows bleiben Rueckstriche erhalten (posix=False), die aeusseren Anfuehrungszeichen
    eines Teils werden aber entfernt - sonst findet Windows einen Pfad in Anfuehrungszeichen nicht."""
    parts = shlex.split(tpl, posix=(os.name != "nt"))
    if os.name == "nt":
        parts = [x[1:-1] if len(x) >= 2 and x[0] == x[-1] and x[0] in "\"'" else x for x in parts]
    return parts


def _no_window_run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW, shell=False, stdin=subprocess.DEVNULL)   # stdin: ohne Konsole (versteckter Server) ist das geerbte Handle ungueltig


class CommandSegmenter(Segmenter):
    """Lokales Modell als Befehl (PartField, SAMesh, eigenes Skript). Der Befehl steht in MESHCUT_SEGMENTER_CMD, z. B.
    `python partfield_run.py --mesh {input} --parts {parts} --out {output}`. meshcut schreibt das Netz als OBJ nach {input} und liest
    die Karte (.npy/.json/.csv, ein Wert pro Dreieck in der Reihenfolge der OBJ-Flaechen) aus {output}."""
    name = "command"

    def __init__(self, timeout=3600):
        self.timeout = timeout

    def labels(self, mesh, parts=None):
        tpl = os.environ.get("MESHCUT_SEGMENTER_CMD", "").strip()
        if not tpl:
            fail("Backend 'command': die Umgebungsvariable MESHCUT_SEGMENTER_CMD ist nicht gesetzt.")
        with tempfile.TemporaryDirectory(prefix="meshcut_seg_") as d:
            inp, outp = Path(d) / "mesh.obj", Path(d) / "labels.npy"
            inp.write_bytes(mc.obj_bytes(mesh.V, mesh.F, "mesh"))
            try:
                argv = [a.format(input=str(inp), output=str(outp), parts=str(parts or "")) for a in split_command(tpl)]
            except (KeyError, IndexError, ValueError) as e:
                fail("MESHCUT_SEGMENTER_CMD ist unbrauchbar (%s). Erlaubt sind {input}, {output}, {parts}." % e)
            try:
                r = _no_window_run(argv, self.timeout)
            except (OSError, subprocess.TimeoutExpired) as e:
                fail("Der Segmentierungsbefehl liess sich nicht ausfuehren (%s)." % (str(e).splitlines() or ["?"])[0])
            if r.returncode != 0:
                # die Ausgabe fremder Programme wird nicht weitergereicht (koennte Schluessel oder Pfade enthalten)
                fail("Der Segmentierungsbefehl endete mit Code %d - zum Nachsehen den Befehl von Hand ausfuehren." % r.returncode)
            if not outp.is_file():
                fail("Der Segmentierungsbefehl hat %s nicht geschrieben." % outp.name)
            return order_by_height(mesh, validate_labels(mc.read_label_file(outp), len(mesh.F)))


class HttpSegmenter(Segmenter):
    """Gehosteter Dienst. Vertrag: meshcut schickt POST an MESHCUT_SEGMENT_URL mit dem Netz als GLB (Content-Type
    model/gltf-binary, Kopf `X-Meshcut-Parts: <n>` falls gewuenscht, `Authorization: Bearer <MESHCUT_SEGMENT_KEY>`); Antwort JSON
    {"labels": [...]} mit einem ganzzahligen Wert pro Dreieck in der Reihenfolge der GLB-Indizes."""
    name = "http"

    def __init__(self, timeout=600):
        self.timeout = timeout

    def labels(self, mesh, parts=None):
        url = os.environ.get("MESHCUT_SEGMENT_URL", "").strip()
        if not url:
            fail("Backend 'http': die Umgebungsvariable MESHCUT_SEGMENT_URL ist nicht gesetzt.")
        check_url(url, "MESHCUT_SEGMENT_URL")
        headers = {"Content-Type": "model/gltf-binary"}
        key = os.environ.get("MESHCUT_SEGMENT_KEY", "").strip()
        if key:
            headers["Authorization"] = "Bearer " + key
        if parts:
            headers["X-Meshcut-Parts"] = str(int(parts))
        body = http_post(url, mc.glb_bytes([("mesh", mesh.V, mesh.F)]), headers, self.timeout)
        try:
            lab = np.asarray(json.loads(body.decode("utf-8"))["labels"])
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            fail("Der Dienst hat keine gueltige Antwort {\"labels\": [...]} geschickt.")
        return order_by_height(mesh, validate_labels(lab, len(mesh.F)))


SEGMENTERS = {c.name: c for c in (ComponentsSegmenter, KMeansSegmenter, CommandSegmenter, HttpSegmenter)}


def get_segmenter(name):
    if name not in SEGMENTERS:
        fail("Unbekanntes Verfahren '%s' - moeglich: %s." % (name, ", ".join(sorted(SEGMENTERS))))
    return SEGMENTERS[name]()


def validate_labels(lab, nfaces):
    lab = np.asarray(lab)
    if lab.shape != (nfaces,):
        fail("Die Karte hat %s Werte, das Netz %d Dreiecke." % (lab.size, nfaces))
    if lab.dtype.kind == "f":
        if not np.isfinite(lab).all() or (lab != np.floor(lab)).any():
            fail("Die Karte muss ganze Zahlen enthalten.")
    elif lab.dtype.kind not in "iub":
        fail("Die Karte muss ganze Zahlen enthalten.")
    lab = lab.astype(np.int64)
    if len(lab) and lab.min() < 0:
        fail("Teilnummern duerfen nicht negativ sein.")
    return lab


# ============================================================================= Netzzugriff (nur kontrolliert)
def check_url(url, what):
    u = urllib.parse.urlparse(url)
    local = (u.hostname or "") in ("localhost", "127.0.0.1", "::1")
    if u.scheme != "https" and not (u.scheme == "http" and local):
        fail("%s muss mit https:// beginnen (http nur fuer localhost)." % what)
    if not u.hostname:
        fail("%s hat keine Adresse." % what)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Weiterleitungen werden nicht gefolgt: sonst ginge der Schluessel (Authorization) an eine Adresse, die niemand geprueft hat."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Weiterleitungen sind nicht erlaubt", headers, fp)


def http_post(url, data, headers, timeout):
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout) as r:
            body = r.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as e:
        fail("Der Dienst antwortet mit Fehler %d." % e.code)
    except (urllib.error.URLError, OSError, ValueError) as e:
        fail("Der Dienst ist nicht erreichbar (%s)." % (str(getattr(e, "reason", e)).splitlines() or ["?"])[0])
    if len(body) > MAX_RESPONSE_BYTES:
        fail("Die Antwort des Dienstes ist zu gross.")
    return body


# ============================================================================= Adapter: Sprachmodell (nur zum Benennen)
class Llm:
    """Schnittstelle: `complete(system, user)` -> Text. Anbieter austauschbar; ohne Schluessel gibt es keinen Aufruf."""
    name = ""

    def complete(self, system, user):
        raise NotImplementedError


class OpenAICompatLlm(Llm):
    name = "openai"

    def complete(self, system, user):
        url = os.environ.get("MESHCUT_LLM_URL", "").strip()
        model = os.environ.get("MESHCUT_LLM_MODEL", "").strip()
        key = os.environ.get("MESHCUT_LLM_KEY", "").strip()
        if not url or not model:
            fail("Sprachmodell: MESHCUT_LLM_URL und MESHCUT_LLM_MODEL muessen gesetzt sein.")
        check_url(url, "MESHCUT_LLM_URL")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        payload = {"model": model, "temperature": 0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        body = http_post(url, json.dumps(payload).encode("utf-8"), headers, 120)
        try:
            return json.loads(body.decode("utf-8"))["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            fail("Das Sprachmodell hat keine verwertbare Antwort geschickt.")


class AnthropicLlm(Llm):
    name = "anthropic"

    def complete(self, system, user):
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            fail("Sprachmodell: ANTHROPIC_API_KEY ist nicht gesetzt.")
        url = os.environ.get("MESHCUT_ANTHROPIC_URL", "").strip() or "https://api.anthropic.com/v1/messages"
        check_url(url, "MESHCUT_ANTHROPIC_URL")
        model = os.environ.get("MESHCUT_LLM_MODEL", "").strip() or "claude-sonnet-5-5"
        headers = {"Content-Type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"}
        payload = {"model": model, "max_tokens": 1024, "system": system, "messages": [{"role": "user", "content": user}]}
        body = http_post(url, json.dumps(payload).encode("utf-8"), headers, 120)
        try:
            return "".join(b.get("text", "") for b in json.loads(body.decode("utf-8"))["content"] if b.get("type") == "text")
        except (ValueError, KeyError, TypeError, AttributeError):
            fail("Das Sprachmodell hat keine verwertbare Antwort geschickt.")


LLMS = {c.name: c for c in (AnthropicLlm, OpenAICompatLlm)}


def default_llm():
    if os.environ.get("MESHCUT_LLM_URL", "").strip():           # ausdruecklich gesetzter Endpunkt gewinnt; der Anthropic-Schluessel geht nie dorthin
        return OpenAICompatLlm()
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return AnthropicLlm()
    fail("Fuer --prompt ist ein Sprachmodell noetig: ANTHROPIC_API_KEY oder MESHCUT_LLM_URL/MESHCUT_LLM_MODEL/MESHCUT_LLM_KEY setzen "
         "(oder --names verwenden, das kommt ohne Sprachmodell aus).")


# ============================================================================= Namen
def clean_name(text):
    s = re.sub(r"[^\w\- .]", "", str(text), flags=re.UNICODE).strip()
    s = re.sub(r"\s+", "_", s)[:MAX_NAME_LEN].strip("_.-")
    return s


def clean_names(raw):
    names = [clean_name(x) for x in raw]
    if any(not n for n in names):
        fail("Teilnamen duerfen nicht leer sein.")
    low = [n.lower() for n in names]
    if len(set(low)) != len(low):
        fail("Teilnamen muessen verschieden sein.")
    return names


def describe_parts(mesh, labels):
    """Kurzbeschreibung je Teil (fuer das Sprachmodell): Lage und Groesse relativ zum ganzen Netz."""
    lo, hi = mesh.V.min(axis=0), mesh.V.max(axis=0)
    ext = np.where(hi - lo > 0, hi - lo, 1.0)
    cen = triangle_centroids(mesh)
    area = np.linalg.norm(np.cross(mesh.V[mesh.F[:, 1]] - mesh.V[mesh.F[:, 0]], mesh.V[mesh.F[:, 2]] - mesh.V[mesh.F[:, 0]]), axis=1) / 2
    total = float(area.sum()) or 1.0
    out = []
    for i in np.unique(labels):
        m = labels == i
        c = ((cen[m] * area[m, None]).sum(axis=0) / (area[m].sum() or 1.0) - lo) / ext
        pl, ph = (mesh.V[mesh.F[m]].reshape(-1, 3).min(axis=0) - lo) / ext, (mesh.V[mesh.F[m]].reshape(-1, 3).max(axis=0) - lo) / ext
        out.append({"part": int(i), "triangles": int(m.sum()), "area_share": round(float(area[m].sum() / total), 3),
                    "center_xyz_0to1": [round(float(x), 3) for x in c], "box_min_xyz_0to1": [round(float(x), 3) for x in pl],
                    "box_max_xyz_0to1": [round(float(x), 3) for x in ph]})
    return out


def extract_json(text):
    m = re.search(r"(\{.*\}|\[.*\])", str(text), re.DOTALL)
    if not m:
        fail("Das Sprachmodell hat kein JSON geliefert.")
    try:
        return json.loads(m.group(1))
    except ValueError:
        fail("Das Sprachmodell hat kein gueltiges JSON geliefert.")


def names_from_prompt(prompt, llm):
    """Schritt 1: aus dem Satz die Teilnamen holen ('trenne Kopf, Arme, Beine' -> ['Kopf','Arme','Beine'])."""
    system = ("Du ziehst aus einem Auftrag zum Zerlegen eines 3D-Modells die gewuenschten Teilnamen. Antworte NUR mit einem JSON-Array "
              "von Strings, in der Reihenfolge des Auftrags, ohne Erklaerung.")
    got = extract_json(llm.complete(system, "Auftrag: " + str(prompt)[:2000]))
    if not isinstance(got, list) or not 2 <= len(got) <= 64 or not all(isinstance(x, str) for x in got):
        fail("Aus dem Auftrag liessen sich keine 2 bis 64 Teilnamen lesen.")
    return clean_names(got)


def assign_names(mesh, labels, names, llm=None):
    """Schritt 2: Namen den Teilen zuordnen. Ohne Sprachmodell in Reihenfolge der Teile (oben -> unten); mit Sprachmodell nach Lage und
    Groesse. Geprueft wird streng: jeder Name genau einmal, nur bekannte Teilnummern."""
    ids = [int(i) for i in np.unique(labels)]
    if len(names) != len(ids):
        fail("%d Namen fuer %d Teile - die Anzahl muss gleich sein." % (len(names), len(ids)))
    if llm is None:
        return {i: n for i, n in zip(ids, names)}
    system = ("Du ordnest Namen den Teilen eines 3D-Modells zu. z zeigt nach oben. Jedes Teil bekommt genau einen der Namen, jeder Name "
              "wird genau einmal benutzt. Antworte NUR mit einem JSON-Objekt {\"<Teilnummer>\": \"<Name>\"}.")
    user = "Namen: %s\nTeile: %s" % (json.dumps(names, ensure_ascii=False), json.dumps(describe_parts(mesh, labels), ensure_ascii=False))
    got = extract_json(llm.complete(system, user))
    if not isinstance(got, dict):
        fail("Das Sprachmodell hat keine Zuordnung geliefert.")
    out = {}
    for k, v in got.items():
        try:
            pid = int(k)
        except (TypeError, ValueError):
            fail("Das Sprachmodell hat eine unbekannte Teilnummer '%s' geliefert." % k)
        if pid not in ids or not isinstance(v, str) or pid in out:
            fail("Das Sprachmodell hat eine unbrauchbare Zuordnung geliefert.")
        out[pid] = v
    if sorted(out) != ids or sorted(out.values()) != sorted(names):
        fail("Das Sprachmodell hat nicht jedem Teil genau einen Namen gegeben.")
    return out


# ============================================================================= Gesamtablauf
def segment_mesh(mesh, backend="kmeans", parts=None, names=None, prompt=None, llm=None):
    if parts is not None and (isinstance(parts, bool) or not isinstance(parts, int) or parts < 1):
        fail("--parts: eine ganze Zahl ab 1 noetig.")
    names = clean_names(names) if names else None
    if prompt and names:
        fail("Entweder --names oder --prompt angeben, nicht beides.")
    if prompt:
        llm = llm or default_llm()
        names = names_from_prompt(prompt, llm)
    if names:
        if parts and parts != len(names):
            fail("--parts %d passt nicht zu %d Namen." % (parts, len(names)))
        parts = len(names)
    seg = get_segmenter(backend)
    labels = validate_labels(seg.labels(mesh, parts), len(mesh.F))
    found = len(np.unique(labels))
    if found < 2:
        fail("Das Verfahren '%s' hat nur ein Teil gefunden." % backend)
    if parts and found != parts:
        fail("Das Verfahren '%s' hat %d Teile geliefert, gewuenscht waren %d." % (backend, found, parts))
    # Teilnummern dicht und stabil (0..n-1)
    ids = np.unique(labels)
    labels = np.searchsorted(ids, labels).astype(np.int64)
    named = assign_names(mesh, labels, names, llm if prompt else None) if names else None
    return labels, named


def segment_file(src, outdir, backend="kmeans", parts=None, names=None, prompt=None, fmt="stl", llm=None):
    """Datei laden, Teile finden (optional benennen), als Zuordnungskarte schreiben. Rueckgabe: Begleitdaten wie bei `split`."""
    mesh = mc.load_mesh(src)
    labels, named = segment_mesh(mesh, backend=backend, parts=parts, names=names, prompt=prompt, llm=llm)
    res = mc.split_labels(mesh, labels)
    res.names = named
    return mc.write_split(res, outdir, fmt, inputs=[src])
