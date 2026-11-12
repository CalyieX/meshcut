"""meshcut als Dienst fuer Agenten und andere Programme: MCP (stdio) und HTTP-API.

Beide sprechen dieselben Werkzeuge: info, split, segment, join, verify, regions, remove, swap. Dateien werden nur unterhalb eines Wurzelordners
gelesen und geschrieben (Umgebungsvariable MESHCUT_ROOT oder `--root`, sonst der aktuelle Ordner); Pfade ausserhalb - auch ueber
Links - werden abgewiesen. Es gibt keine Netzfreigabe ohne Schluessel: die HTTP-API bindet an 127.0.0.1, und nur mit
MESHCUT_TOKEN (Bearer-Schluessel aus der Umgebung) auch an andere Adressen.

    meshcut serve --mcp                 # MCP ueber stdin/stdout (fuer Claude, Codex, andere Agenten)
    meshcut serve --http 8765           # POST http://127.0.0.1:8765/v1/<werkzeug> mit JSON-Koerper
"""
import hmac
import json
import math
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import kkmeshcut as mc
from kkmeshcut import MeshcutError

MAX_BODY = 1 << 20
NL = chr(10)
LOCK = threading.Lock()                                       # ein Werkzeug zur Zeit: die Werkzeuge schreiben auf die Platte
MCP_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")


# ============================================================================= Wurzelordner
class Sandbox:
    def __init__(self, root=None):
        r = Path(root or os.environ.get("MESHCUT_ROOT") or os.getcwd()).resolve()
        if not r.is_dir():
            raise MeshcutError("Wurzelordner %s gibt es nicht." % r)
        self.root = r

    def path(self, value, what):
        if not isinstance(value, str) or not value.strip() or "\0" in value:
            raise MeshcutError("%s: Pfad fehlt." % what)
        p = Path(value)
        p = (p if p.is_absolute() else self.root / p).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise MeshcutError("%s liegt ausserhalb des erlaubten Ordners." % what) from None
        return p


def _scrub(tools, text):
    """Fehlermeldungen nennen den Wurzelordner des Dienstes nicht (Pfade erscheinen relativ dazu)."""
    r = str(tools.sb.root)
    dbl = r.replace(chr(92), chr(92) + chr(92))                                 # repr() einer OSError-Meldung verdoppelt die Backslashes
    for form in (dbl + chr(92) + chr(92), dbl + os.sep, r + os.sep, r.replace(os.sep, "/") + "/", dbl, r, r.replace(os.sep, "/")):
        text = text.replace(form, "" if form.endswith(("/", os.sep)) else ".")
    return text


def _scrub_obj(tools, obj):
    """Auch Erfolgsantworten (z. B. 'Geschrieben: <Pfad>') nennen den Wurzelordner nicht; NaN und Unendlich (kein gueltiges JSON) werden zu null."""
    if isinstance(obj, str):
        return _scrub(tools, obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, list):
        return [_scrub_obj(tools, x) for x in obj]
    if isinstance(obj, dict):
        return {k: _scrub_obj(tools, v) for k, v in obj.items()}
    return obj


# ============================================================================= Werkzeuge
def _opt(args, key, kind):
    v = args.get(key)
    if v is None:
        return None
    if kind == "bool":
        if not isinstance(v, bool):
            raise MeshcutError("%s: true oder false noetig." % key)
    elif kind == "int":
        if isinstance(v, bool) or not isinstance(v, int):
            raise MeshcutError("%s: ganze Zahl noetig." % key)
    elif kind == "str":
        if not isinstance(v, str):
            raise MeshcutError("%s: Text noetig." % key)
    elif kind == "num":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise MeshcutError("%s: Zahl noetig." % key)
    elif kind == "format":
        if v not in mc.FORMATS:
            raise MeshcutError("format: moeglich sind %s." % ", ".join(mc.FORMATS))
    elif kind == "strlist":
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise MeshcutError("%s: Liste von Texten noetig." % key)
    return v


def _summary(side):
    return {"parts": [{"id": p["id"], "file": p["file"], "faces": p["faces"], "cap_faces": p["cap_faces"]} for p in side["parts"]],
            "names": side.get("names"), "format": side["format"], "mode": side["mode"], "source_hash": side["source"]["hash"],
            "precision": side["source"]["precision"]}


class Tools:
    def __init__(self, sandbox):
        self.sb = sandbox

    def info(self, a):
        st = mc.info(self.sb.path(a.get("path"), "path"))
        return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in st.items()}

    def split(self, a):
        src, out = self.sb.path(a.get("input"), "input"), self.sb.path(a.get("outdir"), "outdir")
        opts = {}
        for key, kind in (("plane", "strlist"), ("grid", "str"), ("max", "str"), ("wedges", "int"), ("axis", "str"), ("center", "str"),
                          ("labels", "str"), ("format", "format"), ("no_cap", "bool"), ("flip", "bool"), ("verify", "bool"), ("snap", "num")):
            v = _opt(a, key, kind)
            if v is not None:
                opts[key] = self.sb.path(v, "labels") if key == "labels" else v
        return _summary(mc.split(src, out, **opts))

    def segment(self, a):
        import kkmeshcut_ai as ai
        parts = _opt(a, "parts", "int")
        if (_opt(a, "backend", "str") or "kmeans") == "kmeans" and parts is not None and not 2 <= parts <= 1024:
            raise MeshcutError("parts: 2 bis 1024 Teile (kmeans).")
        src, out = self.sb.path(a.get("input"), "input"), self.sb.path(a.get("outdir"), "outdir")
        return _summary(ai.segment_file(src, out, backend=_opt(a, "backend", "str") or "kmeans", parts=_opt(a, "parts", "int"),
                                        names=_opt(a, "names", "strlist"), prompt=_opt(a, "prompt", "str"),
                                        fmt=_opt(a, "format", "format") or "stl"))

    def join(self, a):
        parts = a.get("parts")
        items = parts if isinstance(parts, list) else [parts]
        out = mc.join([self.sb.path(x, "parts") for x in items], self.sb.path(a.get("output"), "output"))
        return {"output": str(out.relative_to(self.sb.root)) if out.is_absolute() else str(out)}

    def verify(self, a):
        src = a.get("source")
        r = mc.verify(self.sb.path(a.get("parts"), "parts"), source=None if src is None else self.sb.path(src, "source"))
        return {"ok": r["ok"], "checks": [{"name": n, "ok": ok, "text": t} for n, ok, t in r["checks"]],
                "source_hash": r["source_hash"], "joined_hash": r["joined_hash"]}


    def regions(self, a):
        opts = {}
        size = _opt(a, "size", "int")
        if size is not None:
            opts["size"] = size
        js = mc.regions(self.sb.path(a.get("input"), "input"), self.sb.path(a.get("outdir"), "outdir"), **opts)
        return js

    def _edit(self, cmd, a, keys):
        src = self.sb.path(a.get("input"), "input")
        opts = {}
        for key, kind in keys:
            v = _opt(a, key, kind)
            if v is not None:
                opts[key] = v
        if "out" in opts:
            opts["out"] = self.sb.path(opts["out"], "out")
        r = getattr(mc, cmd)(src, **opts)
        out = Path(r["output"]) if r["output"] else None
        return {"output": str(out.relative_to(self.sb.root)) if out is not None and out.is_absolute() and self.sb.root in out.parents else r["output"],
                "report": r["text"]}

    def remove(self, a):
        return self._edit("remove", a, (("region", "int"), ("at", "str"), ("name", "str"), ("out", "str")))

    def swap(self, a):
        return self._edit("swap", a, (("region", "int"), ("at", "str"), ("name", "str"), ("out", "str"), ("shape", "str"), ("axis", "str"),
                                      ("scale", "num"), ("material", "str")))


def tool_specs():
    S = {"type": "string"}
    SEL = {"region": {"type": "integer", "description": "Nummer aus regions.json"}, "at": {"type": "string", "description": "Punkt x,y,z auf dem Element"},
           "name": {"type": "string", "description": "Name eines Objekts, einer Gruppe oder eines Materials"},
           "out": {"type": "string", "description": "Ausgabedatei (Vorgabe: <Eingabe>_edit.<Endung>)"}}
    return [
        {"name": "info", "description": "Kennzahlen eines Netzes (Dreiecke, Volumen, Flaeche, wasserdicht, Hash).",
         "inputSchema": {"type": "object", "properties": {"path": S}, "required": ["path"]}},
        {"name": "split", "description": "Netz exakt zerlegen (Ebenen, Gitter, Druckraum, Keile oder Zuordnungskarte) und mit Deckeln schliessen. "
         "Schreibt Teile + meshcut.json in outdir; join/verify stellen das Original bitgenau wieder her.",
         "inputSchema": {"type": "object", "properties": {
             "input": S, "outdir": S, "plane": {"type": "array", "items": S}, "grid": S, "max": S, "wedges": {"type": "integer"}, "axis": S,
             "center": S, "snap": {"type": "number"}, "labels": S, "format": {"type": "string", "enum": list(mc.FORMATS)}, "no_cap": {"type": "boolean"},
             "flip": {"type": "boolean"}, "verify": {"type": "boolean"}}, "required": ["input", "outdir"]}},
        {"name": "segment", "description": "Teile automatisch finden (kmeans, components, command, http) und optional benennen (names oder prompt); "
         "das Ergebnis ist eine Zuordnungskarte, also exakt wieder zusammensetzbar.",
         "inputSchema": {"type": "object", "properties": {
             "input": S, "outdir": S, "backend": S, "parts": {"type": "integer"}, "names": {"type": "array", "items": S}, "prompt": S,
             "format": {"type": "string", "enum": list(mc.FORMATS)}}, "required": ["input", "outdir"]}},
        {"name": "join", "description": "Teile zusammensetzen: ein Ordner mit meshcut.json (bitgenau) oder eine Liste von Dateien (Naeherung).",
         "inputSchema": {"type": "object", "properties": {"parts": {"anyOf": [S, {"type": "array", "items": S}]}, "output": S}, "required": ["parts", "output"]}},
        {"name": "regions", "description": "Elemente eines Netzes nummerieren (regions.png mit Nummern + regions.json mit Punkt, Rahmen, Material je Element), "
         "damit man 'Teil 7' waehlen kann statt Koordinaten zu raten.",
         "inputSchema": {"type": "object", "properties": {"input": S, "outdir": S, "size": {"type": "integer"}}, "required": ["input", "outdir"]}},
        {"name": "remove", "description": "Ein Element (region | at | name) entfernen und das Loch sauber schliessen; Material, UV und Normalen der Umgebung bleiben. "
         "Die Antwort enthaelt die Pruefzeile (was sich geaendert hat, dass der Rest identisch ist).",
         "inputSchema": {"type": "object", "properties": dict({"input": S}, **SEL), "required": ["input"]}},
        {"name": "swap", "description": "Ein Element (region | at | name) durch einen Grundkoerper gleicher Groesse ersetzen; Material, UV-Bereich und Glaettung automatisch. "
         "shape, axis, scale und material sind nur Feinschliff.",
         "inputSchema": {"type": "object", "properties": dict({"input": S, "shape": {"type": "string", "enum": ["auto", "box", "sphere", "cylinder"]},
                                                              "axis": {"type": "string", "enum": ["auto", "x", "y", "z"]}, "scale": {"type": "number"}, "material": S}, **SEL),
                         "required": ["input"]}},
        {"name": "verify", "description": "Rundlauf eines Teileordners pruefen (Hash je Teil, Deckel auf den Ebenen, Volumen, Flaeche, wasserdicht, Euler, Hash der Quelle).",
         "inputSchema": {"type": "object", "properties": {"parts": S, "source": S}, "required": ["parts"]}},
    ]


def call_tool(tools, name, args):
    if name not in {s["name"] for s in tool_specs()}:
        raise MeshcutError("Unbekanntes Werkzeug '%s'." % name)
    if not isinstance(args, dict):
        raise MeshcutError("Die Argumente muessen ein JSON-Objekt sein.")
    known = next(s for s in tool_specs() if s["name"] == name)["inputSchema"]["properties"]
    unknown = [k for k in args if k not in known]
    if unknown:                                                  # ein Tippfehler wie 'nocap' wuerde sonst still ignoriert
        raise MeshcutError("Unbekanntes Argument '%s' bei %s - moeglich: %s." % (str(unknown[0])[:30], name, ", ".join(sorted(known))))
    with LOCK:
        return _scrub_obj(tools, getattr(tools, name)(args))


# ============================================================================= MCP (stdio)
def _mcp_reply(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    if error:
        msg["error"] = error
    else:
        msg["result"] = result
    return msg


def mcp_handle(tools, msg):
    """Eine JSON-RPC-Nachricht -> Antwort (dict) oder None bei Benachrichtigungen."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        return _mcp_reply(msg.get("id") if isinstance(msg, dict) else None, error={"code": -32600, "message": "Ungueltige Anfrage"})
    method, id_, params = msg["method"], msg.get("id"), msg.get("params") or {}
    if "id" not in msg:
        return None                                              # notifications/initialized, notifications/cancelled ...
    if method == "initialize":
        want = params.get("protocolVersion") if isinstance(params, dict) else None
        return _mcp_reply(id_, {"protocolVersion": want if want in MCP_VERSIONS else MCP_VERSIONS[0], "capabilities": {"tools": {}},
                                "serverInfo": {"name": "meshcut", "version": mc.__version__}})
    if method == "ping":
        return _mcp_reply(id_, {})
    if method == "tools/list":
        return _mcp_reply(id_, {"tools": tool_specs()})
    if method == "tools/call":
        if not isinstance(params, dict) or not isinstance(params.get("name"), str):
            return _mcp_reply(id_, error={"code": -32602, "message": "name fehlt"})
        try:
            res = call_tool(tools, params["name"], params.get("arguments") or {})
            return _mcp_reply(id_, {"content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False)}], "isError": False})
        except MeshcutError as e:
            return _mcp_reply(id_, {"content": [{"type": "text", "text": "Fehler: %s" % _scrub(tools, str(e))}], "isError": True})
        except Exception as e:                                   # nie den Dienst beenden
            return _mcp_reply(id_, {"content": [{"type": "text", "text": "Fehler: %s" % _scrub(tools, (str(e).splitlines() or [type(e).__name__])[0])}], "isError": True})
    return _mcp_reply(id_, error={"code": -32601, "message": "Methode unbekannt: %s" % method})


def serve_mcp(tools, stdin=None, stdout=None):
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    while True:
        line = stdin.readline(MAX_BODY + 1)
        if not line:
            break
        if len(line) > MAX_BODY and not line.endswith(NL):          # zu lange Zeile: Rest bis zum Zeilenende verwerfen
            while line and not line.endswith(NL):
                line = stdin.readline(MAX_BODY)
            stdout.write(json.dumps(_mcp_reply(None, error={"code": -32600, "message": "Zeile zu lang (hoechstens %d Byte)" % MAX_BODY})) + NL)
            stdout.flush()
            continue
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:                                        # auch RecursionError bei tief verschachteltem JSON
            reply = _mcp_reply(None, error={"code": -32700, "message": "Kein gueltiges JSON"})
        else:
            try:
                reply = mcp_handle(tools, msg)
            except Exception as e:                               # eine kaputte Nachricht beendet nie den Dienst
                reply = _mcp_reply(msg.get("id") if isinstance(msg, dict) else None,
                                   error={"code": -32603, "message": "Interner Fehler: %s" % type(e).__name__})
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()


# ============================================================================= HTTP
def make_handler(tools, token, timeout=30):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8", "replace")     # ein einzelnes Surrogat im JSON darf die Antwort nicht abbrechen
            if code >= 400:
                self.close_connection = True                       # der Body einer abgelehnten Anfrage wurde nicht gelesen: sonst wird er als naechste Anfrage geparst (Request-Smuggling)
            self.send_response(code)
            if code >= 400:
                self.send_header("Connection", "close")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            if not token:
                # ohne Schluessel nur lokal - und nur mit Host-Kopf localhost: schuetzt vor DNS-Rebinding aus dem Browser
                host = self.headers.get("Host", "").strip().lower()
                host = host[1:host.find("]")] if host.startswith("[") and "]" in host else host.rsplit(":", 1)[0] if host.count(":") == 1 else host
                return host in ("127.0.0.1", "localhost", "::1")
            got = self.headers.get("Authorization", "")
            return got.startswith("Bearer ") and hmac.compare_digest(got[7:].encode(), token.encode())

        def _bad_framing(self):
            """Content-Length nur als Dezimalzahl und hoechstens einmal; kein Transfer-Encoding. Sonst waere die Grenze zwischen zwei Anfragen auf einer Verbindung unklar."""
            lens = self.headers.get_all("Content-Length") or []
            return len(lens) > 1 or any(not re.fullmatch(r"[0-9]{1,12}", x.strip()) for x in lens) or bool(self.headers.get("Transfer-Encoding"))

        def do_GET(self):
            if self._bad_framing() or int(self.headers.get("Content-Length", "0") or 0) > 0:      # ein GET braucht keinen Body; ungelesen bliebe er als naechste Anfrage stehen
                return self._send(400, {"error": "Ungueltige Anfrage (Content-Length oder Body bei GET)."})
            if not self._authorized():
                return self._send(401, {"error": "Schluessel fehlt oder falsch."})
            if self.path == "/v1/health":
                return self._send(200, {"ok": True, "version": mc.__version__})
            if self.path == "/v1/tools":
                return self._send(200, {"tools": tool_specs()})
            self._send(404, {"error": "Unbekannter Pfad."})

        def do_POST(self):
            if not self._authorized():
                return self._send(401, {"error": "Schluessel fehlt oder falsch."})
            if not self.path.startswith("/v1/"):
                return self._send(404, {"error": "Unbekannter Pfad."})
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                return self._send(415, {"error": "Content-Type: application/json noetig."})     # kein 'einfacher' Browser-Post (CSRF)
            if self.headers.get("Transfer-Encoding"):
                return self._send(411, {"error": "Content-Length noetig (kein chunked)."})
            n = -1 if self._bad_framing() else int(self.headers.get("Content-Length", "0"))
            if n < 0 or n > MAX_BODY:
                return self._send(413, {"error": "Anfrage zu gross (hoechstens %d Byte)." % MAX_BODY})
            try:
                args = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            except Exception:                                    # ValueError, UnicodeDecodeError, RecursionError
                return self._send(400, {"error": "Kein gueltiges JSON."})
            try:
                self._send(200, call_tool(tools, self.path[4:], args))
            except MeshcutError as e:
                self._send(400, {"error": _scrub(tools, str(e))})
            except Exception as e:
                self._send(500, {"error": _scrub(tools, (str(e).splitlines() or [type(e).__name__])[0])})
    H.timeout = timeout                                           # ein stockender oder halber Aufruf haelt keinen Thread dauerhaft fest
    return H


def make_http_server(tools, port, host="127.0.0.1", token=None, timeout=30):
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise MeshcutError("Eine Adresse ausser localhost braucht MESHCUT_TOKEN (Schluessel).")
    srv = ThreadingHTTPServer((host, int(port)), make_handler(tools, token, timeout))
    srv.daemon_threads = True
    return srv


def cmd_serve(a):
    tools = Tools(Sandbox(a.root))
    if a.mcp == bool(a.http is not None):
        raise MeshcutError("Genau eines angeben: --mcp oder --http PORT.")
    if a.mcp:
        for stream, newline in ((sys.stdin, None), (sys.stdout, "\n")):     # Protokoll ist UTF-8, unabhaengig von der Windows-Codepage
            try:
                stream.reconfigure(encoding="utf-8", errors="replace", **({"newline": newline} if newline else {}))
            except (AttributeError, ValueError):
                pass
        serve_mcp(tools)
        return
    token = os.environ.get("MESHCUT_TOKEN", "").strip() or None
    srv = make_http_server(tools, a.http, a.host, token)
    print("meshcut-API auf http://%s:%d/v1/ (Ordner %s)%s" % (a.host, srv.server_address[1], tools.sb.root, ", mit Schluessel" if token else ""), file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
