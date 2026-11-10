"""KI-Schicht (Teile finden und benennen), PLY, MCP-Server und HTTP-API."""
import http.client
import http.server
import json
import os
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest

import kkmeshcut as mc
import kkmeshcut_ai as ai
import kkmeshcut_serve as serve
import shapes

ROOT = Path(__file__).resolve().parents[1]


def two_cubes(gap=3.0):
    a, b = shapes.cube(1.0), shapes.cube(1.0, lo=(gap, 0, 0))
    V = np.concatenate([a.V, b.V])
    F = np.concatenate([a.F, b.F + len(a.V)])
    return mc.Mesh(V, F, "zwei")


def figure():
    """Kugel (Kopf) ueber Quader (Rumpf) ueber Quader (Beine) - drei raeumliche Teile, jeweils fuer sich geschlossen."""
    head = shapes.sphere(12, 8, r=1.0, c=(0, 0, 5.0))
    body = shapes.cube(2.0, lo=(-1, -1, 2.5))
    legs = shapes.cube(2.0, lo=(-1, -1, 0.0))
    V, F, off = [], [], 0
    for m in (head, body, legs):
        V.append(m.V)
        F.append(m.F + off)
        off += len(m.V)
    return mc.Mesh(np.concatenate(V), np.concatenate(F), "figur")


def write(path, mesh, fmt="obj"):
    Path(path).write_bytes(mc.mesh_bytes(fmt, mesh.V, mesh.F))
    return Path(path)


# ----------------------------------------------------------------------------- PLY
@pytest.mark.parametrize("flavor", ["binary_little_endian", "binary_big_endian", "ascii"])
def test_ply_reads_all_three_flavors(tmp_path, flavor):
    m = shapes.cube(2.0)
    V, F = m.V, m.F
    if flavor == "ascii":
        lines = ["ply", "format ascii 1.0", "element vertex %d" % len(V), "property float x", "property float y", "property float z",
                 "element face %d" % len(F), "property list uchar int vertex_indices", "end_header"]
        lines += ["%r %r %r" % tuple(v) for v in V.tolist()] + ["3 %d %d %d" % tuple(f) for f in F.tolist()]
        data = ("\n".join(lines) + "\n").encode()
    else:
        order = "<" if flavor.endswith("little_endian") else ">"
        head = ("ply\nformat %s 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\n"
                "element face %d\nproperty list uchar int vertex_indices\nend_header\n" % (flavor, len(V), len(F))).encode()
        vt = np.zeros(len(V), dtype=np.dtype([("x", order + "f4"), ("y", order + "f4"), ("z", order + "f4"), ("r", "u1")]))
        vt["x"], vt["y"], vt["z"] = V[:, 0], V[:, 1], V[:, 2]
        ft = np.zeros(len(F), dtype=np.dtype([("n", "u1"), ("i", order + "i4", 3)]))
        ft["n"], ft["i"] = 3, F
        data = head + vt.tobytes() + ft.tobytes()
    (tmp_path / "c.ply").write_bytes(data)
    back = mc.load_mesh(tmp_path / "c.ply")
    assert len(back.F) == 12 and mc.edge_report(back.F, len(back.V))["watertight"]
    assert np.allclose(sorted(map(tuple, back.V.tolist())), sorted(map(tuple, V.tolist())))


def test_ply_quads_are_triangulated_and_bad_input_is_refused(tmp_path):
    quad = (b"ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 1\n"
            b"property list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n")
    got = mc.read_meshes_bytes("ply", quad, "q")[0]
    assert len(got.F) == 2
    for bad in (quad.replace(b"4 0 1 2 3", b"4 0 1 2 9"), b"plx\n", quad[:-8], quad.replace(b"ascii", b"binary_middle_endian")):
        (tmp_path / "b.ply").write_bytes(bad)
        with pytest.raises(mc.MeshcutError):
            mc.load_mesh(tmp_path / "b.ply")


def test_ply_roundtrip_is_exact_and_split_works_in_ply(tmp_path):
    m = shapes.sphere(14, 9, r=5.0)
    src = write(tmp_path / "s.obj", m)
    mc.split(src, tmp_path / "o", plane=["x=0.4", "z=0.3"], format="ply")
    assert sorted(f.name for f in (tmp_path / "o").iterdir()) == ["meshcut.json", "part_0.ply", "part_1.ply", "part_2.ply", "part_3.ply"]
    assert mc.verify(tmp_path / "o", source=src)["ok"]
    mc.join(tmp_path / "o", tmp_path / "back.ply")
    back = mc.load_mesh(tmp_path / "back.ply")
    assert mc.triangle_hash(back.V, back.F) == mc.triangle_hash(m.V, m.F)


# ----------------------------------------------------------------------------- Namen
def test_names_are_written_validated_and_used_as_object_names(tmp_path):
    src = write(tmp_path / "f.obj", figure())
    names = ["Kopf", "Rumpf", "Beine"]
    side = mc.segment(src, tmp_path / "o", names=names, format="3mf")
    assert side["names"] == {"0": "Kopf", "1": "Rumpf", "2": "Beine"} and side["mode"] == "labels"
    xml = zipfile.ZipFile(tmp_path / "o" / "parts.3mf").read("3D/3dmodel.model").decode()
    for n in names:
        assert 'name="%s"' % n in xml
    assert mc.verify(tmp_path / "o", source=src)["ok"]
    # Kopf = oben (Teil 0), Beine = unten
    parts = {pid: V for pid, V, F in mc.load_parts(tmp_path / "o")[1]}
    assert parts[0][:, 2].mean() > parts[1][:, 2].mean() > parts[2][:, 2].mean()


@pytest.mark.parametrize("names", [{"0": "../x", "1": "b"}, {"0": "a", "1": "A"}, {"0": "a", "7": "b"}, {"0": "a" * 65}, ["a", "b"], {"x": "a"}])
def test_bad_names_in_the_sidecar_are_refused(tmp_path, names):
    src = write(tmp_path / "f.obj", figure())
    mc.segment(src, tmp_path / "o", parts=3)
    p = tmp_path / "o" / "meshcut.json"
    side = json.loads(p.read_text())
    side["names"] = names
    p.write_text(json.dumps(side))
    with pytest.raises(mc.MeshcutError):
        mc.verify(tmp_path / "o")


def test_names_must_be_unique_and_not_empty():
    with pytest.raises(mc.MeshcutError):
        ai.clean_names(["Arm", "arm"])
    with pytest.raises(mc.MeshcutError):
        ai.clean_names(["Arm", "???"])
    assert ai.clean_names(["Linker Arm", "Kopf/Hals"]) == ["Linker_Arm", "KopfHals"]


# ----------------------------------------------------------------------------- eingebaute Verfahren
def test_components_finds_separate_shells_and_round_trips(tmp_path):
    src = write(tmp_path / "z.obj", two_cubes())
    side = mc.segment(src, tmp_path / "o", backend="components", format="obj")
    assert [p["faces"] for p in side["parts"]] == [12, 12]
    assert mc.verify(tmp_path / "o", source=src)["ok"]
    with pytest.raises(mc.MeshcutError, match="getrennte Huellen"):
        mc.segment(src, tmp_path / "o2", backend="components", parts=3)


def test_kmeans_is_deterministic_and_finds_the_three_blocks(tmp_path):
    mesh = figure()
    a = ai.KMeansSegmenter().labels(mesh, 3)
    b = ai.KMeansSegmenter().labels(mesh, 3)
    assert (a == b).all()
    z = ai.triangle_centroids(mesh)[:, 2]
    for lab, lo, hi in ((0, 3.9, 6.1), (1, 2.4, 4.6), (2, -0.1, 2.1)):
        assert lo <= z[a == lab].mean() <= hi
    src = write(tmp_path / "f.obj", mesh)
    mc.segment(src, tmp_path / "o", parts=3)
    assert mc.verify(tmp_path / "o", source=src)["ok"]


def test_segment_errors_are_clear(tmp_path):
    src = write(tmp_path / "f.obj", figure())
    with pytest.raises(mc.MeshcutError, match="Unbekanntes Verfahren"):
        mc.segment(src, tmp_path / "o", backend="nope", parts=2)
    with pytest.raises(mc.MeshcutError, match="passt nicht"):
        mc.segment(src, tmp_path / "o", parts=2, names="a,b,c")
    with pytest.raises(mc.MeshcutError, match="nicht beides"):
        mc.segment(src, tmp_path / "o", names="a,b", prompt="trenne a und b")
    with pytest.raises(mc.MeshcutError, match="Sprachmodell"):
        mc.segment(src, tmp_path / "o", prompt="trenne Kopf und Beine")


# ----------------------------------------------------------------------------- Adapter mit lokalem Befehl
FAKE_CMD = """
import sys
import numpy as np
inp, out, parts = sys.argv[1], sys.argv[2], sys.argv[3]
n = sum(1 for l in open(inp) if l.startswith('f '))
lab = (np.arange(n) * int(parts or 2)) // n
np.save(out, lab)
"""


def test_command_backend_runs_a_local_model_script(tmp_path, monkeypatch):
    script = tmp_path / "fake.py"
    script.write_text(FAKE_CMD)
    monkeypatch.setenv("MESHCUT_SEGMENTER_CMD", '"%s" "%s" {input} {output} {parts}' % (sys.executable, script))
    src = write(tmp_path / "f.obj", figure())
    side = mc.segment(src, tmp_path / "o", backend="command", parts=4)
    assert len(side["parts"]) == 4 and mc.verify(tmp_path / "o", source=src)["ok"]
    monkeypatch.setenv("MESHCUT_SEGMENTER_CMD", '"%s" -c "import sys; sys.exit(3)"' % sys.executable)
    with pytest.raises(mc.MeshcutError, match="Code 3"):
        mc.segment(src, tmp_path / "o2", backend="command", parts=2)
    monkeypatch.setenv("MESHCUT_SEGMENTER_CMD", '"%s" -c "pass"' % sys.executable)
    with pytest.raises(mc.MeshcutError, match="nicht geschrieben"):
        mc.segment(src, tmp_path / "o3", backend="command", parts=2)
    monkeypatch.delenv("MESHCUT_SEGMENTER_CMD")
    with pytest.raises(mc.MeshcutError, match="MESHCUT_SEGMENTER_CMD"):
        mc.segment(src, tmp_path / "o4", backend="command", parts=2)


# ----------------------------------------------------------------------------- Adapter mit lokalem HTTP-Dienst
class Fake(http.server.BaseHTTPRequestHandler):
    seen = []
    reply = None

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        Fake.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        code, out = Fake.reply(self.path, {k.lower(): v for k, v in self.headers.items()}, body)
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake_server():
    Fake.seen = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def test_http_backend_sends_glb_with_key_and_reads_labels(tmp_path, monkeypatch, fake_server):
    mesh = figure()
    n = len(mesh.F)
    Fake.reply = lambda path, h, body: (200, {"labels": [int(i * 3 // n) for i in range(n)]})
    monkeypatch.setenv("MESHCUT_SEGMENT_URL", fake_server + "/segment")
    monkeypatch.setenv("MESHCUT_SEGMENT_KEY", "geheim-123")
    src = write(tmp_path / "f.obj", mesh)
    side = mc.segment(src, tmp_path / "o", backend="http", parts=3)
    path, headers, body = Fake.seen[0]
    assert path == "/segment" and headers["authorization"] == "Bearer geheim-123" and headers["x-meshcut-parts"] == "3"
    assert body[:4] == b"glTF"
    assert len(side["parts"]) == 3 and mc.verify(tmp_path / "o", source=src)["ok"]
    assert "geheim-123" not in json.dumps(side) and all("geheim-123" not in f.read_text(errors="ignore") for f in (tmp_path / "o").glob("*.json"))


@pytest.mark.parametrize("reply", [(500, {}), (200, {"labels": [0, 1]}), (200, {"labels": "x"}), (200, {"nope": 1}), (200, {"labels": [-1] * 36})])
def test_http_backend_refuses_bad_answers(tmp_path, monkeypatch, fake_server, reply):
    Fake.reply = lambda *a: reply
    monkeypatch.setenv("MESHCUT_SEGMENT_URL", fake_server)
    src = write(tmp_path / "f.obj", shapes.cube(1.0))
    with pytest.raises(mc.MeshcutError):
        mc.segment(src, tmp_path / "o", backend="http", parts=2)


@pytest.mark.parametrize("url", ["http://example.com/x", "ftp://example.com", "file:///etc/passwd", "https://"])
def test_remote_urls_must_be_https(tmp_path, monkeypatch, url):
    monkeypatch.setenv("MESHCUT_SEGMENT_URL", url)
    src = write(tmp_path / "f.obj", shapes.cube(1.0))
    with pytest.raises(mc.MeshcutError):
        mc.segment(src, tmp_path / "o", backend="http", parts=2)


# ----------------------------------------------------------------------------- Sprachmodell
def test_prompt_with_openai_compatible_model(tmp_path, monkeypatch, fake_server):
    def reply(path, h, body):
        req = json.loads(body)
        assert h["authorization"] == "Bearer llm-key" and req["model"] == "mein-modell" and req["temperature"] == 0
        user = req["messages"][1]["content"]
        if user.startswith("Auftrag:"):
            return 200, {"choices": [{"message": {"content": 'Hier: ["Kopf", "Rumpf", "Beine"]'}}]}
        parts = json.loads(user.split("Teile: ", 1)[1])
        top = max(parts, key=lambda p: p["center_xyz_0to1"][2])["part"]
        bottom = min(parts, key=lambda p: p["center_xyz_0to1"][2])["part"]
        mid = next(p["part"] for p in parts if p["part"] not in (top, bottom))
        return 200, {"choices": [{"message": {"content": json.dumps({str(top): "Kopf", str(mid): "Rumpf", str(bottom): "Beine"})}}]}
    Fake.reply = reply
    monkeypatch.setenv("MESHCUT_LLM_URL", fake_server + "/v1/chat/completions")
    monkeypatch.setenv("MESHCUT_LLM_MODEL", "mein-modell")
    monkeypatch.setenv("MESHCUT_LLM_KEY", "llm-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    src = write(tmp_path / "f.obj", figure())
    side = mc.segment(src, tmp_path / "o", prompt="trenne Kopf, Rumpf und Beine")
    assert side["names"] == {"0": "Kopf", "1": "Rumpf", "2": "Beine"}
    assert "llm-key" not in (tmp_path / "o" / "meshcut.json").read_text()
    assert len(Fake.seen) == 2


def test_prompt_with_anthropic_messages_api(tmp_path, monkeypatch, fake_server):
    def reply(path, h, body):
        assert h["x-api-key"] == "ak" and h["anthropic-version"] and json.loads(body)["model"] == "claude-sonnet-5-5"
        return 200, {"content": [{"type": "text", "text": '["Oben", "Unten"]'}]}
    Fake.reply = reply
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    monkeypatch.setenv("MESHCUT_LLM_URL", fake_server + "/v1/messages")
    monkeypatch.delenv("MESHCUT_LLM_MODEL", raising=False)
    names = ai.names_from_prompt("trenne oben und unten", ai.default_llm())
    assert names == ["Oben", "Unten"]


@pytest.mark.parametrize("answer", ['kein json', '["a"]', '[1, 2]', '["a","A"]', '{"0": "x"}'])
def test_llm_names_are_checked(answer):
    class Bad(ai.Llm):
        def complete(self, system, user):
            return answer
    with pytest.raises(mc.MeshcutError):
        ai.names_from_prompt("x", Bad())


@pytest.mark.parametrize("answer", ['{"0": "Kopf", "1": "Kopf"}', '{"0": "Kopf"}', '{"0": "Kopf", "9": "Beine"}', '{"0": "Kopf", "1": "Fuss"}', '[]', 'nix'])
def test_llm_assignment_is_checked(answer):
    class Bad(ai.Llm):
        def complete(self, system, user):
            return answer
    mesh = two_cubes()
    labels = ai.shell_labels(mesh)
    with pytest.raises(mc.MeshcutError):
        ai.assign_names(mesh, labels, ["Kopf", "Beine"], Bad())


# ----------------------------------------------------------------------------- MCP
def run_mcp(root, messages):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    inp = "\n".join(json.dumps(m) for m in messages) + "\n"
    r = subprocess.run([sys.executable, str(ROOT / "kkmeshcut.py"), "serve", "--mcp", "--root", str(root)], input=inp, capture_output=True,
                       text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    return [json.loads(l) for l in r.stdout.splitlines()]


def test_mcp_server_speaks_the_protocol_and_runs_the_tools(tmp_path):
    write(tmp_path / "f.obj", figure())
    call = lambda i, name, args: {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}}
    out = run_mcp(tmp_path, [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        call(3, "info", {"path": "f.obj"}),
        call(4, "segment", {"input": "f.obj", "outdir": "seg", "parts": 3, "names": ["Kopf", "Rumpf", "Beine"], "format": "obj"}),
        call(5, "verify", {"parts": "seg", "source": "f.obj"}),
        call(6, "join", {"parts": "seg", "output": "back.obj"}),
        call(7, "split", {"input": "f.obj", "outdir": "cut", "plane": ["z=2.5"]}),
        {"jsonrpc": "2.0", "id": 8, "method": "nope"},
    ])
    ids = {m["id"]: m for m in out}
    assert ids[1]["result"]["protocolVersion"] == "2025-06-18" and ids[1]["result"]["serverInfo"]["name"] == "meshcut"
    assert {t["name"] for t in ids[2]["result"]["tools"]} == {"info", "split", "segment", "join", "verify"}
    assert 1 not in ids or "result" in ids[1] and None not in ids
    body = lambda i: json.loads(ids[i]["result"]["content"][0]["text"])
    assert not ids[3]["result"]["isError"] and body(3)["faces"] == len(figure().F)
    assert body(4)["names"]["0"] == "Kopf"
    assert body(5)["ok"] is True
    assert (tmp_path / "back.obj").is_file() and (tmp_path / "cut" / "meshcut.json").is_file()
    assert ids[8]["error"]["code"] == -32601


def test_mcp_server_keeps_every_file_inside_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    write(tmp_path / "geheim.obj", shapes.cube(1.0))
    write(root / "ok.obj", shapes.cube(1.0))
    call = lambda i, name, args: {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}}
    out = run_mcp(root, [
        call(1, "info", {"path": "../geheim.obj"}),
        call(2, "info", {"path": str(tmp_path / "geheim.obj")}),
        call(3, "split", {"input": "ok.obj", "outdir": "../draussen", "plane": ["x=0.5"]}),
        call(4, "join", {"parts": "ok.obj", "output": str(tmp_path / "x.obj")}),
        call(5, "info", {"path": "ok.obj/../../geheim.obj"}),
        call(6, "split", {"input": "ok.obj", "outdir": "o", "plane": "x=0.5"}),
        call(7, "split", {"input": "ok.obj", "outdir": "o", "wedges": True}),
        call(8, "info", {}),
    ])
    for m in out:
        assert m["result"]["isError"], m
    assert not (tmp_path / "draussen").exists() and not (tmp_path / "x.obj").exists()
    assert "ausserhalb" in out[0]["result"]["content"][0]["text"]


def test_mcp_bad_lines_do_not_stop_the_server(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    inp = 'kein json\n{"jsonrpc":"2.0","id":1,"method":"ping"}\n[]\n'
    r = subprocess.run([sys.executable, str(ROOT / "kkmeshcut.py"), "serve", "--mcp", "--root", str(tmp_path)], input=inp, capture_output=True, text=True, env=env, timeout=60)
    out = [json.loads(l) for l in r.stdout.splitlines()]
    assert out[0]["error"]["code"] == -32700 and out[1]["result"] == {} and out[2]["error"]["code"] == -32600


# ----------------------------------------------------------------------------- HTTP-API
@pytest.fixture
def api(tmp_path):
    srv = serve.make_http_server(serve.Tools(serve.Sandbox(tmp_path)), 0, token="tok")
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1], tmp_path
    srv.shutdown()
    srv.server_close()


def post(port, path, body, token="tok", ctype="application/json", host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    h = {"Content-Type": ctype}
    if token:
        h["Authorization"] = "Bearer " + token
    if host:
        h["Host"] = host
    c.request("POST", path, body=body if isinstance(body, bytes) else json.dumps(body), headers=h)
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"{}")


def test_http_api_needs_the_key_and_runs_tools(api):
    port, root = api
    write(root / "f.obj", figure())
    assert post(port, "/v1/info", {"path": "f.obj"}, token=None)[0] == 401
    assert post(port, "/v1/info", {"path": "f.obj"}, token="falsch")[0] == 401
    code, body = post(port, "/v1/segment", {"input": "f.obj", "outdir": "seg", "parts": 3, "format": "obj"})
    assert code == 200 and len(body["parts"]) == 3
    code, body = post(port, "/v1/verify", {"parts": "seg", "source": "f.obj"})
    assert code == 200 and body["ok"]
    assert post(port, "/v1/info", {"path": "../x.obj"})[0] == 400
    assert post(port, "/v1/nope", {})[0] == 400
    assert post(port, "/v1/info", b"{kaputt")[0] == 400
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("GET", "/v1/tools", headers={"Authorization": "Bearer tok"})
    assert c.getresponse().status == 200


def test_http_api_refuses_browser_style_requests(tmp_path):
    srv = serve.make_http_server(serve.Tools(serve.Sandbox(tmp_path)), 0)               # ohne Schluessel, nur lokal
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        write(tmp_path / "f.obj", shapes.cube(1.0))
        assert post(port, "/v1/info", {"path": "f.obj"}, token=None)[0] == 200
        assert post(port, "/v1/info", {"path": "f.obj"}, token=None, ctype="text/plain")[0] == 415            # kein einfacher Cross-Site-Post
        assert post(port, "/v1/info", {"path": "f.obj"}, token=None, host="boese.example")[0] == 401           # DNS-Rebinding
    finally:
        srv.shutdown()
        srv.server_close()


def test_http_api_outside_localhost_requires_a_key(tmp_path):
    with pytest.raises(mc.MeshcutError, match="MESHCUT_TOKEN"):
        serve.make_http_server(serve.Tools(serve.Sandbox(tmp_path)), 0, host="0.0.0.0")


def test_http_request_size_is_limited(api):
    port, root = api
    big = json.dumps({"path": "x" * (serve.MAX_BODY + 10)}).encode()
    try:
        code, _ = post(port, "/v1/info", big)
    except (ConnectionError, http.client.HTTPException):
        return                                                    # Verbindung wurde abgebrochen: auch in Ordnung
    assert code == 413


def test_write_split_refuses_unusable_names(tmp_path):
    mesh = figure()
    res = mc.split_labels(mesh, ai.shell_labels(mesh))
    for bad in ({0: "../x"}, {0: "a" * 65}, {9: "unbekannt"}, {0: ""}):
        res.names = bad
        with pytest.raises(mc.MeshcutError, match="Teilnamen"):
            mc.write_split(res, tmp_path / "o", "stl")
    res.names = {0: "Kopf", 1: "Rumpf", 2: "Beine"}
    assert mc.write_split(res, tmp_path / "ok", "stl")["names"]["2"] == "Beine"


def test_ply_unknown_format_is_named(tmp_path):
    data = (b"ply\nformat binary_middle_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
            b"element face 0\nproperty list uchar int vertex_indices\nend_header\n" + b"\0" * 12)
    (tmp_path / "f.ply").write_bytes(data)
    with pytest.raises(mc.MeshcutError, match="nicht unterstuetzt"):
        mc.load_mesh(tmp_path / "f.ply")
