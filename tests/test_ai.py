"""KI-Schicht (Teile finden und benennen), PLY, MCP-Server und HTTP-API."""
import http.client
import http.server
import json
import os
import struct
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
    monkeypatch.setenv("MESHCUT_ANTHROPIC_URL", fake_server + "/v1/messages")
    monkeypatch.delenv("MESHCUT_LLM_URL", raising=False)
    monkeypatch.delenv("MESHCUT_LLM_MODEL", raising=False)
    llm = ai.default_llm()
    assert isinstance(llm, ai.AnthropicLlm)
    assert ai.names_from_prompt("trenne oben und unten", llm) == ["Oben", "Unten"]


def test_anthropic_key_never_goes_to_llm_url(tmp_path, monkeypatch, fake_server):
    """Beide gesetzt: der ausdruecklich gewaehlte Endpunkt gewinnt, der Anthropic-Schluessel geht nie dorthin."""
    def reply(path, h, body):
        assert "x-api-key" not in h and h.get("authorization") == "Bearer fremd"
        return 200, {"choices": [{"message": {"content": '["A", "B"]'}}]}
    Fake.reply = reply
    monkeypatch.setenv("ANTHROPIC_API_KEY", "geheim-anthropic")
    monkeypatch.setenv("MESHCUT_LLM_URL", fake_server + "/v1/chat/completions")
    monkeypatch.setenv("MESHCUT_LLM_MODEL", "m")
    monkeypatch.setenv("MESHCUT_LLM_KEY", "fremd")
    llm = ai.default_llm()
    assert isinstance(llm, ai.OpenAICompatLlm)
    assert ai.names_from_prompt("zwei teile", llm) == ["A", "B"]
    assert "geheim-anthropic" not in repr(Fake.seen)


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
    assert {t["name"] for t in ids[2]["result"]["tools"]} == {"info", "split", "segment", "join", "verify", "regions", "remove", "swap"}
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


# ----------------------------------------------------------------------------- Befunde der zweiten Pruefrunde
def test_redirects_do_not_carry_the_key_to_another_address(tmp_path, monkeypatch, fake_server):
    class Redirect(http.server.BaseHTTPRequestHandler):
        hits = []

        def log_message(self, *a):
            pass

        def do_POST(self):
            Redirect.hits.append((self.path, self.headers.get("Authorization")))
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(302)
            self.send_header("Location", "http://localhost:%d/zweiter" % self.server.server_address[1])
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            Redirect.hits.append((self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("MESHCUT_SEGMENT_URL", "http://127.0.0.1:%d/erster" % srv.server_address[1])
        monkeypatch.setenv("MESHCUT_SEGMENT_KEY", "geheim")
        src = write(tmp_path / "f.obj", shapes.cube(1.0))
        with pytest.raises(mc.MeshcutError, match="Fehler 302"):
            mc.segment(src, tmp_path / "o", backend="http", parts=2)
        assert [p for p, _ in Redirect.hits] == ["/erster"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_command_labels_must_be_whole_numbers_before_anything_else(tmp_path, monkeypatch):
    script = tmp_path / "frac.py"
    script.write_text("import sys, numpy as np\nnp.save(sys.argv[1], np.array([0.9] * 6 + [1.9] * 6))\n")
    monkeypatch.setenv("MESHCUT_SEGMENTER_CMD", '"%s" "%s" {output}' % (sys.executable, script))
    src = write(tmp_path / "c.obj", shapes.cube(1.0))
    with pytest.raises(mc.MeshcutError, match="ganze Zahlen"):
        mc.segment(src, tmp_path / "o", backend="command", parts=2)


def test_output_of_a_foreign_command_is_not_passed_on(tmp_path, monkeypatch):
    script = tmp_path / "leak.py"
    script.write_text("import sys\nsys.stderr.write('GEHEIMER_SCHLUESSEL')\nsys.exit(1)\n")
    monkeypatch.setenv("MESHCUT_SEGMENTER_CMD", '"%s" "%s"' % (sys.executable, script))
    src = write(tmp_path / "c.obj", shapes.cube(1.0))
    with pytest.raises(mc.MeshcutError) as e:
        mc.segment(src, tmp_path / "o", backend="command", parts=2)
    assert "GEHEIMER" not in str(e.value) and "Code 1" in str(e.value)


def test_a_half_sent_request_does_not_hold_a_thread_forever(tmp_path):
    import socket
    import time
    srv = serve.make_http_server(serve.Tools(serve.Sandbox(tmp_path)), 0, token="tok", timeout=1)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=10)
        s.sendall(b"POST /v1/info HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer tok\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{")
        t0 = time.time()
        got = s.recv(4096)                                  # der Dienst muss die Verbindung selbst beenden
        assert time.time() - t0 < 8 and (got == b"" or b"HTTP/1.1" in got)
        s.close()
    finally:
        srv.shutdown()
        srv.server_close()


BAD_JOURNALS = [
    (["../../victim.txt", "part_0.obj"], {}),                       # Pfad ausserhalb
    (["part_0.obj", "meshcut.json"], {"0": "../victim.txt"}),       # moved zeigt woanders hin
    (["C:/Windows/x.txt"], {}),
    (["part_0.obj"], {"5": "part_0.obj"}),                          # Index ausserhalb
    (["part_0.obj"], {"-1": "part_0.obj"}),                         # negativer Index (Python-Listen-Falle)
    (["part_0.obj", "part_1.obj"], {"0": "part_1.obj"}),            # Name passt nicht zum Index
    (["part_0.obj"], ["x"]),                                        # moved keine Zuordnung
    ("kaputt", {}),
    ([7], {}),
    (["part_0.obj/../../victim.txt"], {}),                          # beginnt wie ein Teilname (match statt fullmatch)
    (["part_0.obj" + chr(10)], {}),                                # Zeilenumbruch am Ende (fullmatch ohne $-Falle)
    (["part_01.obj"], {}),
    (["a.txt"], {}),
]


def _plant_journal(tmp_path, state, names, moved, marker=True, journal=True, aged=True):
    """Ein fremdes Journal in einem Zwischenordner; `aged` macht ihn alt genug, damit lesende Befehle ihn ueberhaupt anfassen."""
    src = write(tmp_path / "f.obj", shapes.cube(2.0, lo=(-1, -1, -1)))
    mc.split(src, tmp_path / "o", plane=["x=0"], format="obj")
    victim = tmp_path / "victim.txt"
    victim.write_text("bleibt")
    aside = tmp_path / "o" / ".meshcut_boese"
    aside.mkdir()
    if marker:
        (aside / mc.MARKER).write_text("meshcut")
    if journal:
        (aside / "journal.json").write_text(json.dumps({"state": state, "names": names, "moved": moved}))
    (aside / "0").write_text("x")
    if aged:
        os.utime(aside, (1, 1))
    return aside, victim


@pytest.mark.parametrize("names,moved", BAD_JOURNALS)
@pytest.mark.parametrize("state", ["moving", "done"])
def test_recovery_rejects_every_bad_journal(tmp_path, names, moved, state):
    """Jede Pruefung des Journals einzeln: ein unbrauchbares Journal bleibt liegen, nichts wird verschoben oder geloescht."""
    aside, victim = _plant_journal(tmp_path, state, names, moved)
    before = sorted(p.name for p in (tmp_path / "o").iterdir())
    mc.recover_outdir(tmp_path / "o", 0)
    assert victim.read_text() == "bleibt" and aside.is_dir() and (aside / "0").is_file()
    assert sorted(p.name for p in (tmp_path / "o").iterdir()) == before


def test_recovery_bad_state_is_rejected(tmp_path):
    aside, victim = _plant_journal(tmp_path, "weg", ["part_0.obj"], {})
    mc.recover_outdir(tmp_path / "o", 0)
    assert aside.is_dir() and (tmp_path / "o" / "part_0.obj").is_file()


def test_recovery_ignores_folders_without_marker_or_journal(tmp_path):
    aside, _ = _plant_journal(tmp_path, "moving", ["part_0.obj"], {}, marker=False, journal=False)
    mc.recover_outdir(tmp_path / "o", 0)
    assert aside.is_dir() and (aside / "0").is_file()


def test_recovery_removes_marker_only_staging_but_only_when_old(tmp_path):
    aside, _ = _plant_journal(tmp_path, "moving", ["part_0.obj"], {}, journal=False, aged=False)
    mc.recover_outdir(tmp_path / "o", 600)
    assert aside.is_dir()
    mc.recover_outdir(tmp_path / "o", 0)
    assert not aside.exists()


@pytest.mark.parametrize("state,content", [("done", "geblieben"), ("moving", "x")])
def test_recovery_good_journal_is_carried_out(tmp_path, state, content):
    """Gegenprobe: ein gueltiges Journal wird wirklich abgearbeitet (sonst waeren die Ablehnungen oben wertlos).
    done: der neue Satz bleibt, der Rest geht weg. moving: das zur Seite gelegte Original kommt zurueck."""
    aside, _ = _plant_journal(tmp_path, state, ["part_0.obj"], {"0": "part_0.obj"})
    (tmp_path / "o" / "part_0.obj").write_text("geblieben")
    mc.recover_outdir(tmp_path / "o", 0)
    assert not aside.exists() and (tmp_path / "o" / "part_0.obj").read_text() == content


@pytest.mark.parametrize("names,moved", [(["../../victim.txt", "part_0.obj"], {}), (["part_0.obj", "meshcut.json"], {"0": "../victim.txt"}),
                                         (["C:/Windows/x.txt"], {}), (["part_0.obj"], {"5": "part_0.obj"}), ("kaputt", {})])
def test_recovery_journal_cannot_reach_outside_its_folder(tmp_path, names, moved):
    """Das Journal liegt als Text im Ordner und darf nichts ausserhalb davon loeschen oder verschieben (auch nicht ueber verify)."""
    src = write(tmp_path / "f.obj", figure())
    mc.segment(src, tmp_path / "o", parts=3, format="obj")
    aside, victim = _plant_journal(tmp_path, "moving", names, moved)
    mc.verify(tmp_path / "o")
    assert victim.read_text() == "bleibt" and (tmp_path / "o" / "part_0.obj").is_file() and aside.is_dir()


def test_foreign_commands_never_inherit_stdin(monkeypatch):
    """Ein versteckt gestarteter Dienst hat keine Konsole; erbte das Kindprogramm sein stdin, scheiterte der Start (Windows: WinError 6)."""
    seen = {}

    def spy(cmd, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ai.subprocess, "run", spy)
    ai._no_window_run(["x"], 5)
    assert seen["stdin"] is subprocess.DEVNULL and seen["shell"] is False and seen["capture_output"] is True and seen["timeout"] == 5


# ============================================================================= gezielte Tests (Mutanten aus der zweiten Pruefrunde)
# ----------------------------------------------------------------------------- Schluessel und Zugriff
@pytest.mark.parametrize("auth", ["Bearer tok2", "Bearer to", "Bearer tox", "Bearer ", "tok", "Basic tok"])
def test_http_key_must_match_exactly(api, auth):
    port, root = api
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("POST", "/v1/info", body=b"{}", headers={"Authorization": auth, "Content-Type": "application/json"})
    assert c.getresponse().status == 401


@pytest.mark.parametrize("path", ["/v1/tools", "/v1/health", "/v1/nope"])
def test_http_get_needs_the_key_too(api, path):
    port, _ = api
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("GET", path)
    assert c.getresponse().status == 401
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("GET", path, headers={"Authorization": "Bearer tok"})
    assert c.getresponse().status == (404 if path == "/v1/nope" else 200)


def test_http_refuses_chunked_and_survives_deeply_nested_json(api):
    port, _ = api
    status = None
    for _ in range(3):                                            # der Dienst antwortet und schliesst, bevor der Client fertig gesendet hat; unter Last kann die Verbindung dabei zurueckgesetzt werden
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            try:
                c.request("POST", "/v1/info", body=iter([b"{}"]), headers={"Authorization": "Bearer tok", "Content-Type": "application/json"}, encode_chunked=True)
            except (BrokenPipeError, ConnectionResetError):
                pass
            status = c.getresponse().status
            break
        except (ConnectionResetError, ConnectionAbortedError, http.client.RemoteDisconnected):
            continue
        finally:
            c.close()
    assert status == 411
    deep = b"[" * 200000 + b"]" * 200000
    assert post(port, "/v1/info", deep)[0] == 400
    assert post(port, "/v1/info", {"path": "x"})[0] == 400                                    # der Dienst lebt weiter


def test_sandbox_refuses_null_byte_and_non_text(tmp_path):
    sb = serve.Sandbox(tmp_path)
    for bad in ("a\0b", "", "   ", None, 5, ["x"]):
        with pytest.raises(mc.MeshcutError, match="Pfad fehlt"):
            sb.path(bad, "p")


def test_tools_run_one_at_a_time(tmp_path):
    import time
    state = {"active": 0, "peak": 0}

    class Slow(serve.Tools):
        def info(self, a):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.15)
            state["active"] -= 1
            return {}
    tools = Slow(serve.Sandbox(tmp_path))
    ts = [threading.Thread(target=serve.call_tool, args=(tools, "info", {})) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert state["peak"] == 1


def test_split_options_must_have_the_right_type(tmp_path):
    write(tmp_path / "f.obj", shapes.cube(1.0))
    tools = serve.Tools(serve.Sandbox(tmp_path))
    for key, bad in (("no_cap", "yes"), ("no_cap", 1), ("flip", "true"), ("verify", 0), ("wedges", True), ("wedges", "3"), ("grid", 2)):
        with pytest.raises(mc.MeshcutError):
            serve.call_tool(tools, "split", {"input": "f.obj", "outdir": "o", "plane": ["x=0.5"], key: bad})


# ----------------------------------------------------------------------------- MCP
def test_mcp_initialize_answers_with_a_version_it_speaks():
    tools = serve.Tools(serve.Sandbox(os.getcwd()))
    for ask, want in (("2024-11-05", "2024-11-05"), ("2025-03-26", "2025-03-26"), ("1999-01-01", serve.MCP_VERSIONS[0]), (None, serve.MCP_VERSIONS[0])):
        r = serve.mcp_handle(tools, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": ask} if ask else {}})
        assert r["result"]["protocolVersion"] == want


def test_mcp_any_tool_failure_is_an_error_result_not_the_end_of_the_service(tmp_path):
    class Broken(serve.Tools):
        def info(self, a):
            raise RuntimeError("kaputt\nzweite Zeile")
    out = serve.mcp_handle(Broken(serve.Sandbox(tmp_path)), {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "info", "arguments": {}}})
    assert out["result"]["isError"] is True and out["result"]["content"][0]["text"] == "Fehler: kaputt"


def test_mcp_stdio_survives_garbage_and_speaks_utf8(tmp_path):
    write(tmp_path / "f.obj", shapes.cube(1.0))
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="cp1252")                 # Windows-Codepage darf das Protokoll nicht verbiegen
    deep = "[" * 100000 + "]" * 100000
    msgs = [deep,
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "info", "arguments": {"path": "Pfad-äöü-€.obj"}}}, ensure_ascii=False),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"})]
    r = subprocess.run([sys.executable, str(ROOT / "kkmeshcut.py"), "serve", "--mcp", "--root", str(tmp_path)], input=("\n".join(msgs) + "\n").encode("utf-8"),
                       capture_output=True, env=env, timeout=120)
    out = [json.loads(l) for l in r.stdout.decode("utf-8").splitlines()]
    assert out[0]["error"]["code"] in (-32700, -32600) and out[-1] == {"jsonrpc": "2.0", "id": 2, "result": {}}
    assert out[1]["result"]["isError"]
    assert b"\r\n" not in r.stdout


# ----------------------------------------------------------------------------- Adapter
@pytest.mark.parametrize("url", ["http://example.com/x", "http://10.0.0.1:80/x", "https://", "https:///pfad", "ftp://localhost/x", "localhost:80"])
def test_check_url_refuses(url):
    with pytest.raises(mc.MeshcutError):
        ai.check_url(url, "X")


@pytest.mark.parametrize("url", ["https://example.com/x", "http://localhost:8080/x", "http://127.0.0.1/x", "http://[::1]:9/x"])
def test_check_url_accepts(url):
    ai.check_url(url, "X")


def test_check_url_names_the_reason():
    with pytest.raises(mc.MeshcutError, match="https://"):
        ai.check_url("http://example.com", "MESHCUT_X")
    with pytest.raises(mc.MeshcutError, match="keine Adresse"):
        ai.check_url("https:///pfad", "MESHCUT_X")


@pytest.mark.parametrize("bad", [np.array([0.5, 1.0]), np.array([np.nan, 1.0]), np.array([np.inf, 1.0]), np.array([-1, 1]), np.array([0, 1, 2]),
                                 np.array(["a", "b"]), np.zeros((2, 2), int)])
def test_validate_labels_refuses(bad):
    with pytest.raises(mc.MeshcutError):
        ai.validate_labels(bad, 2)


def test_validate_labels_accepts_whole_floats_and_bools():
    assert ai.validate_labels(np.array([0.0, 2.0]), 2).tolist() == [0, 2]
    assert ai.validate_labels(np.array([True, False]), 2).tolist() == [1, 0]


@pytest.mark.parametrize("labels", [[0.5] * 12, [-1] * 12, [0] * 11])
def test_http_backend_checks_the_map_it_gets(tmp_path, monkeypatch, fake_server, labels):
    Fake.reply = lambda *a: (200, {"labels": labels})
    monkeypatch.setenv("MESHCUT_SEGMENT_URL", fake_server)
    src = write(tmp_path / "f.obj", shapes.cube(1.0))
    with pytest.raises(mc.MeshcutError, match="ganze Zahlen|negativ|Werte"):
        mc.segment(src, tmp_path / "o", backend="http", parts=2)


def test_oversized_answers_are_refused(tmp_path, monkeypatch, fake_server):
    Fake.reply = lambda *a: (200, {"labels": [0, 1] * 6})
    monkeypatch.setattr(ai, "MAX_RESPONSE_BYTES", 10)
    monkeypatch.setenv("MESHCUT_SEGMENT_URL", fake_server)
    src = write(tmp_path / "f.obj", shapes.cube(1.0))
    with pytest.raises(mc.MeshcutError, match="zu gross"):
        mc.segment(src, tmp_path / "o", backend="http", parts=2)


class _Fixed(ai.Segmenter):
    name = "fest"

    def __init__(self, labels):
        self._labels = np.asarray(labels)

    def labels(self, mesh, parts=None):
        return self._labels


def test_segment_mesh_checks_what_a_backend_returns(monkeypatch):
    mesh = shapes.cube(1.0)
    n = len(mesh.F)
    monkeypatch.setitem(ai.SEGMENTERS, "fest", lambda: _Fixed([0] * n))
    with pytest.raises(mc.MeshcutError, match="nur ein Teil"):
        ai.segment_mesh(mesh, backend="fest")
    monkeypatch.setitem(ai.SEGMENTERS, "fest", lambda: _Fixed([i % 3 for i in range(n)]))
    with pytest.raises(mc.MeshcutError, match="3 Teile geliefert, gewuenscht waren 2"):
        ai.segment_mesh(mesh, backend="fest", parts=2)
    labels, _ = ai.segment_mesh(mesh, backend="fest")                       # ohne Wunsch ist jede Anzahl ab 2 recht
    assert labels.tolist() == [i % 3 for i in range(n)]


def test_components_must_find_the_wished_number():
    with pytest.raises(mc.MeshcutError, match="2 getrennte Huellen, nicht 3"):
        ai.ComponentsSegmenter().labels(two_cubes(), 3)
    assert len(set(ai.ComponentsSegmenter().labels(two_cubes(), 2).tolist())) == 2


def test_kmeans_starts_at_the_point_nearest_to_the_center_of_mass():
    X = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [10, 0, 0]])
    w = np.full(5, 0.2)                                                     # Schwerpunkt x = 3.2 -> naechster Punkt x = 3
    assert ai.start_centers(X, w, 3)[:, 0].tolist() == [3.0, 10.0, 0.0]
    assert ai.start_centers(X, np.array([1.0, 0, 0, 0, 0]), 2)[:, 0].tolist() == [0.0, 10.0]           # Gewichte zaehlen


def test_part_description_gives_area_shares():
    mesh = two_cubes()
    V = mesh.V.copy()
    V[:8] *= 2.0                                                            # erster Wuerfel doppelt so gross: 4 von 5 Teilen der Flaeche
    d = ai.describe_parts(mc.Mesh(V, mesh.F, "x"), ai.shell_labels(mesh))
    assert sorted(p["area_share"] for p in d) == [0.2, 0.8]


def test_llm_may_only_use_the_given_names():
    class Foreign(ai.Llm):
        def complete(self, system, user):
            return '{"0": "Fremd", "1": "Beine"}'
    mesh = two_cubes()
    with pytest.raises(mc.MeshcutError):
        ai.assign_names(mesh, ai.shell_labels(mesh), ["Kopf", "Beine"], Foreign())


def test_command_timeout_is_enforced(tmp_path, monkeypatch):
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(20)\n")
    monkeypatch.setenv("MESHCUT_SEGMENTER_CMD", '"%s" "%s"' % (sys.executable, script))
    with pytest.raises(mc.MeshcutError, match="liess sich nicht ausfuehren"):
        ai.CommandSegmenter(timeout=1).labels(shapes.cube(1.0), 2)


def test_command_is_never_run_through_a_shell(tmp_path, monkeypatch):
    script = tmp_path / "ok.py"
    script.write_text("import sys, numpy as np\nnp.save(sys.argv[1], np.array([0] * 6 + [1] * 6))\n")
    marker = tmp_path / "marker.txt"
    monkeypatch.setenv("MESHCUT_SEGMENTER_CMD", '"%s" "%s" {output} & echo x > "%s"' % (sys.executable, script, marker))
    ai.CommandSegmenter().labels(shapes.cube(1.0), 2)
    assert not marker.exists()


# ----------------------------------------------------------------------------- PLY (Fehler- und Sonderfaelle)
HEAD = b"ply\nformat %s 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nelement face %d\nproperty list uchar int vertex_indices\nend_header\n"


@pytest.mark.parametrize("data,match", [
    (b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list foo int vertex_indices\nend_header\n", "unbekannter Typ"),
    (b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty bar z\nend_header\n", "unbekannter Typ"),
    (b"ply\nformat ascii 1.0\nelement vertex -3\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n", "negative Anzahl"),
    (HEAD % (b"binary_little_endian", 3, 1) + struct.pack("<3f", 0, 0, 0), "abgeschnitten"),                                  # Eckpunkte fehlen
    (HEAD % (b"binary_little_endian", 3, 1) + struct.pack("<9fB3i", 0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0, 1, 9), "Eckpunkt, den es nicht gibt"),
])
def test_ply_refuses_broken_files(data, match):
    with pytest.raises(mc.MeshcutError, match=match):
        mc.read_meshes_bytes("ply", data, "x")


def test_ply_binary_mixed_face_sizes_use_the_slow_path():
    verts = struct.pack("<12f", 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0)
    faces = struct.pack("<B3i", 3, 0, 1, 2) + struct.pack("<B4i", 4, 0, 1, 2, 3)
    got = mc.read_meshes_bytes("ply", HEAD % (b"binary_little_endian", 4, 2) + verts + faces, "x")[0]
    got = got.V[got.F]
    assert got.shape == (3, 3, 3)                                            # 1 Dreieck + 1 Viereck (= 2 Dreiecke)
    assert got[0].tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 0]]


def test_ply_faces_with_fewer_than_three_corners_are_skipped():
    data = (b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 3\n"
            b"property list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n2 0 1\n1 0\n3 0 1 2\n")
    got = mc.read_meshes_bytes("ply", data, "x")[0]
    assert got.F.shape == (1, 3)


def test_ply_short_faces_are_skipped_but_their_indices_are_checked():
    head = (b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 2\n"
            b"property list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
    with pytest.raises(mc.MeshcutError, match="Eckpunkt, den es nicht gibt"):
        mc.read_meshes_bytes("ply", head + b"2 0 99\n", "x")


def test_serve_option_helper_checks_types():
    for kind, bad, good in (("bool", "yes", True), ("bool", 1, False), ("int", True, 3), ("int", "3", 0), ("str", 5, "x"), ("strlist", "x", ["a"]), ("strlist", [1], [])):
        with pytest.raises(mc.MeshcutError):
            serve._opt({"k": bad}, "k", kind)
        assert serve._opt({"k": good}, "k", kind) == good
    assert serve._opt({}, "k", "bool") is None and serve._opt({"k": None}, "k", "int") is None


def test_mcp_loop_survives_a_handler_that_raises(monkeypatch):
    import io

    def boom(tools, msg):
        raise RuntimeError("innen kaputt")
    monkeypatch.setattr(serve, "mcp_handle", boom)
    out = io.StringIO()
    serve.serve_mcp(None, io.StringIO('{"jsonrpc":"2.0","id":3,"method":"ping"}\n[]\n'), out)
    got = [json.loads(l) for l in out.getvalue().splitlines()]
    assert [g["error"]["code"] for g in got] == [-32603, -32603] and got[0]["id"] == 3 and got[1]["id"] is None


def test_mcp_stdio_decodes_utf8_whatever_the_console_codepage(tmp_path):
    name = "\u65e5\u672c-\u0434.obj"                                          # nicht in cp1252
    write(tmp_path / name, shapes.cube(1.0))
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="cp1252")
    msg = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "info", "arguments": {"path": name}}}, ensure_ascii=False)
    r = subprocess.run([sys.executable, str(ROOT / "kkmeshcut.py"), "serve", "--mcp", "--root", str(tmp_path)], input=(msg + "\n").encode("utf-8"),
                       capture_output=True, env=env, timeout=120)
    out = json.loads(r.stdout.decode("utf-8"))
    assert out["result"]["isError"] is False, out


def test_ply_winding_is_kept_on_the_fast_path():
    verts = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    got = mc.read_meshes_bytes("ply", HEAD % (b"binary_little_endian", 3, 1) + verts + struct.pack("<B3i", 3, 0, 1, 2), "x")[0]
    assert got.V[got.F][0].tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]


def test_label_files_never_unpickle(tmp_path):
    p = tmp_path / "l.npy"
    np.save(p, np.array([{"a": 1}, 2], dtype=object), allow_pickle=True)
    with pytest.raises(mc.MeshcutError, match="unlesbar"):
        mc.read_label_file(p)


# ----------------------------------------------------------------------------- Runde 3: abgeleitete Pfade und Geschwisterbefehle
@pytest.mark.parametrize("tool", ["split", "segment"])
@pytest.mark.parametrize("fmt", ["stl/../../../outside/secret.txt", "exe", "", 5, "STL "])
def test_format_is_checked_in_every_tool(tmp_path, tool, fmt):
    root = tmp_path / "root"
    root.mkdir()
    write(root / "f.obj", shapes.cube(1.0))
    tools = serve.Tools(serve.Sandbox(root))
    args = {"input": "f.obj", "outdir": "o", "format": fmt, "plane": ["x=0.5"]} if tool == "split" else {"input": "f.obj", "outdir": "o", "format": fmt, "parts": 2}
    with pytest.raises(mc.MeshcutError, match="moeglich sind"):
        serve.call_tool(tools, tool, args)
    assert not (root / "o").exists() or not any((root / "o").iterdir())


def test_core_refuses_unknown_formats_too(tmp_path):
    write(tmp_path / "f.obj", shapes.cube(1.0))
    with pytest.raises(mc.MeshcutError, match="gibt es nicht"):
        ai.segment_file(tmp_path / "f.obj", tmp_path / "o", parts=2, fmt="../x")


def test_split_labels_must_stay_inside_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    write(root / "f.obj", shapes.cube(1.0))
    (tmp_path / "aussen.txt").write_text("\n".join(["0", "1"] * 6))
    tools = serve.Tools(serve.Sandbox(root))
    with pytest.raises(mc.MeshcutError, match="ausserhalb"):
        serve.call_tool(tools, "split", {"input": "f.obj", "outdir": "o", "labels": str(tmp_path / "aussen.txt")})


@pytest.mark.parametrize("src", ["", 0, [], False])
def test_verify_tool_refuses_a_malformed_source_instead_of_skipping_it(tmp_path, src):
    write(tmp_path / "f.obj", shapes.cube(1.0))
    tools = serve.Tools(serve.Sandbox(tmp_path))
    serve.call_tool(tools, "split", {"input": "f.obj", "outdir": "o", "plane": ["x=0.5"]})
    with pytest.raises(mc.MeshcutError):
        serve.call_tool(tools, "verify", {"parts": "o", "source": src})


def test_verify_tool_compares_with_the_given_source(tmp_path):
    write(tmp_path / "f.obj", shapes.cube(1.0))
    write(tmp_path / "g.obj", shapes.cube(2.0))
    tools = serve.Tools(serve.Sandbox(tmp_path))
    serve.call_tool(tools, "split", {"input": "f.obj", "outdir": "o", "plane": ["x=0.5"]})
    good = serve.call_tool(tools, "verify", {"parts": "o", "source": "f.obj"})
    bad = serve.call_tool(tools, "verify", {"parts": "o", "source": "g.obj"})
    assert good["ok"] and "quelle" in [c["name"] for c in good["checks"]]
    assert not bad["ok"] and "quelle" in [c["name"] for c in bad["checks"] if not c["ok"]]


def test_verify_notices_a_reordered_source_file(tmp_path):
    mesh = shapes.cube(2.0, sub=2)
    src = write_obj_ai(tmp_path / "f.obj", mesh)
    mc.split(src, tmp_path / "o", plane=["x=1"])
    perm = np.arange(len(mesh.F))[::-1]
    write_obj_ai(src, mc.Mesh(mesh.V, mesh.F[perm]))             # gleiche Dreiecke, andere Reihenfolge
    r = mc.verify(tmp_path / "o", src)
    failed_names = [c[0] for c in r["checks"] if not c[1]]
    assert "quelle-reihenfolge" in failed_names and "quelle" not in failed_names


def write_obj_ai(path, mesh):
    Path(path).write_bytes(mc.mesh_bytes("obj", mesh.V, mesh.F))
    return Path(path)


def test_sidecar_without_order_hash_is_refused(tmp_path):
    src = write(tmp_path / "f.obj", shapes.cube(1.0))
    mc.split(src, tmp_path / "o", plane=["x=0.5"])
    p = tmp_path / "o" / "meshcut.json"
    side = json.loads(p.read_text())
    del side["source"]["order_hash"]
    p.write_text(json.dumps(side))
    with pytest.raises(mc.MeshcutError):
        mc.verify(tmp_path / "o")
    with pytest.raises(ValueError, match="source"):
        mc.check_sidecar(side)                                   # schon die Pruefung der Begleitdatei verlangt das Feld


def test_sequence_hash_depends_on_precision():
    m = shapes.cube(1.0)
    assert mc.sequence_hash(m.V, m.F, "float64") != mc.sequence_hash(m.V, m.F, "float32")


def test_sandbox_follows_links_out_of_the_root_never(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    out = tmp_path / "aussen"
    out.mkdir()
    link = root / "ziel"
    if os.name == "nt":                                          # Junction braucht keine Sonderrechte
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(out)], capture_output=True, stdin=subprocess.DEVNULL, creationflags=0x08000000)
        if r.returncode:
            pytest.skip("Junction nicht moeglich")
    else:
        try:
            os.symlink(out, link, target_is_directory=True)
        except OSError:
            pytest.skip("keine Symlinks erlaubt")
    with pytest.raises(mc.MeshcutError, match="ausserhalb"):
        serve.Sandbox(root).path("ziel/x.obj", "p")


# ----------------------------------------------------------------------------- Runde 4: Dienst
def test_mcp_overlong_line_is_refused_and_the_loop_goes_on(monkeypatch):
    import io
    monkeypatch.setattr(serve, "MAX_BODY", 200)
    inp = io.StringIO("x" * 1000 + "\n" + '{"jsonrpc":"2.0","id":5,"method":"ping"}\n')
    out = io.StringIO()
    serve.serve_mcp(None, inp, out)
    got = [json.loads(l) for l in out.getvalue().splitlines()]
    assert got[0]["error"]["code"] == -32600 and "zu lang" in got[0]["error"]["message"] and got[1]["result"] == {} and got[1]["id"] == 5


@pytest.mark.parametrize("host,code", [("localhost", 200), ("localhost:8765", 200), ("127.0.0.1", 200), ("127.0.0.1:1", 200), ("[::1]", 200), ("[::1]:8765", 200),
                                        ("evil.example", 401), ("evil.example:80", 401), ("localhost.evil.example", 401), ("", 401)])
def test_local_host_header_forms(tmp_path, host, code):
    srv = serve.make_http_server(serve.Tools(serve.Sandbox(tmp_path)), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        import socket
        s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=10)
        s.sendall(("GET /v1/health HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n" % host).encode())
        got = b""
        while True:
            d = s.recv(4096)
            if not d:
                break
            got += d
        assert got.split(b" ", 2)[1] == str(code).encode()
    finally:
        srv.shutdown()
        srv.server_close()
