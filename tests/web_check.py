"""Browser-Pruefung der Web-Version (docs/): Zerlegen und Zusammensetzen im echten Browser muss den Hash der Quelle ergeben.

    python tests/web_check.py webkit      # die Engine von iPhone/Safari
    python tests/web_check.py chromium
    python tests/web_check.py firefox

Braucht `pip install playwright numpy` und `playwright install <engine>`. Lokaler Webserver, Pyodide kommt vom CDN.
"""
import base64
import functools
import http.server
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from playwright.sync_api import sync_playwright  # noqa: E402

import kkmeshcut as mc  # noqa: E402
import shapes  # noqa: E402

DOCS = ROOT / "docs"

JS = """
async ([b64, how, value, fmt]) => {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  for (let i = 0; i < 240 && document.getElementById("go").disabled; i++) await wait(500);
  if (document.getElementById("go").disabled) return {error: "Programm wurde nie bereit: " + document.getElementById("status").textContent};
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const f = document.getElementById("files");
  const pick = (cmd) => { document.getElementById("cmd").value = cmd; document.getElementById("cmd").dispatchEvent(new Event("change")); };
  const press = async () => { document.getElementById("go").click(); await wait(300); for (let i = 0; i < 480 && document.getElementById("go").disabled; i++) await wait(250); };
  pick("split");
  document.getElementById("how").value = how; document.getElementById("how").dispatchEvent(new Event("change"));
  document.getElementById(how === "wedges" ? "wedges" : how).value = value;
  document.getElementById("format").value = fmt;
  let dt = new DataTransfer(); dt.items.add(new File([bytes], "in.stl")); f.files = dt.files;
  await press();
  const splitStatus = document.getElementById("status").textContent;
  const names = py.FS.readdir("/work/out/parts").filter((n) => !n.startsWith("."));
  const parts = names.map((n) => [n, py.FS.readFile("/work/out/parts/" + n)]);
  pick("join");
  dt = new DataTransfer(); for (const [n, b] of parts) dt.items.add(new File([b], n)); f.files = dt.files;
  await press();
  const joinStatus = document.getElementById("status").textContent;
  const joined = py.FS.readdir("/work/out").filter((n) => n.startsWith("joined."));
  let outB64 = "";
  if (joined.length) {
    const jb = py.FS.readFile("/work/out/" + joined[0]);
    let bin = "";
    for (let i = 0; i < jb.length; i += 8192) bin += String.fromCharCode.apply(null, jb.subarray(i, i + 8192));
    outB64 = btoa(bin);
  }
  return {names, splitStatus, joinStatus, joined, outB64};
}
"""


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "chromium"
    how = sys.argv[2] if len(sys.argv) > 2 else "grid"             # grid | plane | wedges
    value, expect_parts = {"grid": ("2x2x1", 4), "plane": ("x=0.4; z=0.3", 4), "wedges": ("4", 4)}[how]
    fmt = "stl" if how != "wedges" else "obj"
    m = shapes.sphere(14, 9)
    data = base64.b64encode(mc.mesh_bytes("stl", m.V, m.F)).decode()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(DOCS)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d/" % srv.server_address[1]
    with sync_playwright() as p:
        browser = getattr(p, engine).launch()
        page = browser.new_page()
        page.set_default_timeout(300_000)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        res = page.evaluate(JS, [data, how, value, fmt])
        browser.close()
    srv.shutdown()
    print(engine, how, {k: v for k, v in res.items() if k != "outB64"})
    if errors:
        print("Seitenfehler:", errors)
    ok = (not res.get("error") and len(res.get("names", [])) == expect_parts + 1 and "FEHLER" not in res["splitStatus"]
          and "Hash" in res["joinStatus"] and "= Quelle" in res["joinStatus"] and res["joined"])
    if not ok:
        sys.exit("FEHLGESCHLAGEN: die Web-Version setzt in " + engine + " nicht bitgenau zusammen")
    # die geschriebene Ergebnisdatei selbst pruefen (nicht nur die Meldung der Seite): sie muss dieselbe Dreiecksmenge enthalten
    back = mc.read_meshes_bytes("stl", base64.b64decode(res["outB64"]), "joined")[0]
    want = mc.triangle_hash(m.V, m.F, "float32")
    got = mc.triangle_hash(back.V, back.F, "float32")
    if got != want:
        sys.exit("FEHLGESCHLAGEN: die Ergebnisdatei aus %s hat %d Dreiecke und Hash %s... statt %s..." % (engine, len(back.F), got[:12], want[:12]))
    print("OK: %s (%s) - Zerlegen und Zusammensetzen im Browser sind bitgenau" % (engine, how))


if __name__ == "__main__":
    main()
