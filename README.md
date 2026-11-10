# K&K meshcut

![license](https://img.shields.io/badge/license-MIT-blue) ![python](https://img.shields.io/badge/python-3.9%2B-informational) ![round trip](https://img.shields.io/badge/Rundlauf-bitgenau-brightgreen) ![tests](https://github.com/CalyieX/meshcut/actions/workflows/ci.yml/badge.svg)

[![Unterstützen auf Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/kkwonderland)

**3D-Netze zerlegen und wieder zusammensetzen, mit Beweis, dass nichts verloren geht.**
*Cut 3D meshes into parts and put them back together, with a round-trip proof.*

**English in short:** a small command-line tool and Python library that splits triangle meshes (STL, OBJ, GLB, 3MF) by planes, grids, print-bed size, wedges around an axis, or along a per-triangle label map, caps every cut with a watertight lid, and joins the parts back. With the `meshcut.json` sidecar the join restores the *original triangles* and a SHA-256 over the triangle set proves it (bit-exact in 64-bit formats, after float32 rounding in STL/GLB). It also runs in the browser (Pyodide), nothing is uploaded. Everything below is in German.

**Auf einen Blick**

- **Rundlauf mit Beweis:** `split` → `join` ergibt dieselben Dreiecke (SHA-256 über die Dreiecksmenge), `verify` prüft zusätzlich Volumen, Fläche, wasserdicht und Euler-Zahl.
- **Viele Arten zu schneiden:** Ebenen (`x=12.5`, schräg), Gitter (`--grid 2x2x3`), Druckraum (`--max 220x220x250`), Keile um eine Achse (`--wedges 8`) und eine Zuordnungskarte (ein Wert pro Dreieck, z. B. aus einer KI-Segmentierung).
- **Sauber geschnitten:** Neue Eckpunkte entstehen genau **einmal pro Kante** und werden von beiden Seiten geteilt: keine Spalten, keine T-Kreuzungen. Jeder Schnitt bekommt einen Deckel (auch mit Löchern und Inseln).
- **Läuft überall:** Windows, Mac, Linux; nur numpy als Abhängigkeit; und [im Browser](https://calyiex.github.io/meshcut/) (auch iPhone), ohne Upload.

Gebaut von K&K AI als Schwesterwerkzeug von [pixelcut](https://github.com/CalyieX/pixelcut) (dort Bilder, hier Netze).

---

## Installation

```bash
pip install kk-meshcut        # Befehl: meshcut, Python-Modul: kkmeshcut
```

(Der Name `meshcut` auf PyPI gehört einem fremden Projekt, deshalb `kk-meshcut`. Der Befehl heißt trotzdem `meshcut`.)

## Schnellstart

```bash
# Ein Torus, an zwei Ebenen in vier Teile zerlegt, danach wieder zusammengesetzt
meshcut split examples/torus.stl teile --plane x=0 --plane z=0 --verify
meshcut join teile zurueck.obj
```

`--verify` liest die *gespeicherten* Dateien neu ein und meldet:

```
OK     hash        zusammengesetzt 9e2b374f3c40... = Hash der Quelle (float32)
OK     wasserdicht jedes Teil wasserdicht
OK     euler       Euler-Zahl je Teil gerade
OK     volumen     Summe der Teile 19069.6383337, Original 19069.6383337
OK     flaeche     Flaeche ohne Deckel 5501.38007, Original 5501.38007
OK     quelle      Quelldatei 9e2b374f3c40... = Hash in der Begleitdatei
OK     hausdorff   Hausdorff (Eckpunkte, ohne Begleitdatei zusammengesetzt) 0, erlaubt 7.76e-05
```

## Befehle

| Befehl | Was er tut |
|---|---|
| `meshcut split EIN AUSORDNER ...` | zerlegt ein Netz, schreibt die Teile und `meshcut.json` |
| `meshcut join ORDNER AUS.stl` | setzt die Teile eines Ordners bitgenau wieder zusammen (mit `meshcut.json`) |
| `meshcut join teil1.stl teil2.stl AUS.stl` | ohne Begleitdatei: Näherung (Deckelpaare weg, Dreiecke bleiben fein) |
| `meshcut verify ORDNER [--source ORIGINAL]` | prüft den Rundlauf der gespeicherten Dateien |
| `meshcut info DATEI` | Eckpunkte, Dreiecke, Volumen, Fläche, wasserdicht, Euler-Zahl, Hash |
| `meshcut labels DATEI KARTE.npy --grid 2x2x2` | schreibt eine Zuordnungskarte (Teilnummer pro Dreieck) für ein Gitter |

### Arten zu zerlegen (`split`)

```bash
meshcut split m.stl out --plane x=12.5                  # eine Ebene (x, y oder z = Wert)
meshcut split m.stl out --plane x=0 --plane 1,1,0,4     # mehrere, auch schräg: a,b,c,d  =  a*x+b*y+c*z = d
meshcut split m.stl out --grid 2x2x3                    # Gitter über den Begrenzungsquader, Teilnummer = Zelle
meshcut split m.stl out --max 220x220x250               # so viele gleichmäßige Schnitte, dass jedes Teil in den Druckraum passt
meshcut split m.stl out --wedges 8 --axis z             # 8 Keile um die z-Achse (gerade Zahl; --center 10,5 verschiebt die Achse)
meshcut split m.stl out --labels karte.npy              # Zuordnungsmodus: Geometrie bleibt unverändert
```

Weitere Optionen: `--format stl|obj|3mf|glb` (3MF und GLB legen alle Teile in **eine** Datei), `--no-cap` (ohne Deckel, für offene Netze), `--flip` (Dreiecke umdrehen, wenn das Netz nach innen orientiert ist), `--snap` (siehe unten), `--verify`.

**Zuordnungskarte:** eine Datei (`.npy`, `.json` oder `.csv`) mit **einer ganzen Zahl pro Dreieck** (Teilnummer, Lücken erlaubt, höchstens 1024 Teile). So lassen sich Ergebnisse einer Netz-Segmentierung (z. B. PartField, SAMesh, P3-SAM) verwenden: einfach ihre Dreiecks-Labels als `.npy` speichern. meshcut zeichnet selbst nur Gitter-Karten (`meshcut labels`).

## Was genau garantiert wird

Zwei Stufen, die man nicht vermischen darf:

**Zuordnungsmodus.** Die Eckpunkte und Dreiecke bleiben unverändert, es werden nur Dreiecke auf Teile verteilt. Setzt man die Teile wieder zusammen, kommt dieselbe Dreiecksmenge heraus: gleicher SHA-256, **bitgenau**.

**Schnittmodus.** Dreiecke werden an Ebenen zerlegt. Der Rundlauf ist trotzdem exakt, weil meshcut in `meshcut.json` für jedes Teildreieck die Nummer seines Quelldreiecks und für jedes zerschnittene Dreieck dessen Original-Koordinaten (64 Bit) speichert. `join` entfernt die Deckel und setzt die Original-Dreiecke wieder ein; der Hash stimmt mit dem der Quelle überein. Ohne Begleitdatei bleibt die Geometrie gleich (Hausdorff-Abstand ~ Rundungsfehler), nur die Dreiecke bleiben so fein, wie der Schnitt sie gemacht hat.

**Dateiformat und Genauigkeit.** STL und GLB speichern 32 Bit: der Hash wird dann nach Rundung auf float32 gebildet (steht in `meshcut.json` als `precision`). Wer **wirklich 64 Bit** will, nimmt `--format obj` oder `3mf` (Text mit 17 Stellen, ohne Verlust).

**Rundlauf-Kriterien des Schnittmodus** (alle in `verify`): Volumen aller Teile = Volumen des Originals, Fläche ohne Deckel = Originalfläche, jedes Teil wasserdicht (jede Kante genau zweimal, gegenläufig), Euler-Zahl gerade, Hash gleich, Hausdorff-Abstand der ohne Begleitdatei zusammengesetzten Teile zur Quelle. Erlaubt sind 1e-9 relativ bei 64 Bit und 1e-4 bei 32 Bit.

### Wie geschnitten wird

1. Jeder Eckpunkt bekommt ein Vorzeichen zur Ebene (davor / dahinter / auf der Ebene). Weil das Vorzeichen pro Eckpunkt feststeht, entscheiden beide Dreiecke an einer Kante gleich.
2. Je Kante, die die Ebene kreuzt, entsteht **genau ein** neuer Eckpunkt (Endpunkte sortiert, `p = a + t·(b−a)`, `t = da/(da−db)`), beide Seiten verweisen darauf.
3. Der Rand jedes Teils liegt danach in der Ebene. Daraus entstehen geschlossene Umrisse; Außenumrisse und Löcher werden erkannt, Löcher über Brückenkanten angebunden, die Fläche mit Ohrenschnitt trianguliert. Alle Randpunkte bleiben erhalten, deshalb schließt der Deckel ohne Spalt.
4. Der Deckel des einen Teils ist das gespiegelte Gegenstück des anderen. Bei mehreren Ebenen werden auch Deckel weiter zerschnitten und bleiben dabei Paare.
5. **Dreiecke, die in der Ebene liegen**, gehören zu der Seite, hinter der das Material liegt. Flächenlose Dreiecke in der Ebene gehen auf die Seite ihrer Nachbarn.
6. `--snap` (Standard 1e-12): Eckpunkte, die nur um Rundungsrauschen neben der Ebene liegen (z. B. `cos(90°)` bei einer Kugel), gelten als *auf* der Ebene. Das verschiebt keinen Eckpunkt, es vermeidet nur Haarrisse-Dreiecke. `--snap 0` nimmt das exakte Vorzeichen.

## Python

```python
import kkmeshcut as mc

side = mc.split("examples/torus.stl", "teile", plane=["x=0", "z=0"], verify=True)   # dict aus meshcut.json
mc.join("teile", "zurueck.obj")
r = mc.verify("teile", source="examples/torus.stl")      # {'ok': True, 'checks': [...], ...}
print(mc.info("examples/torus.stl")["volume"])
```

Die Schnittstelle führt dieselben Befehle aus wie die Kommandozeile (gleiche Optionen mit Unterstrich: `no_cap=True`), schweigt auf stdout und stderr und wirft `mc.MeshcutError` statt das Programm zu beenden. Tiefer unten liegen `Mesh`, `split_planes`, `split_labels`, `triangle_hash`, `mesh_stats` für eigene Abläufe mit numpy-Arrays.

## Im Browser

**[calyiex.github.io/meshcut](https://calyiex.github.io/meshcut/)** – derselbe Kern läuft in Pyodide/WebAssembly, nichts wird hochgeladen. Datei wählen, Schnitt einstellen, Teile als ZIP speichern; zum Zusammensetzen alle Teile samt `meshcut.json` wieder auswählen. Bis 40 MB je Datei. Bei jedem Commit prüft das CI die Seite in WebKit (Safari/iPhone-Engine), Chromium und Firefox: Zerlegen und Zusammensetzen im Browser müssen den Hash der Quelle ergeben.

## Grenzen (ehrlich)

- **Nur Geometrie.** UV-Koordinaten, Farben, Normalen und Materialien werden nicht übernommen (v0.1). Eckpunkte mit gleichen Koordinaten werden beim Einlesen verschmolzen (GLB-Nähte, STL-Dreiecksuppe).
- **Deckel brauchen ein wasserdichtes, nach außen orientiertes Netz.** Sonst bricht `split` mit einer klaren Meldung ab (`--no-cap` schneidet trotzdem, `--flip` dreht die Orientierung).
- **Keile nur in gerader Zahl.** Ungerade Keilzahlen bräuchten Halbebenen.
- **Stifte und Passungen** (Dübel, Zapfen/Loch mit Spiel) gibt es in v0.1 noch nicht. manifold3d wird nur in den Tests als unabhängige Gegenprobe benutzt (Volumen der Teile gegen `split_by_plane`), nicht zur Laufzeit.
- **Triangulierung der Deckel** ist gültig und deckungsgleich, aber nicht „schön“: Ohrenschnitt erzeugt gern lange schmale Dreiecke. Für den Druck unkritisch, für Simulation ungeeignet. Läuft der Ohrenschnitt in einen Notfall (Punkte exakt aufeinander), meldet `split` das.
- **Geschwindigkeit.** Der Kern ist numpy; die Deckel-Triangulierung ist Python. Rund 100 000 Dreiecke in unter einer Sekunde, Deckel mit mehreren tausend Randpunkten brauchen Sekunden.
- Nichtmanifold-Eckpunkte, an denen sich zwei Teile nur in einem Punkt berühren, sind erlaubt; `verify` zählt sie für die Euler-Zahl getrennt.

## Entwicklung

```bash
pip install -e ".[check]"     # pytest, hypothesis, trimesh, manifold3d (Gegenproben)
python -m pytest -q tests
MESHCUT_EXAMPLES=300 python -m pytest -q tests/test_property.py     # gründlicher
```

Die Tests prüfen unter anderem: Zufallsebenen, Ebenen durch Eckpunkte/Kanten/Flächen (hypothesis), Dreiecke in der Ebene, Keile um eine Achse, Torus/Hohlkugel (Löcher und Inseln im Deckel), alle vier Formate, Absturz-Injektion beim Schreiben, Fremddateien im Ausgabeordner und den Vergleich mit trimesh und manifold3d. Die Browser-Version liegt als Kopie in `docs/kkmeshcut.py`; ein Test stellt sicher, dass beide Dateien gleich sind.

Lizenz: MIT, © K&K AI.
