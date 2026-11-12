# K&K meshcut

![license](https://img.shields.io/badge/license-MIT-blue) ![python](https://img.shields.io/badge/python-3.9%2B-informational) ![round trip](https://img.shields.io/badge/Rundlauf-bitgenau-brightgreen) ![tests](https://github.com/CalyieX/meshcut/actions/workflows/ci.yml/badge.svg)

[![Unterstützen auf Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/kkwonderland)

**3D-Netze zerlegen und wieder zusammensetzen, mit Beweis, dass nichts verloren geht.**
*Cut 3D meshes into parts and put them back together, with a round-trip proof.*

**English in short:** a small command-line tool and Python library that splits triangle meshes (STL, OBJ, PLY, GLB, 3MF) by planes, grids, print-bed size, wedges around an axis, or along a per-triangle label map, caps every cut with a watertight lid, and joins the parts back. With the `meshcut.json` sidecar the join restores the *original triangles* and a SHA-256 over the triangle set proves it (bit-exact in 64-bit formats, after float32 rounding in STL/GLB). `meshcut segment` finds (and names) parts automatically - built-in methods, a local model, or any hosted service behind one adapter - and the result is always a per-triangle label map, so the exact join and the proof still apply. An MCP server and an HTTP API expose the same tools to agents. It also runs in the browser (Pyodide), nothing is uploaded. Everything below is in German.

**Auf einen Blick**

- **Rundlauf mit Beweis:** `split` → `join` ergibt dieselben Dreiecke (SHA-256 über die Dreiecksmenge), `verify` prüft zusätzlich Volumen, Fläche, wasserdicht und Euler-Zahl.
- **Viele Arten zu schneiden:** Ebenen (`x=12.5`, schräg), Gitter (`--grid 2x2x3`), Druckraum (`--max 220x220x250`), Keile um eine Achse (`--wedges 8`) und eine Zuordnungskarte (ein Wert pro Dreieck, z. B. aus einer KI-Segmentierung).
- **Sauber geschnitten:** Neue Eckpunkte entstehen genau **einmal pro Kante** und werden von beiden Seiten geteilt: keine Spalten, keine T-Kreuzungen. Jeder Schnitt bekommt einen Deckel (auch mit Löchern und Inseln).
- **KI als Zulieferer, Beweis bleibt:** `meshcut segment` findet Teile (eingebaut, lokales Modell oder gehosteter Dienst) und benennt sie auf Wunsch (`--names Kopf,Arme,Beine` oder `--prompt "trenne Kopf, Arme, Beine"`). Egal woher die Karte kommt: sie läuft durch denselben Weg, `join` und `verify` bleiben exakt.
- **Ein Satz genügt:** `regions` nummeriert die Elemente eines Modells, `remove` und `swap` entfernen oder ersetzen eines (Loch zu, Material, UV und Normalen bleiben), und eine Prüfzeile sagt, was sich geändert hat und dass der Rest identisch ist. Kein Blender, keine Parameter.
- **Für Agenten gebaut:** MCP-Server (`meshcut serve --mcp`) und HTTP-API (`meshcut serve --http 8765`), beide nur in einem Wurzelordner.
- **Läuft überall:** Windows, Mac, Linux; nur numpy als Abhängigkeit; und [im Browser](https://calyiex.github.io/meshcut/) (auch iPhone), ohne Upload.

Gebaut von K&K AI als Schwesterwerkzeug von [pixelcut](https://github.com/CalyieX/pixelcut) (dort Bilder, hier Netze).

---

## Leitidee

meshcut ist ein KI-Werkzeug mit Harness, kein Modellierprogramm zum Selberklicken. Niemand soll in Blender-Manier stundenlang Löcher flicken, Normalen glätten und Materialien neu zuweisen: Der Assistent nennt das Ziel, meshcut misst am Modell, entscheidet selbst und liefert ein Ergebnis, bei dem man keinen Unterschied zum Original sieht (Schattierung, UV und Materialien bleiben). Jede neue Funktion folgt dem: ein Befehl pro Aufgabe, gute Voreinstellungen statt Parameterlisten, eine Prüfzeile zum Schluss.

## Warum meshcut? Drei Alleinstellungsmerkmale

Stand der Recherche: 10.11.2026. Die Cloud-Dienste (Meshy Auto Split, Tripo) verändern oder erfinden Geometrie (Verstärkung, „Part Completion“) und verlangen ein Bezahlkonto; KI-Modelle (PartField, SAMesh, P3-SAM) liefern nur Labels ohne Schnitt; Slicer-Werkzeuge (PrusaSlicer, Bambu Studio) und Blender schneiden von Hand und beweisen nichts.

1. **Exakter Schnitt mit Beweis.** Neue Eckpunkte genau einmal pro Kante, ein wasserdichter Deckel je Schnitt, das Original-Netz bleibt außerhalb des Schnitts unverändert. Die Teile sind einzeln geschlossen, und `verify` weist es nach (wasserdicht, Euler, Volumen, Fläche, Deckel auf den Ebenen).
2. **Rundlauf bitgenau.** `join` stellt die *ursprünglichen Dreiecke* wieder her; ein SHA-256 über die Dreiecksmenge, ein zweiter über ihre Reihenfolge und ein Hash je Teil belegen es. Bricht das Schreiben mittendrin ab (Fehler, `kill -9`, Stromausfall), bleibt der Ordner entweder ganz alt oder ganz neu; Dateien werden vor dem Umbenennen mit `fsync` gesichert (die Ordnereinträge nur unter Linux/macOS, unter Windows verlässt sich meshcut auf NTFS).
3. **Die Zuordnungskarte als offene Schnittstelle.** Jedes Verfahren, das jedem Dreieck eine Teilnummer gibt, kann meshcut füttern: eingebaut, als Befehl für ein lokales Modell, als gehosteter Dienst. Danach ist es dasselbe wie ein Schnitt: exakt zusammensetzbar und nachprüfbar. So bleibt die KI austauschbar, der Beweis bleibt.

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
OK     hausdorff   Hausdorff (Eckpunkte, ohne Begleitdatei zusammengesetzt) 0, erlaubt 0.000103
```

## Befehle

| Befehl | Was er tut |
|---|---|
| `meshcut split EIN AUSORDNER ...` | zerlegt ein Netz, schreibt die Teile und `meshcut.json` |
| `meshcut join ORDNER AUS.stl` | setzt die Teile eines Ordners bitgenau wieder zusammen (mit `meshcut.json`) |
| `meshcut join teil1.stl teil2.stl AUS.stl` | ohne Begleitdatei: Näherung (Deckelpaare weg, Dreiecke bleiben fein) |
| `meshcut verify ORDNER [--source ORIGINAL]` | prüft den Rundlauf der gespeicherten Dateien |
| `meshcut segment EIN AUSORDNER [--parts N] [--names a,b,c] [--prompt "…"] [--backend …]` | findet Teile automatisch (und benennt sie), schreibt sie als Zuordnungskarte, prüft den Rundlauf |
| `meshcut regions EIN AUSORDNER` | nummeriert die Elemente (Teile, Objekte, Gruppen, Materialien): `regions.png` mit Nummern, `regions.json` mit Punkt, Rahmen, Fläche, Material |
| `meshcut remove EIN --region N` | entfernt ein Element, schließt das Loch sauber (Material, UV, Normalen der Umgebung), schreibt `EIN_edit.obj` |
| `meshcut swap EIN --region N [--shape box\|sphere\|cylinder]` | ersetzt ein Element durch einen Grundkörper gleicher Größe, mit dem Material des Originals |
| `meshcut tools` | alle Befehle als Tool-Definition (JSON-Schema) für KI-Assistenten, aus dem Parser erzeugt |
| `meshcut serve --root ORDNER --mcp` oder `--http PORT` | MCP-Server (stdio) bzw. HTTP-API für Agenten |
| `meshcut info DATEI` | Eckpunkte, Dreiecke, Volumen, Fläche, wasserdicht, Euler-Zahl, Hash |
| `meshcut labels DATEI KARTE.npy --grid 2x2x2` | schreibt eine Zuordnungskarte (Teilnummer pro Dreieck) für ein Gitter |

### Arten zu zerlegen (`split`)

```bash
meshcut split m.stl out --plane x=12.5                  # eine Ebene (x, y oder z = Wert)
meshcut split m.stl out --plane x=0 --plane 1,1,0,4     # mehrere, auch schräg: a,b,c,d  =  a*x+b*y+c*z = d (negativ: --plane=-1,0,0,5)
meshcut split m.stl out --grid 2x2x3                    # Gitter über den Begrenzungsquader, Teilnummer = Zelle
meshcut split m.stl out --max 220x220x250               # so viele gleichmäßige Schnitte, dass jedes Teil in den Druckraum passt
meshcut split m.stl out --wedges 8 --axis z             # 8 Keile um die z-Achse (gerade Zahl; --center 10,5 verschiebt die Achse)
meshcut split m.stl out --labels karte.npy              # Zuordnungsmodus: Geometrie bleibt unverändert
```

Weitere Optionen: `--format stl|obj|ply|3mf|glb` (3MF und GLB legen alle Teile in **eine** Datei), `--no-cap` (ohne Deckel, für offene Netze), `--flip` (Dreiecke umdrehen, wenn das Netz nach innen orientiert ist), `--snap` (siehe unten), `--verify`.

**Zuordnungskarte:** eine Datei (`.npy`, `.json` oder `.csv`) mit **einer ganzen Zahl pro Dreieck** (Teilnummer, Lücken erlaubt, höchstens 1024 Teile). So lassen sich Ergebnisse einer Netz-Segmentierung (z. B. PartField, SAMesh, P3-SAM) verwenden: einfach ihre Dreiecks-Labels als `.npy` speichern. meshcut zeichnet selbst nur Gitter-Karten (`meshcut labels`).

## KI: Teile finden und benennen (`segment`)

```bash
meshcut segment figur.stl teile --parts 3                          # eingebaut: räumliche Gruppen (k-Means), ohne Modell und ohne Netz
meshcut segment figur.stl teile --names Kopf,Rumpf,Beine --format 3mf   # Anzahl und Namen; Namen werden Objektnamen in 3MF/GLB
meshcut segment szene.glb teile --backend components               # eigenständige Hüllen (Gehäuse, lose Teile)
meshcut segment figur.stl teile --prompt "trenne Kopf, Arme, Beine" # Sprachmodell holt die Namen aus dem Satz
```

Das Ergebnis ist **immer eine Zuordnungskarte** (ein Wert pro Dreieck). Teil 0 ist das höchste Teil (z), dann abwärts, damit Namen stabil bleiben. Die Namen stehen in `meshcut.json` (`names`) und werden beim Einlesen streng geprüft (Buchstaben, Ziffern, Leerzeichen, `- . _`, höchstens 64 Zeichen, eindeutig).

| Verfahren (`--backend`) | Quelle | Einrichtung |
|---|---|---|
| `kmeans` (Vorgabe), `components` | eingebaut, deterministisch | keine |
| `command` | lokales Modell als Befehl (PartField, SAMesh, eigenes Skript) | `MESHCUT_SEGMENTER_CMD='python mein_modell.py {input} {output} {parts}'` |
| `http` | gehosteter Dienst, der ein GLB annimmt und `{"labels": [...]}` zurückgibt | `MESHCUT_SEGMENT_URL`, `MESHCUT_SEGMENT_KEY` |

Sprachmodell zum Benennen (`--prompt`): `ANTHROPIC_API_KEY` (Modell über `MESHCUT_LLM_MODEL`, Vorgabe `claude-sonnet-5-5`) **oder** ein OpenAI-kompatibler Endpunkt mit `MESHCUT_LLM_URL`, `MESHCUT_LLM_MODEL`, `MESHCUT_LLM_KEY`. Ist `MESHCUT_LLM_URL` gesetzt, gewinnt der Endpunkt; der Anthropic-Schlüssel geht nur an `api.anthropic.com`. Das Modell bekommt nur Namen und Kennzahlen der Teile (Lage, Größe), kein Netz; die Antwort wird geprüft (jeder Name genau einmal, nur bekannte Teile).

**Sicherheit.** Schlüssel kommen nur aus der Umgebung, nie aus Dateien oder Argumenten, und stehen weder in Ausgaben noch in `meshcut.json`. Entfernte Adressen müssen `https` sein (`http` nur für localhost), Weiterleitungen werden nicht befolgt, und was ein eigener Befehl auf stderr schreibt, reicht meshcut nicht weiter. Netzzugriff gibt es nur, wenn Sie ihn einrichten (`http`-Segmentierer, `--prompt`); `--prompt` mit `ANTHROPIC_API_KEY` ruft `api.anthropic.com` auf (oder den https-Endpunkt aus `MESHCUT_ANTHROPIC_URL`, falls Sie ihn setzen; der Schlüssel geht dann dorthin).

**Lizenzen der bekannten Modelle (Stand 10.11.2026, vor dem Einsatz selbst prüfen):** PartField (NVIDIA) nur nicht-kommerziell; P3-SAM/Hunyuan3D-Part (Tencent) gilt laut Lizenz nicht in EU, UK und Südkorea; SAMesh hat keine Lizenzdatei; PartCrafter (MIT) erzeugt Teile aus einem Bild, zerlegt aber kein vorhandenes Netz. meshcut bringt keines davon mit, der Adapter ruft nur an, was Sie selbst eingerichtet haben.

## Mit KI-Assistenten bedienen

meshcut ist dafür gebaut, dass ein KI-Assistent (Claude, ChatGPT, Codex und andere) es steuert: ein Befehl pro Aufgabe, klare Fehlerzeilen, Ergebnisse als Dateien. Sag zum Beispiel:

> „Nimm die Antenne vom Roboter ab und schließ die Stelle sauber.“
> „Ersetze den Knauf durch eine Kugel in derselben Größe, mit demselben Material.“
> „Teil das Modell so, dass es auf mein Druckbett 220 × 220 passt, und prüf, dass nichts fehlt.“

Der Assistent wählt den Befehl (`regions` + `remove`, `regions` + `swap`, `split --max` + `verify`), meshcut führt ihn exakt aus. meshcut selbst versteht keine Sätze, es ist die verlässliche Ausführung dahinter. Das Modellieren von Hand entfällt: der Assistent sagt, was er will, meshcut erledigt Auswahl, Lochschluss, Material, UV und Normalen selbst.

Das Werkzeug für Assistenten (Harness):

- `meshcut regions modell.obj ordner` nummeriert die erkennbaren Elemente: zusammenhängende Teile, Objekte, Gruppen und Materialien (`regions.png` mit Nummern im Bild, `regions.json` mit Punkt, Rahmen, Fläche, Volumen und Material). Der Assistent schaut das Bild an und nennt die Nummer, ohne Koordinaten zu raten; alternativ wählt `--at x,y,z` das kleinste Element an einem Punkt (bei gleich großen gewinnt die kleinere Nummer) und `--name` ein Material, Objekt oder eine Gruppe (auch wenn das Element in der Liste als „Teil N“ steht; passt ein Name auf mehrere Elemente, bricht meshcut ab). Das Bild zeigt eine Ansicht: verdeckte Elemente haben keine Nummer im Bild (`visible: false` in `regions.json`), sie sind trotzdem über Nummer, `--at` oder `--name` wählbar. Mehr als 500 Elemente werden nicht nummeriert; dann wählt `--at` unter den nummerierten das kleinste.
- `meshcut tools` gibt alle Befehle als Tool-Definition (JSON-Schema) aus, die jeder Agent so registrieren kann. Sie wird aus dem Parser erzeugt und bleibt damit immer aktuell. Der MCP-Server und die HTTP-API bieten `regions`, `remove` und `swap` ebenfalls an.
- `remove` und `swap` messen am Modell selbst: Das Loch wird mit einem Ohrenschnitt in der Randebene geschlossen (ohne neue Punkte, jedes Dreieck mit eigener Normale). Folgt die Umgebung einer Wölbung (Kugel, Zylinder, Schale), legt meshcut einen Paraboloid an Rand und ersten Nachbarring und füllt mit Zwischenringen auf dieser Fläche; das greift nur bei einem sternförmigen Rand und einer Wölbung ab 1 % des Lochdurchmessers, eine ebene Fläche bleibt exakt eben. Material und UV-Werte kommen von der Umgebung (die UV der Zwischenringe aus der Lage, am Rand gefittet), die Glättung des Ersatzes kommt aus der Form. Ein Rand, der weder ebene Triangulation noch einen Fächer erlaubt, wird abgewiesen. Ragt die Wölbung über ein Drittel des Lochdurchmessers (große Löcher in Kugeln, stark verjüngte Rohre), bleibt die Füllung flach, statt eine Spitze ins Leere zu setzen. Ein Loch mit nicht ebenem, aber triangulierbarem Rand (etwa ein Fenster von 180° in einem Rohr) wird flach gefüllt und meshcut sagt das ausdrücklich dazu (`Hinweis`); auch bei `swap` ohne `--shape` meldet es, wenn keine Form gut passt (Abweichung über 0,15 der Diagonale). Liegt das Loch auf einer UV-Naht, bekommen die neuen Ecken fortlaufende UV-Werte (z. B. 1,02 statt 0,02), die Textur muss dafür wiederholen. Der Ersatz hat die Größe des Originals mal `--scale` (0,01 bis 100); ohne Material im Original bleibt er ohne Material (`usemtl off`). Der Assistent muss keine Parameter mitliefern (`--shape`, `--axis`, `--scale`, `--material` sind nur Feinschliff).
- Bei `swap` ohne `--shape` rechnet meshcut alle Kandidaten (Quader, Kugel, Zylinder je Achse), misst bei jedem den Abstand zum Original in beide Richtungen und nimmt den besten; die Zeile „Kandidaten verglichen“ zeigt die Werte. Beim Lochschluss wird ein Ohrenschnitt mit Splittern gegen einen Fächer vom Mittelpunkt abgewogen.
- Der Messprüfstand `bench/edit_bench.py` baut Netze mit bekannter Wahrheit (Platte und Kugel mit Höcker, rechteckig und C-förmig, freie Teile zum Ersetzen), entfernt/ersetzt und misst Außenbereich, offene Kanten, Abweichung, Fläche, Normalen, UV, Windung, Dreiecksgüte und Doppellauf, in fünf Maßstäben von 1e-9 bis 1e9. Die Schwellen stehen im Kopf der Datei und wurden vor dem Messen festgelegt; eine nachträgliche Lockerung ist dort ausgewiesen.
- Am Ende steht eine Prüfzeile: „von 24 Dreiecken 2 entfernt, 2 neu dazugekommen; die übrigen 22 sind unverändert (Eckpunkt, UV, Normale, Material verglichen)“. Meshcut liest dafür die geschriebene Datei zurück, vergleicht jede unberührte Fläche und prüft bei den neuen Flächen Material, Zahl und endliche Koordinaten; die Zahlen in der Zeile stammen aus dieser Datei. Stimmt etwas nicht, bricht es ab und entfernt die Ausgabe. Die Prüfung belegt nicht, dass das Ergebnis schön aussieht; die Deckel sind einfache Dreiecke, kein Remeshing. Bearbeitet werden höchstens 500 000 Flächen (größere Netze vorher mit `split` teilen).

OBJ wird vollständig mitgeführt (Material, UV, Normalen, Objekt- und Gruppennamen, Glättungsgruppen, `mtllib` wird mitkopiert). STL, PLY, 3MF und GLB werden hier nur mit Geometrie geschrieben (Farben, Materialien, UV gehen verloren); dort prüft meshcut Dreieckszahl und Koordinaten, und die Prüfzeile sagt es. Die Eingabedatei wird nie überschrieben.

## Für Agenten: MCP und HTTP

```bash
meshcut serve --root ./arbeit --mcp          # MCP über stdin/stdout (Claude, Codex u. a.)
MESHCUT_TOKEN=geheim meshcut serve --root ./arbeit --http 8765
```

Werkzeuge: `info`, `split`, `segment`, `join`, `verify`, `regions`, `remove`, `swap` (Schema über `tools/list` bzw. `GET /v1/tools`). Dateien werden nur unterhalb von `--root` gelesen und geschrieben, auch nicht über Links oder `..`. Die HTTP-API bindet an `127.0.0.1`; ohne `MESHCUT_TOKEN` nimmt sie nur Anfragen mit `Host: localhost` und `Content-Type: application/json` an (Schutz vor Angriffen aus dem Browser). Eine andere Adresse (`--host`) geht nur mit Schlüssel (Bearer).

```bash
curl -s -H "Authorization: Bearer geheim" -H "Content-Type: application/json" \
  -d '{"input":"figur.stl","outdir":"teile","parts":3}' http://127.0.0.1:8765/v1/segment
```

## Was genau garantiert wird

Zwei Stufen, die man nicht vermischen darf:

**Zuordnungsmodus.** Die Eckpunkte und Dreiecke bleiben unverändert, es werden nur Dreiecke auf Teile verteilt. Setzt man die Teile wieder zusammen, kommt dieselbe Dreiecksmenge heraus: gleicher SHA-256, **bitgenau**.

**Schnittmodus.** Dreiecke werden an Ebenen zerlegt. Der Rundlauf ist trotzdem exakt, weil meshcut in `meshcut.json` für jedes Teildreieck die Nummer seines Quelldreiecks und für jedes zerschnittene Dreieck dessen Original-Koordinaten (64 Bit) speichert. `join` entfernt die Deckel und setzt die Original-Dreiecke wieder ein; der Hash stimmt mit dem der Quelle überein. Ohne Begleitdatei bleibt die Geometrie gleich (Hausdorff-Abstand ~ Rundungsfehler), nur die Dreiecke bleiben so fein, wie der Schnitt sie gemacht hat.

**Dateiformat und Genauigkeit.** STL und GLB speichern 32 Bit: der Hash wird dann nach Rundung auf float32 gebildet (steht in `meshcut.json` als `precision`). Wer **wirklich 64 Bit** will, nimmt `--format obj` oder `3mf` (Text mit 17 Stellen, ohne Verlust).

**Rundlauf-Kriterien des Schnittmodus** (alle in `verify`):
- **Teile unverändert:** jede Teildatei hat denselben Hash wie beim Schreiben (Fragmente und Deckel eingeschlossen). Verschobene, beschädigte oder ausgetauschte Teile fallen sofort auf, auch ohne `--source`.
- **Deckel auf den Schnittebenen:** jedes Deckel-Dreieck liegt mit allen drei Ecken auf einer der in `meshcut.json` gespeicherten Ebenen. Ein Schnitt an der falschen Stelle fällt auf.
- Volumen aller Teile = Volumen des Originals, Fläche ohne Deckel = Originalfläche, jedes Teil wasserdicht (jede Kante genau zweimal, gegenläufig), Euler-Zahl gerade, Hash gleich.
- Mit `--source`: Hash der Quelldatei und Hausdorff-Abstand der ohne Begleitdatei zusammengesetzten Teile. Bei großen Netzen prüft der Hausdorff-Abstand eine **Stichprobe** der Eckpunkte (steht dann im Text); die Sicherheit kommt aus den Hashes, der Hausdorff-Abstand ist eine zusätzliche, unabhängige Gegenprobe.

Erlaubt sind 1e-9 relativ bei 64 Bit und 1e-4 bei 32 Bit; bei STL/GLB weit vom Ursprung kommt die float32-Rundung der Koordinaten dazu.

**Einheiten.** 3MF-Dateien werden beim Einlesen nach Millimeter umgerechnet (`inch`, `centimeter`, `meter`, `micron`, `foot`); meshcut schreibt 3MF immer in Millimeter. Die Größe in der Welt bleibt gleich, nur die Zahlen ändern sich.

**Abbruch beim Schreiben.** `split` ersetzt einen Ausgabeordner als Ganzes: Fehler und Ausnahmen rollen zurück, und auch nach einem harten Abbruch (Stromausfall, `kill -9`) legt ein Journal im versteckten `.meshcut_…`-Ordner fest, was passiert: Der nächste Aufruf von `split` auf diesen Ordner stellt den alten Satz wieder her (oder schließt ein fertiges Ersetzen ab) und meldet das auf stderr. `join` und `verify` räumen nur eigene Reste (Marker `.meshcut-staging` oder Journal, keine Links) auf, die älter als zehn Minuten sind, damit sie einen gerade laufenden `split` in einem anderen Prozess nicht stören. Zwei `split` gleichzeitig in denselben Ordner sind nicht vorgesehen (innerhalb des Dienstes sorgt eine Sperre dafür, dass immer nur ein Werkzeug läuft).

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

mc.regions("figur.obj", "nummern")                          # nummern/regions.png + regions.json
r = mc.swap("figur.obj", region=3, shape="sphere")          # {'output': 'figur_edit.obj', 'text': '... Pruefung: ...'}
mc.remove("figur.obj", name="Antenne", out="ohne.obj")
```

Die Schnittstelle führt dieselben Befehle aus wie die Kommandozeile (gleiche Optionen mit Unterstrich: `no_cap=True`), schweigt auf stdout und stderr und wirft `mc.MeshcutError` statt das Programm zu beenden. Tiefer unten liegen `Mesh`, `split_planes`, `split_labels`, `triangle_hash`, `mesh_stats` für eigene Abläufe mit numpy-Arrays.

## Im Browser

**[calyiex.github.io/meshcut](https://calyiex.github.io/meshcut/)** – derselbe Kern läuft in Pyodide/WebAssembly, nichts wird hochgeladen. Datei wählen, Schnitt einstellen, Teile als ZIP speichern; zum Zusammensetzen alle Teile samt `meshcut.json` wieder auswählen. Bis 40 MB je Datei. Bei jedem Commit prüft das CI die Seite in WebKit (Safari/iPhone-Engine), Chromium und Firefox: Zerlegen und Zusammensetzen im Browser müssen den Hash der Quelle ergeben.

## Grenzen (ehrlich)

- **`split`, `join`, `segment` behandeln nur Geometrie.** UV-Koordinaten, Farben, Normalen und Materialien werden dort nicht übernommen; Texturen und Vertexfarben bleiben also nicht erhalten. (`regions`, `remove` und `swap` führen sie in OBJ mit, siehe unten.) Eckpunkte mit gleichen Koordinaten werden beim Einlesen verschmolzen (GLB-Nähte, STL-Dreiecksuppe).
- **Deckel brauchen ein wasserdichtes, nach außen orientiertes Netz.** Sonst bricht `split` mit einer klaren Meldung ab (`--no-cap` schneidet trotzdem, `--flip` dreht die Orientierung).
- **Keile nur in gerader Zahl.** Ungerade Keilzahlen bräuchten Halbebenen.
- **Stifte und Passungen** (Dübel, Zapfen/Loch mit Spiel) gibt es in v0.1 noch nicht, ebenso keine Bearbeitung mehrerer Objekte einer Datei auf einmal (3MF/GLB mit mehreren Objekten werden als ein Netz gelesen). manifold3d wird nur in den Tests als unabhängige Gegenprobe benutzt (Volumen der Teile gegen `split_by_plane`), nicht zur Laufzeit.
- **Triangulierung der Deckel** ist gültig und deckungsgleich, aber nicht „schön“: Ohrenschnitt erzeugt gern lange schmale Dreiecke. Für den Druck unkritisch, für Simulation ungeeignet. Läuft der Ohrenschnitt in einen Notfall (Punkte exakt aufeinander), meldet `split` das.
- **Geschwindigkeit.** Der Kern ist numpy; die Deckel-Triangulierung ist Python. Rund 100 000 Dreiecke in unter einer Sekunde, Deckel mit mehreren tausend Randpunkten brauchen Sekunden.
- **Ebene durch einen konkaven Eckpunkt oder eine Kante des Netzes:** berühren sich die Teile danach an einer Kante, ist ein Teil nicht wasserdicht. `split` meldet das („Teil … ist nicht wasserdicht“), `verify` ebenfalls.
- **Die Reihenfolge der Dreiecke in einer Teildatei gehört zum Beweis.** Die Deckel stehen am Ende der Datei; schreibt ein anderes Programm (Slicer, Mesh-Editor) eine Teildatei mit denselben Dreiecken in anderer Reihenfolge neu, schlagen `ebenen`, `hash`, `reihenfolge` und `flaeche` an, obwohl die Geometrie gleich ist. Dann `join` ohne Ordner (aus den Dateien, Näherung) oder die Teile unverändert lassen.
- **32-Bit-Dateien und Topologie.** Bei STL/GLB können nahe beieinanderliegende Schnittpunkte beim Runden auf float32 zusammenfallen; dann meldet `verify` eventuell `wasserdicht`, obwohl der 64-Bit-Satz in Ordnung wäre. Wer das nicht will, nimmt `--format obj`, `ply` oder `3mf`.
- **Die Begleitdatei ist keine Signatur.** Die Hashes in `meshcut.json` belegen, dass nichts zufällig oder unbemerkt verloren ging oder vertauscht wurde. Wer die Teile *und* `meshcut.json` absichtlich zusammen umschreibt und alle Zahlen neu rechnet, erzeugt einen in sich stimmigen Satz, den `verify` ohne Quelle nicht von einem echten unterscheidet. Für Echtheit gegenüber Dritten: `verify --source ORIGINAL` mit der Originaldatei oder den Hash der Quelle (`source_hash`) getrennt aufbewahren.
- **Dienst: ein Werkzeug zur Zeit.** Ein Aufruf über MCP/HTTP hält die Sperre, bis er fertig ist, und der HTTP-Dienst hat keine Obergrenze für gleichzeitige Verbindungen (er ist für Vertrauenswürdige gedacht, nicht fürs offene Netz); ein eigener Segmentierungsbefehl darf bis zu einer Stunde laufen. Wer den Dienst teilt, schaltet ihn mit `MESHCUT_TOKEN` ab und lässt nur Vertrauenswürdige heran: jeder Aufrufer kann den eingerichteten Befehl oder das Sprachmodell (Kosten) auslösen.
- **Eingelesene Dateien sind begrenzt** (Schutz vor Dateien, die klein sind und sich beim Auflösen vervielfachen): höchstens 10 Millionen Dreiecke nach dem Auflösen aller Instanzen (GLB-Knoten, 3MF-Komponenten), 100 000 Knoten-/Komponentenbesuche, 128 MiB entpackte 3MF-Modelldatei, 5000 Ecken je Fläche und ein Arbeitsbudget für alle Flächen mit mehr als drei Ecken (Vierecke zählen ebenfalls; konvexe Vierecke gehen einen schnellen Weg, alles andere kostet mehr). Das ist ein Deckel gegen Hänger, kein Speicherschutz: 10 Millionen Dreiecke brauchen mehrere GB Arbeitsspeicher, 3MF-Dateien etwa das 20-fache der XML-Größe. Wer meshcut als Dienst für Fremde betreibt, begrenzt zusätzlich Speicher und Laufzeit des Prozesses selbst.
- **Sehr große Koordinaten.** Weit vom Ursprung (ab etwa dem 1e8-fachen der Netzgröße) geht in 64 Bit die Auflösung aus; in STL/GLB (32 Bit) schon ab etwa dem 1e5-fachen. Vorher das Netz in die Nähe des Ursprungs verschieben.
- **Bearbeiten (`remove`, `swap`) ist kein exakter Schnitt.** Es gibt keinen Rundlauf-Beweis wie bei `split`; stattdessen liest meshcut die geschriebene Datei zurück und belegt, dass jede unberührte Fläche (Eckpunkt-, UV-, Normalen- und Materialtext, Reihenfolge) unverändert ist, und prüft, dass keine neue Fläche gegenläufig auf einer vorhandenen liegt (gerichtete Doppelkanten). Das Schließen eines Lochs und der Ersatz sind eine gute, keine perfekte Wahl: ebene und konkave Ränder werden mit einem Ohrenschnitt aus Randpunkten gefüllt (bei konkaven Löchern auf gewölbter Fläche bleibt der Deckel flach: im Prüfstand verliert er dort etwa 8 % Fläche), sternförmige Löcher auf gewölbter Fläche folgen der Wölbung; Inseln im Loch (ein behaltenes Dreieck mitten im Rand) werden nicht erkannt; als Ersatz gibt es Quader, Zylinder und Kugel in der Größe des Originals (keine Nachbildung der Form, keine Textur-Neuberechnung, die UV nehmen den Bereich des Originals). Fehlen dem Modell UV oder Normalen, schreibt meshcut auch keine.
- **Elemente heißen nach Reihenfolge.** Die Nummern aus `regions` gelten für genau diese Datei in ihrem jetzigen Zustand; nach einer Bearbeitung neu nummerieren. Als Elemente zählen zusammenhängende Teile (über gleiche Eckpunktlage, UV-Nähte trennen nichts), Objekte, Gruppen und Materialien; Elemente, die dieselben Flächen haben wie ein früheres, stehen nicht doppelt da. Linien und Punkte (`l`, `p`) eines OBJ werden nicht übernommen (meshcut sagt es).
- **Die eingebauten Verfahren sind einfach.** `kmeans` teilt nach Lage, nicht nach Bedeutung; für echte Teilerkennung (Arm, Griff, Rad) braucht es ein Modell über `command` oder `http`. Das ist beabsichtigt: meshcut garantiert, was nach der Karte kommt, nicht die Qualität der Karte.
- **Höchstens 1024 Teile** (`--grid`, `--max`): wird vor dem Schneiden geprüft.
- Nichtmanifold-Eckpunkte, an denen sich zwei Teile nur in einem Punkt berühren, sind erlaubt; `verify` zählt sie für die Euler-Zahl getrennt.

## Entwicklung

```bash
pip install -e ".[check]"     # pytest, hypothesis, trimesh, manifold3d (Gegenproben)
python -m pytest -q tests
MESHCUT_EXAMPLES=300 python -m pytest -q tests/test_property.py     # gründlicher
```

Die Tests prüfen unter anderem (Befunde aus den Prüfrunden stehen in `tests/test_review.py`, auch der harte Abbruch mit `os._exit`): Zufallsebenen, Ebenen durch Eckpunkte/Kanten/Flächen (hypothesis), Dreiecke in der Ebene, Keile um eine Achse, Torus/Hohlkugel (Löcher und Inseln im Deckel), alle fünf Formate, Absturz-Injektion beim Schreiben, Fremddateien im Ausgabeordner und den Vergleich mit trimesh und manifold3d. Dazu kommt ein Mutationstest (294 absichtlich eingebaute Fehler in Kern, KI-Schicht und Dienst, jeder wird von den Tests bemerkt; `python tests/mutate.py`) sowie Tests für KI-Adapter (lokale Fake-Dienste), MCP und HTTP-API. Die Browser-Version liegt als Kopie in `docs/kkmeshcut.py`; ein Test stellt sicher, dass beide Dateien gleich sind.

Lizenz: MIT, © K&K AI.
