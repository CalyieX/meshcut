# Schwellen des Urteils (eingefroren 17.11.2026, vor der ersten Messung)

Urteil = letzte Ausgabezeile von `remove`/`swap`: `verdict: ok | unsicher - Grund | nicht ok - Grund`. Exit-Code 2 bei "nicht ok" (Datei wird trotzdem geschrieben).

nicht ok:  neue offene Kanten (nachher > vorher) | Knickwinkel am Fuellrand im Mittel > 60 Grad (JUMP_BAD)
unsicher:  Knick > 25 Grad (JUMP_UNSURE) | Volumen > 20 % anders (VOLUME_BIG) | entfernte Flaeche > 30 % der Gesamtflaeche (BIG_HOLE) |
           >= 2 getrennte Loecher (MANY_HOLES) | flache Fuellung auf Woelbung (uneven/bent, ohne ruhiges Volumen) | keine Form passt (BAD_SHAPE) |
           Material-Tippfehler | Farben/Material/UV gehen im Zielformat verloren | Kanten mit mehr als zwei Flaechen (nicht mannigfaltig)
ok:        sonst.

Messlatte: bench/edit_bench.py (35 Faelle) darf hoechstens 2 Fehlalarme (unsicher/nicht ok) liefern; die absichtlich schlechten Faelle muessen "unsicher" oder "nicht ok" sein.
Aenderungen an den Schwellen nach der Messung stehen unten mit Datum und Grund.

## Fremdmodelle (eingefroren 17.11.2026, vor dem ersten Lauf von bench/real_bench.py)
11 echte Modelle (alligator, beetle, cheburashka, cow, fandisk, homer, rocker-arm, spot, suzanne, teapot, woody; Quelle alecjacobson/common-3d-test-models), je ein Fleck von 1 %, 3 %, 6 % der Dreiecke = 33 Faelle.
gut = Abweichung <= 2 % der Diagonale UND mittlerer Normalenwinkel <= 30 Grad UND nicht mehr offene Kanten als das Original.
bestanden = gut ODER Urteil unsicher/nicht ok; ein verschwiegener Fehler ('ok', aber nicht gut) faellt durch. Ziel: >= 85 % 'ok' und wirklich gut, 0 verschwiegene Fehler.
Wilson-Intervall 95 % wird mit ausgegeben. Zu erwarten (vor dem Lauf, nicht nachtraeglich angepasst): grosse Flecken auf stark gekruemmten Modellen werden nicht gut -> sollen 'unsicher' sein.

## Nachtrag 17.11.2026, NACH dem ersten Fremdmodell-Lauf (ehrlich: die Regel ist aus Messwerten abgeleitet)
Befund vor jeder Aenderung (alter Fueller, Fleck am offenen Rand ausgeschlossen): 9 von 33 = 27 % 'ok und gut', 3 verschwiegene Fehler, 91 % bestanden (REAL_RESULTS_vor_glaettung.md).
Ein neuer Fueller fuer nicht ebene Rand-Loecher ab 12 Randpunkten (Dreiecke kleinster Flaeche, Liepa-Verfeinerung, Glaettung mit Anschluss an den ersten Nachbarring) schliesst alle Loecher,
liess aber 7 verschwiegene Fehler stehen (REAL_RESULTS_glaettung_v1.md). Deshalb eine zusaetzliche Regel (LARGE_HOLE/BENT_NOTE): ein Loch ab 2 % der Oberflaeche, dessen Umgebung
(Rand + erster Ring) um mehr als 2 % des Durchmessers von der Randebene abweicht und das nicht mit einer belegten Kugel-/Paraboloidflaeche gefuellt wurde, ist 'unsicher'.
Kalibriert an alligator, beetle, cheburashka, cow, fandisk, homer (Schwellen 0,02/0,02 liegen zwischen den guten und den verschwiegenen Faellen dort);
geprueft an den uebrigen fuenf (rocker-arm, spot, suzanne, teapot, woody), die dabei nicht angepasst wurden. Ergebnis steht in REAL_RESULTS.md.
