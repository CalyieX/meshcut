# Schwellen des Urteils (eingefroren 17.11.2026, vor der ersten Messung)

Urteil = letzte Ausgabezeile von `remove`/`swap`: `verdict: ok | unsicher - Grund | nicht ok - Grund`. Exit-Code 2 bei "nicht ok" (Datei wird trotzdem geschrieben).

nicht ok:  neue offene Kanten (nachher > vorher) | Knickwinkel am Fuellrand im Mittel > 60 Grad (JUMP_BAD)
unsicher:  Knick > 25 Grad (JUMP_UNSURE) | Volumen > 20 % anders (VOLUME_BIG) | entfernte Flaeche > 30 % der Gesamtflaeche (BIG_HOLE) |
           >= 2 getrennte Loecher (MANY_HOLES) | flache Fuellung auf Woelbung (uneven/bent, ohne ruhiges Volumen) | keine Form passt (BAD_SHAPE) |
           Material-Tippfehler | Farben/Material/UV gehen im Zielformat verloren | Kanten mit mehr als zwei Flaechen (nicht mannigfaltig)
ok:        sonst.

Messlatte: bench/edit_bench.py (35 Faelle) darf hoechstens 2 Fehlalarme (unsicher/nicht ok) liefern; die absichtlich schlechten Faelle muessen "unsicher" oder "nicht ok" sein.
Aenderungen an den Schwellen nach der Messung stehen unten mit Datum und Grund.
