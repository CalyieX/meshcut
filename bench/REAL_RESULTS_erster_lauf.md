# Fremdmodelle: Messergebnis (bench/real_bench.py)

```
Fremdmodelle: 33 Faelle (11 Modelle x 3 Fleckgroessen)
  'ok' gemeldet und wirklich gut: 4 von 33 = 12 % (95 %-Intervall 5-27 %), Ziel >= 85 %
  davon 'ok' gemeldet insgesamt: 7; unsicher/nicht ok/abgelehnt: 26
  verschwiegene Fehler ('ok', aber nicht gut): 3
  bestanden (gut oder gemeldet): 30 von 33 = 91 % (Intervall 76-97 %)
```

| Modell | Fleck | Dreiecke | Urteil | Abweichung | Normale | gut | Grund |
|---|---|---|---|---|---|---|---|
| alligator | 1 % | 5981 | nicht ok | 0.0 | 0.0 | nein | das Loch ist nicht ganz zu: 15 offene Kanten mehr als vorher |
| alligator | 3 % | 5981 | nicht ok | 0.0 | 0.0 | nein | das Loch ist nicht ganz zu: 23 offene Kanten mehr als vorher |
| alligator | 6 % | 5981 | nicht ok | 0.0 | 0.0 | nein | das Loch ist nicht ganz zu: 46 offene Kanten mehr als vorher |
| beetle | 1 % | 2053 | nicht ok | 0.0 | 0.0 | nein | das Loch ist nicht ganz zu: 12 offene Kanten mehr als vorher; das Netz hat Kanten, an denen mehr als zwei Flae |
| beetle | 3 % | 2053 | unsicher | 0.0 | 0.0 | ja | das Netz hat Kanten, an denen mehr als zwei Flaechen zusammentreffen |
| beetle | 6 % | 2053 | nicht ok | 0.0 | 0.0 | nein | das Loch ist nicht ganz zu: 18 offene Kanten mehr als vorher; das Netz hat Kanten, an denen mehr als zwei Flae |
| cheburashka | 1 % | 13334 | ok | 0.0065 | 23.7 | ja |  |
| cheburashka | 3 % | 13334 | abgelehnt | - | - | - | Das Loch ist zu verwickelt, um es sauber zu schliessen (Rand mit 84 Punkten, weder eben noch sternfo |
| cheburashka | 6 % | 13334 | abgelehnt | - | - | - | Das Loch ist zu verwickelt, um es sauber zu schliessen (Rand mit 146 Punkten, weder eben noch sternf |
| cow | 1 % | 5804 | abgelehnt | - | - | - | Das Loch ist zu verwickelt, um es sauber zu schliessen (Rand mit 36 Punkten, weder eben noch sternfo |
| cow | 3 % | 5804 | abgelehnt | - | - | - | Selbstpruefung fehlgeschlagen: die Fuellung liegt teilweise falsch herum oder ueberlappt sich (E:\Te |
| cow | 6 % | 5804 | abgelehnt | - | - | - | Das Loch ist zu verwickelt, um es sauber zu schliessen (Rand mit 19 Punkten, weder eben noch sternfo |
| fandisk | 1 % | 12946 | ok | 0.0134 | 9.7 | ja |  |
| fandisk | 3 % | 12946 | ok | 0.024 | 19.2 | nein |  |
| fandisk | 6 % | 12946 | ok | 0.0261 | 7.9 | nein |  |
| homer | 1 % | 12000 | abgelehnt | - | - | - | Selbstpruefung fehlgeschlagen: die Fuellung liegt teilweise falsch herum oder ueberlappt sich (E:\Te |
| homer | 3 % | 12000 | unsicher | 0.0204 | 24.2 | nein | die Oberflaeche knickt an der Fuellung sichtbar ab (im Mittel 25 Grad) |
| homer | 6 % | 12000 | abgelehnt | - | - | - | Selbstpruefung fehlgeschlagen: die Fuellung liegt teilweise falsch herum oder ueberlappt sich (E:\Te |
| rocker-arm | 1 % | 20088 | abgelehnt | - | - | - | Selbstpruefung fehlgeschlagen: die Fuellung liegt teilweise falsch herum oder ueberlappt sich (E:\Te |
| rocker-arm | 3 % | 20088 | unsicher | 0.0207 | 32.6 | nein | die Oberflaeche knickt an der Fuellung sichtbar ab (im Mittel 31 Grad) |
| rocker-arm | 6 % | 20088 | abgelehnt | - | - | - | Das Loch ist zu verwickelt, um es sauber zu schliessen (Rand mit 163 Punkten, weder eben noch sternf |
| spot | 1 % | 5856 | ok | 0.0074 | 13.1 | ja |  |
| spot | 3 % | 5856 | ok | 0.0209 | 25.9 | nein |  |
| spot | 6 % | 5856 | abgelehnt | - | - | - | Das Loch ist zu verwickelt, um es sauber zu schliessen (Rand mit 77 Punkten, weder eben noch sternfo |
| suzanne | 1 % | 968 | unsicher | 0.0052 | 7.9 | ja | die Oberflaeche knickt an der Fuellung sichtbar ab (im Mittel 27 Grad); die Fuellung ist flach, die Flaeche dr |
| suzanne | 3 % | 968 | unsicher | 0.0137 | 18.8 | ja | die Oberflaeche knickt an der Fuellung sichtbar ab (im Mittel 27 Grad); die Fuellung ist flach, die Flaeche dr |
| suzanne | 6 % | 968 | nicht ok | 0.1212 | 64.5 | nein | die Oberflaeche knickt an der Fuellung stark ab (im Mittel 74 Grad); die Fuellung ist flach, die Flaeche drumh |
| teapot | 1 % | 6320 | unsicher | 0.0111 | 17.5 | ja | die Fuellung ist flach, die Flaeche drumherum ist gewoelbt |
| teapot | 3 % | 6320 | abgelehnt | - | - | - | Das Loch ist zu verwickelt, um es sauber zu schliessen (Rand mit 45 Punkten, weder eben noch sternfo |
| teapot | 6 % | 6320 | unsicher | 0.012 | 21.4 | ja | die Oberflaeche knickt an der Fuellung sichtbar ab (im Mittel 31 Grad); die Fuellung ist flach, die Flaeche dr |
| woody | 1 % | 1267 | ok | 0.0 | 0.0 | ja |  |
| woody | 3 % | 1267 | nicht ok | 0.0 | 0.0 | nein | das Loch ist nicht ganz zu: 8 offene Kanten mehr als vorher |
| woody | 6 % | 1267 | nicht ok | 0.0 | 180.0 | nein | das Loch ist nicht ganz zu: 11 offene Kanten mehr als vorher; die Oberflaeche knickt an der Fuellung stark ab  |
