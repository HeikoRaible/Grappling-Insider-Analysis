# GrapplingInsider

Dies ist unsere Abgabe des Moduls Text- und Webmining. Unsere Webseite: https://grapplinginsider.com/

## Booklet

Das booklet liegt hier als `booklet.pdf`, ist aber auch nochmal als `.tex` unter `booklet/` aufzufinden.
Wenn man die Inhaltsangabe, Überschriften, Bilder und eingefärbten Ausgaben zählt, beinhaltet es deutlich mehr als die gewünschte Seitenanzahl. Diese sind jedoch leicht zu überspringen und abseits derer scheint es uns im Rahmen.

## Requirements

Installiere zunächst die notwendigen Python packages mithilfe `pip install -r requirements.txt`

## Spider

Unsere Spider ist im Ordner `crawler/` aufzufinden und lässt sich mit `python GrapplingInsiderSpyder.py` ausführen. Unser `.csv` Ergebnis dieses Crawlingvorgangs befindet sich in `daten/GrapplingInsider.csv`.

## HANA

Unsere Datenbank liegt in `daten/hana_export` und besteht aus folgenden drei Tabellen:

`GRAPPLING_INSIDER_CONTENT` mit den Spalten `url` und `title`

`GRAPPLING_INSIDER_CATEGORIES` mit den Spalten `url` und `category`, wobei es mehrere Einträge je `url` geben kann

`GRAPPLING_INSIDER_EXTERNAL_LINKS` mit den Spalten `url`, `external_link_url` und `external_link_text`, wobei letzteres der angezeigte text des Links darstellt

## Code

Die Aufgaben zu Praktikum 1 sind in `src/DarmstadtSpiderAnalysis.py` und `src/GrapplingInsiderSpiderAnalysis.py` hinterlegt. Praktikum 2 und 3 sind in `src/hana.py` aufzufinden. Praktikum 4 befindet sich in `src/Praktikum4_TopicModels-Memory-Patch.ipynb`.