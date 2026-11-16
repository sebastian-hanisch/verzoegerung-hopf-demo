# Verzögerung und Hopf-Grenze – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-verzoegerung-hopf-demo.streamlit.app/)**

Teil der Linie **Dynamische Systeme** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“: stetige Zeit mit Verzögerung. Die Demo zeigt, was eine Verzögerung allein leistet, und wie viel davon erst der Rechner (der Zeitschritt) erzeugt. Dazu kommt das Lorenz-System als zweite Art, in stetiger Zeit chaotisch zu werden. Verwandt sind [Logistische Abbildung](https://github.com/sebastian-hanisch/logistische-abbildung-demo) (Verlust der Stabilität eines Fixpunkts in Schritten), [Chaos messen](https://github.com/sebastian-hanisch/lyapunov-chaos-demo) (Lyapunov-Exponent einer Abbildung), [Schrittweite als Dynamik](https://github.com/sebastian-hanisch/schrittweite-dynamik-demo) und [Chaos im Lager](https://github.com/sebastian-hanisch/lager-chaos-demo) (Verzögerung plus Klemmung).

## Warum dieses Problem

Die stetige logistische Gleichung $x' = r\,x\,(1-x)$ ohne Verzögerung läuft für jeden Start x₀ > 0 monoton zum Gleichgewicht 1 (sie hat eine geschlossene Lösung). Erst die **Verzögerung** macht den Fixpunkt instabil, und zwar über ein Paar auf der imaginären Achse (Hopf) und nicht über den Wert −1 wie bei einer Abbildung in Schritten. Kernfrage: Ab welchem Produkt aus Rückkopplung und Verzögerung kippt es in Schwingung, wie wächst die Schwingung dahinter, und was davon sieht man nur, weil der Rechner einen zu groben Zeitschritt nimmt?

## Modell

Vier Gleichungen, jede mit ihrer Vorgeschichte (die vollständige Beschreibung steht im 📐-Expander der App):

* **Lineare Gleichung** $x'(t) = -a\,x(t-\tau)$, Vorgeschichte 1 auf $[-\tau, 0]$. Die Kennlinie $s + a e^{-s\tau} = 0$ hat die Lösungen $s_k = W_k(-a\tau)/\tau$ (Lambertsche W-Funktion); die Grenze liegt bei $a\tau = \pi/2$ mit der Periode $4\tau$.
* **Verzögerte Logistik (Hutchinson)** $x'(t) = r\,x(t)\,(1-x(t-\tau))$, Vorgeschichte konstant 0,5. Bei $x = 1$ ist die Linearisierung $u' = -r\,u(t-\tau)$, also dieselbe Grenze $r\tau = \pi/2$ (eigene Herleitung, nicht aus einer gelesenen Quelle).
* **Mackey–Glass** $x'(t) = \beta\,x(t-\tau)/(1 + x(t-\tau)^n) - \gamma\,x(t)$ mit $\beta = 0{,}2$, $\gamma = 0{,}1$, $n = 10$, Vorgeschichte 1,2 plus Normalrauschen (Standardabweichung 0,05, SplitMix64). Gleichgewicht $x^* = 1$, Linearisierung $u' = -0{,}1\,u - 0{,}4\,u(t-\tau)$, Hopf-Grenze des Gleichgewichts bei $\tau_c = 4{,}708$ (geschlossen).
* **Lorenz** $x' = \sigma(y-x)$, $y' = x(\rho-z)-y$, $z' = xy-\beta z$ mit $\sigma = 10$, $\beta = 8/3$; Hopf der Fixpunkte bei $\rho_H = 470/19 = 24{,}737$.

Alle Aussagen gelten **in diesen Modellen bei diesen Parametern**.

## Methodik

* **Heun-Verfahren, blockweise vektorisiert.** Innerhalb eines Blocks der Länge $\tau$ ist der verzögerte Wert bekannt, und die Gleichung ist in $x(t)$ allein linear. Ein Block ist dann eine Summe (lineare Gleichung), ein Produkt (Hutchinson) oder eine lineare Rekursion (Mackey–Glass, `scipy.signal.lfilter`). Das ergibt dieselben Zahlen wie das Schritt-für-Schritt-Verfahren (Unterschied höchstens 1e-11, `tests/test_algorithm.py`) und rechnet viel schneller.
* **Wachstumsrate der Simulation:** Steigung von $\ln|x|$ über die Extrema der Lösung (parabolisch verfeinert); bei monotonem Abklingen über die Beträge bei ganzen $\tau$.
* **Mackey–Glass-Exponent:** zwei Wege auf derselben Bahn. *Zweibahn-Verfahren:* zwei Bahnen im Abstand $10^{-8}$ (Norm des Fensters aus $m+1$ Werten), der Abstand wird nach jedem Block der Länge $\tau$ auf $10^{-8}$ zurückgesetzt. *Variationsgleichung:* der Tangentialverlauf mit der Ableitung des Verzögerungsterms von Hand. Die Live-Messung der App dauert 60 000 Zeiteinheiten nach 5000 Einschwingzeit.
* **Lorenz-Spektrum:** QR-Verfahren mit RK4 für Zustand und drei Tangentialvektoren. Damit die Live-Rechnung kurz bleibt, laufen **160 kurze Bahnen gleichzeitig** (Einschwingzeit 60, danach 10 für die Richtungen, dann 25 Zeiteinheiten Messdauer, dt = 0,01); das Spektrum ist das Mittel über die Bahnen, die Streuung des Mittels steht daneben.
* **Vorgerechnet** (nicht live): die Gitter $\lambda(\tau)$ bei Mackey–Glass und $\lambda_1(\rho)$ bei Lorenz, erzeugt von `tools/vorrechnen.py` (etwa 2 Minuten) und als `vzh_vorgerechnet.json` im Repo; die App kennzeichnet sie als vorgerechnet.
* **Zufall** nur über den ganzzahligen SplitMix64 (`vzh_rng.py`, kopiert aus `saal-abstand-demo`); der Zufalls-Seed in der Seitenleiste wirkt auf den Startverlauf bei Mackey–Glass und die Startwerte bei Lorenz.

## Befunde (gemessen, keine Behauptungen)

Zahl → Test: die Tests stehen in `tests/test_claims.py` (Band oder Zählung); „Läufe“ nennt die Zahl der Zufalls-Seeds bzw. Startverläufe.

| Nr | Frage | Befund | Test |
|---|---|---|---|
| B1 | Grenze der linearen Gleichung | Methode der Schritte (a·τ = 1, Vorgeschichte 1): x(1) = 0, x(2) = −1/2, x(3) = −1/6, x(4) = 5/24. Die gemessene Wachstumsrate (Heun, 400 Schritte je τ) stimmt mit Re W₀(−a·τ) in 7 von 7 Werten (0,5; 1,0; 1,4; 1,55; 1,6; 1,7; 2,0) auf höchstens 1,2e-6. Bei a·τ = π/2 ist W₀ rein imaginär, die Periode 4 τ; bei a·τ = 1 sind s = −0,318 ± 1,337 i (Periode 4,70 τ), bei 2 sind es +0,173 ± 1,674 i. Der Verlauf ist genau für a·τ ≤ 1/e = 0,368 monoton. | `test_b1_*` |
| B2 | Hutchinson hinter der Grenze | Heun, 400 Schritte je τ, Vorgeschichte 0,5, 2000 Verzögerungen, Spannweite der letzten 300: unter 1e-6 bei r·τ = 1,40 und 1,55; **0,786** bei 1,60; 1,276 bei 1,65; 2,101 bei 1,8; 3,503 bei 2,2; 7,582 bei 3,0. Gegenprobe mit 20 Anfangsverläufen und 3 Schrittweiten (60 Läufe): bei 1,55 in 60 von 60 Läufen höchstens 1e-6 (gemessen 3,5e-8), bei 1,60 in 60 von 60 zwischen 0,785 und 0,787. Direkt unter π/2 ist die Spannweite der letzten 300 von 2000 Verzögerungen kein Beleg für eine Dauerschwingung: bei r·τ = 1,57 (unter π/2) liefert dieselbe Rechnung noch 0,083 (Periode 4,001), eine Transiente (nach der Prüfung 0,0035 nach 10 000 τ und 0,0 nach 30 000 τ); die App meldet dort „noch nicht entschieden“, wenn sich die Spannweite von einem Fenster zum vorigen um mehr als 2 % ändert. | `test_b2_*` |
| B3 | Amplitude nahe der Grenze | Fünf Punkte r·τ = 1,58 bis 1,70 (3000 Verzögerungen): Spannweite² liegt auf einer Geraden mit Nullstelle bei **1,5691** (π/2 = 1,5708); die ersten vier Punkte allein: 1,5700. Periode der Dauerschwingung 4,006 (bei 1,58), 4,020, 4,035, 4,057, 4,096 τ (bei 1,70). Das ist eine **Messung** in diesem Modell, kein Beleg für ein Wurzelgesetz der Amplitude. | `test_b2_gerade_spannweite_quadrat_nullstelle_und_perioden` |
| B4 | Zeitschritt | Euler-Grenze 2m·sin(π/(4m+2)): 1,000 (m = 1), 1,236 (2), 1,423 (5), 1,495 (10), 1,563 (100). Bei a·τ = 1,45 (Re s = −0,0568) rechnet Euler mit m = 1, 2, 3 und 5 Schritten je τ die Gleichung als instabil (Rate +0,186; +0,093; +0,051; +0,012), mit m = 7 und 10 als stabil (−0,006; −0,021). Heun rechnet bei a·τ = 1,45 schon mit m = 1 stabil (−0,161, das 2,83-fache der richtigen Rate), mit m = 5 −0,062. | `test_b3_*` |
| B5 | Mackey–Glass, Grenze und Gleichgewicht | τ_c = 4,7082; bei τ = 17 hat die Linearisierung die Wurzel mit größtem Realteil +0,0413 (Gleichgewicht instabil). | `test_b4_hopf_grenze_und_wurzel_bei_tau_17` |
| B6 | Mackey–Glass, Exponent bei τ = 17 | Zweibahn-Verfahren, Heun mit Schritt 0,1: über **20 Zufalls-Seeds** und 12 000 Zeiteinheiten Mittel 0,0052, Streuung 0,0003 (etwa 6 %), Spanne 0,0047 bis 0,0057 (eine unabhängige Nachrechnung mit anderen Zufalls-Seeds fand eine etwas größere Streuung: 0,0004, etwa 8 %); über 20 Zufalls-Seeds und 60 000 Zeiteinheiten (Live-Messung der App) Mittel 0,0051, Streuung 0,0001, Spanne 0,0049 bis 0,0054 (−5 % bis +6 % um das Mittel; die Nachrechnung mit anderen Zufalls-Seeds: 0,0002, etwa 4 %), Vorhersagezeit 1/λ im Mittel 195 Zeiteinheiten. Spanne der Lösung 0,42 bis 1,32. Zweibahn-Verfahren und Variationsgleichung stimmen auf 2e-6 überein. | `test_b4_exponent_*`, `test_b4_wertebereich_*`, Orakel `test_mg_zweibahn_gegen_variationsgleichung` |
| B7 | Mackey–Glass, Gitter in τ (vorgerechnet) | Variationsgleichung, 3 Zufalls-Seeds, 40 000 Zeiteinheiten je Lauf: für τ = 5 bis 16 in 12 von 12 Werten \|λ\| < 1e-4; für τ = 17 bis 30 in 13 von 14 Werten λ > 0,0005 (nicht bei τ = 19: −1e-5; bei τ = 21 liegt λ mit 0,0010 an der Schwelle der App, je Zufalls-Seed 0,0007 bis 0,0015); Maximum 0,0104 bei τ = 24. Nicht gegen eine Veröffentlichung geprüft. | `test_b4_vorgerechnetes_gitter_in_tau` |
| B8 | Lorenz, Summe der Exponenten | In allen 22 vorgerechneten Werten von ρ ist die Summe der drei Exponenten gleich der Spur −41/3 = −13,667 (auf 5e-4); live in 8 von 8 Werten (ρ = 10, 20, 28, 40, 60, 100, 150, 160) auf 2e-3. Der Exponent in Flussrichtung ist in den 15 chaotischen Werten des Gitters betragsmäßig unter 0,003. | `test_b5_vorgerechnete_summe_*`, `test_b6_*` |
| B9 | Lorenz, λ₁ bei ρ = 28 | Vorgerechnet (100 Bahnen, 100 Zeiteinheiten Messdauer, dt = 0,005): 0,902 ± 0,002; bei dt = 0,01, 0,005, 0,0025: 0,904; 0,908; 0,903. Live (160 Bahnen, über 5 Zufalls-Seeds): 0,896 bis 0,917, Mittel 0,906. Die Vormessung des Plans nannte ein Band von 0,89 bis 0,91; ein Literaturvergleich wurde nicht gemacht. | `test_b5_vorgerechnetes_lambda1_*`, `test_b5_live_lambda1_*` |
| B10 | Lorenz, λ₁ über ρ (vorgerechnet) | Von 22 Werten ρ = 5 bis 160 haben 15 ein λ₁ über 0,05 (ρ = 25, 28, 30, 35, 40, 45, 50, 60, 70, 80, 90, 110, 120, 130, 140); keines haben ρ = 5, 10, 15, 20 (Fixpunkte) und 100, 150, 160. Dort ist λ₁ = −0,0025 (100), +0,0014 (150), +0,0013 (160). Gegenprobe: in 300 von 300 Bahnen (je 100 für ρ = 100, 150, 160; Einschwingzeit 50, Messdauer 100, dt = 0,01) ist λ₁ höchstens 0,03 (gemessen höchstens 0,019). Bei 100, 150 und 160 ist also kein Chaos gemessen; ob es eine Periode ist, wurde nicht geprüft. Chaos über ρ = 100 hinaus gibt es dennoch (ρ = 110 bis 140). | `test_b5_vorgerechnetes_lambda1_gitter_*`, `test_b5_kein_chaos_bei_100_150_160_*` |
| B11 | Lorenz, Vorhersagehorizont bei ρ = 28 | Zwei Bahnen im Anfangsabstand 1e-9, je 30 Startwerte, 5 Zufalls-Seeds: Zeit bis zum Abstand 1 im Mittel 21,3 bis 22,4 je Zufalls-Seed, einzelne Paare 16 bis 29 Zeiteinheiten (Zufalls-Seeds 1, 2, 3, 35, 36; bei Zufalls-Seed 37 bis 15,1); aus λ₁ = 0,90 folgt ln(10⁹)/0,90 = 23,0. Bei ρ = 20 erreicht kein Paar den Abstand 1. | `test_b5_vorhersagehorizont_*`, `test_b5_zwei_bahnen_bei_rho_20_*` |
| B12 | Hopf der Lorenz-Fixpunkte | ρ_H = σ(σ+β+3)/(σ−β−1) = 470/19 = 24,737, aus den Eigenwerten per Bisektion auf 1e-6 gleich; Frequenz dort 9,6245; Eigenwerte −13,667 und ±9,6245 i. Der Bereich 24,06 bis ρ_H ist hier **nicht gemessen**. | `test_b5_hopf_schwelle_und_spur`, Orakel (Bisektion, Fraktionen) |

„Jede Zahl steht in den Tests“ gilt für diese Tabelle; die Zahlen der Abschnitte „Befunde und Korrekturen“ und „Ehrliche Grenzen“ stehen, wo nicht anders genannt, in denselben Tests oder sind als Messung dieser README gekennzeichnet.

## Befunde und Korrekturen gegenüber dem Plan

* **Der Ausreißer bei a·τ = 1,7 ist geklärt.** Die Vormessung nannte +0,05567 gegen +0,05635 (Abweichung 0,0007). Ursache: Das Feld im Messskript war um ein Element zu lang (`numpy.empty`), das letzte Element blieb unbeschrieben (Speicherinhalt, hier 0) und zählte als Extremum am Ende der Zeitreihe; bei 1,7 verschob dieser Hebelpunkt die Steigung, bei den anderen Werten fiel er kaum ins Gewicht. Die Messung im Bau (Feld genau so lang wie die Lösung, Extrema parabolisch verfeinert) stimmt in 7 von 7 Werten auf 1,2e-6 mit der W-Funktion überein. Die Vormessungszahl war an dieser Stelle unbrauchbar.
* **Die Streuung des Mackey–Glass-Exponenten ist größer als im Plan.** Der Plan nannte, der Exponent ändere sich über beide Zeitschritte und beide Startverläufe um höchstens 0,00006 (rund 1 %). Im Bau streuen 20 Zufalls-Seeds bei 12 000 Zeiteinheiten mit 0,0003 (Spanne 0,0047 bis 0,0057), also etwa 6 % des Mittels; erst bei 60 000 Zeiteinheiten sind es 0,0001 (etwa 3 %). Der Planwert 0,00533 (35 000 Zeiteinheiten, ein Startverlauf) liegt in der Spanne; das Band 0,0050 bis 0,0056 des Plans war für einen einzelnen Lauf zu eng. Die Live-Verdikte der App nennen darum die Streuung.
* **Das Zweibahn-Verfahren setzt den Abstand nach jedem Block τ zurück**, nicht nach jeder Zeiteinheit wie die Vormessung. Zweibahn-Verfahren und Variationsgleichung (zwei Wege) stimmen auf 2e-6 überein (Orakel-Test).
* **Der Lorenz-Tab rechnet 160 kurze Bahnen gleichzeitig statt einer langen Bahn** (Live-Rechnung unter etwa 2 Sekunden). Die Streuung des Mittels (±0,006 bei λ₁) steht in der App; die langen Läufe sind vorgerechnet. Das Gitter zeigt Neues: Bei ρ = 110 bis 140 ist das System wieder chaotisch (λ₁ = 1,4 bis 1,6), bei 100, 150 und 160 nicht. Die Aussage des Plans „bei ρ = 100 und 160 kein Chaos“ bleibt richtig, die naheliegende Lesart „über ρ = 100 kein Chaos“ ist falsch.
* **Kurze Messungen täuschen bei ρ = 100.** Mit Einschwingzeit 30 lag das live gemessene λ₁ in 3 Zufalls-Seeds bei +0,016 bis +0,042, mit 60 bei +0,011 bis +0,016 (der langen Messung nach ist es −0,0025): einzelne Startwerte sind dort noch kurz chaotisch. Die App nimmt 60. Beim Vorhersagehorizont erreicht bei ρ = 100 in 5 Zufalls-Seeds (1, 2, 3, 35, 36) mit je 30 Paaren genau 1 von 150 Paaren den Abstand 1 (nach 12,6 Zeiteinheiten, Zufalls-Seed 35; bei den Seeds 35 bis 39 sind es 2 von 150): das ist vermutlich ein Einschwingvorgang (nicht gesondert gemessen).
* **Heun hat ein Orakel.** Der Plan wollte Heun bei kleinem m „nur live gemessen, ohne Orakel“ kennzeichnen; die Wurzeln des charakteristischen Polynoms der Heun-Rekursion sind ein zweiter Rechenweg und stimmen mit der Simulation auf 2e-3 überein (Test über 42 Kombinationen).
* **Experiment 4** nutzt das gemeinsame Produkt a·τ der Seitenleiste (voreingestellt 1,45) statt eines festen Werts; die Euler-Grenzen des Plans (1,000; 1,236; 1,423; 1,495) bestätigen sich.
* **Presets:** „Von Hand“ setzt a·τ = 1,0, den Handwert des Beispiels, obwohl 1 eine Zweierpotenz ist (Absicht: Handrechnung); alle anderen Voreinstellungen vermeiden Zweierpotenzen (a·τ = 1,45; τ = 12 und 17; ρ = 20 und 28). Voreingestellt sind die ruhigen Fälle (Mackey–Glass τ = 12, Lorenz ρ = 20); die Presets „Mackey–Glass“ und „Lorenz“ führen zu den chaotischen.
* **Neu gegenüber dem Plan:** die vorgerechneten Gitter $\lambda(\tau)$ und $\lambda_1(\rho)$ (Plan Z5: „nur live“), die Zählprobe „kein Chaos bei ρ = 100, 150, 160“ und die Wurzel der Mackey–Glass-Linearisierung bei beliebigem τ über die W-Funktion statt über Newton-Startwerte (das Newton-Verfahren dient als Gegenprobe).

## Ehrliche Grenzen

* **Mackey–Glass ist nur bei τ = 17 ein Plan-Befund**; alle anderen τ der App und des vorgerechneten Gitters stehen ohne Literaturvergleich da (Farmer u. a. nicht gelesen). Der Exponent streut mit dem Zufalls-Seed (siehe B6).
* **Die Amplitude hinter der Hopf-Grenze** ist nur über fünf Punkte gemessen; der Wurzelverlauf der Hopf-Normalform (aus dem Gedächtnis bekannt, nicht geprüft) ist nicht belegt.
* **Lorenz:** die Live-Messung streut (λ₁ ±0,01); der Bereich $24{,}06 < \rho < \rho_H$ (Chaos neben stabilen Fixpunkten, aus dem Gedächtnis bekannt) ist nicht gemessen; die Fenster bei ρ = 100, 150 und 160 sind nur als λ₁ ≈ 0 festgestellt, nicht als Periode; das Gitter hat die Schrittweite 5 bis 10 in ρ.
* **Eine Gleichung je Familie:** Hutchinson, Mackey–Glass und Lorenz sind je ein Modell; andere Nichtlinearitäten können anders reagieren.
* **Nicht mit echtem Mausklick im Browser getestet:** die Plotly-Diagramme haben keine Auswahl; die App wurde mit Streamlits AppTest und einem Start auf Skriptfehler geprüft.
* **Hutchinson-Start:** konstante Vorgeschichte 0,5; andere Anfangsverläufe nur in der Gegenprobe (B2).

## Verifikation

* **Kern:** Methode der Schritte gegen Handwerte und eine eigene Bruchrechnung; Blockfassung gegen Schritt-für-Schritt (lineare Gleichung, Hutchinson, Mackey–Glass); Euler-Grenze gegen die Wurzeln des Polynoms; Handwerte von Mackey–Glass (x\* = 1, b = 0,4, τ_c = 4,7082) und Lorenz (Spur, ρ_H, Eigenwerte); SplitMix64 gegen den Referenzwert der Startausgabe.
* **Orakel** (`tests/test_oracle_verzoegerung_hopf.py`, anderer Rechenweg): Wachstumsrate gegen `scipy.special.lambertw`; Simulation gegen die Wurzeln des charakteristischen Polynoms der Rekursion (`numpy.roots`, 42 Kombinationen aus Verfahren, m und a·τ); Methode der Schritte gegen Brüche; Heun-Ordnung 2; Hutchinson gegen die Linearisierung (Kleinstabweichung wächst wie Re W₀); Mackey–Glass: Zweibahn-Verfahren gegen Variationsgleichung, Variationsgleichung gegen zentrale Differenz, Wurzel gegen Newton aus einem Gitter von Startwerten; Lorenz: Summe der Exponenten gegen die Spur, Jacobi-Eigenwerte gegen das charakteristische Polynom, RK4 gegen `scipy.integrate.solve_ivp` (DOP853) und Ordnung 4, Hopf-Schwelle per Bisektion.
* **Eigenschaften** (`tests/test_eigenschaften_verzoegerung_hopf.py`): Fortsetzbarkeit, Skalieren der Vorgeschichte (nur linear), Positivität von Hutchinson, Lorenz-Symmetrie, gleicher Zufalls-Seed gleiche Ausgabe.
* **Randfälle** (`tests/test_edge_cases.py`): m = 1, ein Abschnitt, ungültige Eingaben, a·τ = π/2 nur im Band, Euler bei m = 1 und a·τ = 1 genau auf der Grenze, Verzögerung größer als die Messdauer, Reglergrenzen von τ und ρ.
* **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede Zahl im Text an handgesetzten Zeilen; jedes Preset durchläuft sein erwartetes Verdikt.
* **Oberfläche:** AppTest (jeder Button, jeder Regler am Minimum und Maximum, jedes Preset mit seiner Hauptaussage, Permalink inklusive Begrenzung), Layout (Schnellstart in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, Experiment-Regler im Hauptbereich), Footer, Laufzeit-Importe.
* **Testzahl:** 233 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (sechs Experimente, Kernfrage, Grenzen, Formeln) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, scipy, plotly |
| `vzh_algorithm.py` | Kern: lineare Gleichung, Methode der Schritte, Kennlinie, Hutchinson, Mackey–Glass |
| `vzh_lorenz.py` | Tab Lorenz: Spektrum (QR, gleichzeitige Bahnen), Zwei-Bahn-Divergenz, Hopf-Schwelle |
| `vzh_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `vzh_evaluation.py` | Kennzahlen und Verdikte der sechs Experimente, Laden der vorgerechneten Reihen |
| `vzh_layout.py` | Fester Kopf (Stylesheet) |
| `vzh_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `vzh_rng.py` | SplitMix64 (kopiert aus `saal-abstand-demo`) |
| `vzh_visualization.py` | Plotly-Abbildungen (Achsen gesperrt, deutsche Zahlenschreibweise) |
| `vzh_vorgerechnet.json` | vorgerechnete Reihen (Lorenz λ₁(ρ), Mackey–Glass λ(τ)) |
| `tools/vorrechnen.py` | erzeugt `vzh_vorgerechnet.json` (`--schnell` nur zum Ausprobieren) |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Claims, Randfälle, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, vorgerechnete Reihen |

## Bewusst nicht umgesetzt

* **PDF-Export** (die Demo ist ein Experimentierfeld ohne einzelnes Ergebnisblatt).
* **Amplituden-Wurzelgesetz** und der **Bereich 24,06 < ρ < ρ_H** bei Lorenz (offen gelassen, siehe Ehrliche Grenzen).
* **Lotka–Volterra und SIR** als Tab (andere Demos der Website behandeln sie, ohne Überschneidung mit Hopf, Mackey–Glass oder Lorenz).
* **Zeitumkehr-Skalierung** als Test: die lineare Gleichung und Hutchinson sind im Code bereits in Einheiten von τ geschrieben, die Eigenschaft wäre leer.
* **Literaturvergleich** der Exponenten (Mackey–Glass, Lorenz): die Quellen wurden nicht gelesen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die vorgerechneten Reihen neu erzeugen: `python tools/vorrechnen.py` (etwa 2 Minuten).

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Dynamische Systeme: Ruhepunkt, Kreislauf, Chaos](https://sebastianhanisch.net/konzepte-dynamische-systeme.html).
