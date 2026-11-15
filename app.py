"""Verzögerung und Hopf-Grenze - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stetige Zeit mit Verzögerung: die Hopf-Grenze der linearen Gleichung (Lambertsche W-Funktion), die verzögerte Logistik hinter der Grenze,
der Zeitschritt als Fehlerquelle, Mackey-Glass und als Tab das Lorenz-System. Aufbau nach DEMO-PLAYBOOK §2.

Lauffähig mit: streamlit run app.py
"""

import math

import streamlit as st

import vzh_algorithm as A
import vzh_constants as C
import vzh_evaluation as E
import vzh_lorenz as L
from vzh_layout import dunkles_design, kopf_css
from vzh_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from vzh_visualization import (
    build_hut_fit,
    build_hut_zeit,
    build_kurve_wachstum,
    build_lorenz_abstand,
    build_lorenz_bahn,
    build_lorenz_gitter,
    build_mg_gitter,
    build_mg_zeit,
    build_phase,
    build_stuecke,
    build_wurzeln,
    build_zeitschritt,
)

st.set_page_config(page_title="Verzögerung und Hopf-Grenze – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _hand(a, n):
    return E.analyse_hand(a, n)


@st.cache_data(show_spinner=False)
def _kennlinie(a):
    return E.analyse_kennlinie(a)


@st.cache_data(show_spinner=False)
def _hut(r):
    return E.analyse_hutchinson(r)


@st.cache_data(show_spinner=False)
def _hut_fit():
    return E.analyse_hutchinson_fit()


@st.cache_data(show_spinner=False)
def _zeit(a, m, verfahren):
    return E.analyse_zeitschritt(a, m, verfahren)


@st.cache_data(show_spinner=False)
def _mg(tau, seed):
    return E.analyse_mackey_glass(tau, seed)


@st.cache_data(show_spinner=False)
def _lorenz(rho, seed):
    return E.analyse_lorenz(rho, seed)


@st.cache_data(show_spinner=False)
def _vorgerechnet():
    return E.lade_vorgerechnet()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in vzh_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⏳ Verzögerung und Hopf-Grenze")
st.markdown(
    """
Eine **Verzögerung** allein macht aus einer Gleichung, die ohne sie ruhig zum Gleichgewicht läuft, eine schwingende: Bei $x'(t) = -a\\,x(t-\\tau)$ liegt die Grenze
genau bei **$a\\tau = \\pi/2$**, und die Schwingung dort hat die Periode $4\\tau$. Dahinter schwingt die verzögerte Logistik dauerhaft (gemessen bis $r\\tau = 3$), und ihre Spannweite wächst mit $r\\tau$;
mit stärkerer Nichtlinearität und längerer Verzögerung (**Mackey–Glass**, $\\tau = 17$) misst die Demo einen positiven Lyapunov-Exponenten, obwohl die Gleichung nur eine Größe hat. Das **Lorenz-System** zeigt die zweite Art,
in stetiger Zeit chaotisch zu werden: nicht durch eine Verzögerung, sondern durch drei Größen. Alles gilt in diesen Modellen und bei diesen Parametern.

Weiter unten: sechs Abschnitte **🔬 Experiment** (von Hand rechnen, Kennlinie, hinter der Grenze, Zeitschritt, Mackey–Glass, Lorenz), **🚧 Wo die Annahmen enden** und der Expander
**📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Verzögerung", expanded=True):
    st.markdown(
        """
1. **Die Gleichung schaut in die Vergangenheit.** Bei $x'(t) = -a\\,x(t-\\tau)$ hängt die Änderung jetzt vom Wert vor der Zeit $\\tau$ ab; der Anfangszustand ist deshalb ein ganzer Verlauf auf $[-\\tau, 0]$, nicht ein Wert.
2. **Methode der Schritte.** Auf dem nächsten Abschnitt der Länge $\\tau$ ist der verzögerte Wert aus dem vorigen bekannt, die Gleichung wird dort zu einem gewöhnlichen Integral. So entsteht die Lösung Abschnitt für Abschnitt (Experiment 1).
3. **Wachsende oder abklingende Schwingung.** Der Ansatz $x = e^{st}$ gibt $s + a e^{-s\\tau} = 0$ mit den Lösungen $s = W_k(-a\\tau)/\\tau$. Liegt das größte $\\mathrm{Re}\\,s$ unter null, klingt die Lösung ab; bei $a\\tau = \\pi/2$ ist $s = i\\pi/(2\\tau)$, die Periode also $4\\tau$.
4. **Der Rechner kann die Grenze verschieben.** Euler mit wenigen Schritten je $\\tau$ rechnet die Gleichung schon unter $\\pi/2$ als instabil (Experiment 4).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Verzögerung und Rückkopplung**")
    at = st.slider("Produkt a·τ (bei Hutchinson: r·τ)", *bounds("at_slider"), step=C.AT_STEP, key="at_slider", format="%.2f",
                   help="Gilt für die Experimente 1 bis 4 und das Bild oben. Die Grenze der linearen Gleichung liegt bei π/2 = 1,571.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed (Startverlauf bei Mackey–Glass, Startwerte bei Lorenz)", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

at = float(at)
seed = int(seed)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

kennlinie = _kennlinie(at)
hut = _hut(at)
fit = _hut_fit()

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
m1.metric("Größtes Re s der Kennlinie", E.de(kennlinie.rechts.real, 4, plus=True), delta=f"{E.de(at - math.pi / 2, 3, plus=True)} gegen π/2", delta_color="off",
          help="Re s < 0: die lineare Gleichung klingt ab; Re s > 0: sie wächst.")
m2.metric("Hutchinson: Spannweite", E.de(hut.span, 3), delta="der letzten 300 Verzögerungen", delta_color="off")
m3.metric("Periode der Dauerschwingung", (E.de(hut.period, 3) + " τ") if hut.span >= C.HUT_SPAN_NULL and hut.eingeschwungen else "–", delta="Hutchinson, in Verzögerungen", delta_color="off")
st.plotly_chart(build_hut_zeit(hut), width="stretch", key="main_chart")
st.caption("Verzögerte Logistik $x'(t) = r\\,x(t)\\,(1 - x(t-\\tau))$ bei $r\\tau$ wie links gewählt, die letzten 40 Verzögerungen; gestrichelt das Gleichgewicht 1.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ab welchem Produkt aus Rückkopplung und Verzögerung kippt es in Schwingung?")
kind, text = E.verdict_kennlinie(kennlinie)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Von Hand rechnen (Methode der Schritte)")
st.caption("Lineare Gleichung $x'(t) = -a\\,x(t-\\tau)$ mit Vorgeschichte $x = 1$ auf $[-\\tau, 0]$; Produkt a·τ aus der Seitenleiste, Zeit in Einheiten von τ.")
stuecke = st.slider("Zahl der Abschnitte der Länge τ", *bounds("stuecke_slider"), key="stuecke_slider", help="Wie viele Abschnitte die Methode der Schritte aufbaut.")
hand = _hand(at, int(stuecke))
st.plotly_chart(build_stuecke(hand), width="stretch", key="hand_chart")
kind, text = E.verdict_hand(hand)
getattr(st, kind)(text)
st.caption(f"Die exakten Abschnitte (Polynome) gegen Heun mit {C.M_HAND} Schritten je τ: größter Unterschied an den Stützwerten {E.de(hand.abweichung, 6)}. "
           "Die gestrichelte Hüllkurve zeigt nur die Steigung $e^{\\mathrm{Re}\\,s\\cdot t}$, nicht die Höhe.")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Kennlinie und Lambertsche W-Funktion")
st.caption("Die Wurzeln $s_k = W_k(-a\\tau)/\\tau$ der Kennlinie $s + a e^{-s\\tau} = 0$; Produkt a·τ aus der Seitenleiste. Die Simulation misst die Wachstumsrate unabhängig davon.")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_wurzeln(kennlinie), width="stretch", key="wurzeln_chart")
with c2:
    st.plotly_chart(build_kurve_wachstum(kennlinie), width="stretch", key="kurve_chart")
kind, text = E.verdict_kennlinie(kennlinie)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Hinter der Hopf-Grenze (verzögerte Logistik)")
st.caption(f"Heun mit {C.M_STANDARD} Schritten je τ, Vorgeschichte konstant {E.de(C.HUT_START, 1)}, {C.HUT_N_TAU} Verzögerungen; Spannweite und Periode aus den letzten {C.HUT_TAIL_TAU}. "
           "Produkt r·τ aus der Seitenleiste.")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_phase(hut.x_zuvor, hut.x_jetzt, "x(t − τ)", "x(t)"), width="stretch", key="hut_phase")
with c2:
    st.plotly_chart(build_hut_fit(fit, at, hut.span), width="stretch", key="hut_fit")
kind, text = E.verdict_hutchinson(hut, fit)
getattr(st, kind)(text)
st.caption("Die Punkte der Geraden (r·τ = 1,58 bis 1,70) sind in dieser App immer dieselben; die Nullstelle ist eine **Messung** in diesem Modell, kein Wurzelgesetz der Amplitude.")

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Der Zeitschritt als Fehlerquelle")
st.caption("Dieselbe lineare Gleichung, gerechnet mit wenigen Schritten je τ. Produkt a·τ aus der Seitenleiste (voreingestellt 1,45, also unter π/2).")
c1, c2 = st.columns(2)
with c1:
    m = st.select_slider("Schritte je Verzögerung m", options=C.M_OPTIONEN, key="m_select")
with c2:
    verfahren = st.radio("Verfahren", C.VERFAHREN, key="verfahren_radio", horizontal=True)
zeit = _zeit(at, int(m), verfahren)
st.plotly_chart(build_zeitschritt(zeit), width="stretch", key="zeit_chart")
kind, text = E.verdict_zeitschritt(zeit)
getattr(st, kind)(text)
st.caption("Die Kurven kommen aus den Wurzeln des charakteristischen Polynoms der Rekursion (ein anderer Rechenweg als die Simulation); das Kreuz ist die Simulation.")

st.markdown("---")

# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Mackey–Glass (eine Größe, trotzdem chaotisch)")
st.caption(f"$x'(t) = \\beta\\,x(t-\\tau)/(1 + x(t-\\tau)^n) - \\gamma\\,x(t)$ mit $\\beta = {E.de(C.MG_BETA, 1)}$, $\\gamma = {E.de(C.MG_GAMMA, 1)}$, $n = {int(C.MG_N)}$; "
           f"Heun mit Schritt etwa {E.de(C.MG_H, 1)}, Einschwingzeit {int(C.MG_T_BURN)}, Messdauer {int(C.MG_T_MESS)} Zeiteinheiten.")
mg_tau = st.slider("Verzögerung τ", *bounds("mg_tau_slider"), step=C.MG_TAU_STEP, key="mg_tau_slider", format="%.1f")
with st.spinner("Rechnet die Bahn und den Exponenten (zwei Verfahren) …"):
    mg = _mg(float(mg_tau), seed)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_mg_zeit(mg), width="stretch", key="mg_zeit")
with c2:
    st.plotly_chart(build_phase(mg.x_zuvor, mg.x_jetzt, "x(t − τ)", "x(t)", C.COLOR_MAIN), width="stretch", key="mg_phase")
kind, text = E.verdict_mackey_glass(mg)
getattr(st, kind)(text)
st.caption(f"Rechenzeit dieser Messung beim ersten Aufruf: {E.de(mg.rechenzeit, 2)} s (beide Verfahren zusammen). Der Exponent streut mit dem Zufalls-Seed um etwa ±6 % (Spanne über 20 Zufalls-Seeds, siehe README): andere Startverläufe in der Seitenleiste ausprobieren.")
voraus = _vorgerechnet()
if voraus is not None:
    st.plotly_chart(build_mg_gitter(voraus["mackey_glass"], float(mg_tau), mg.lam_zweibahn), width="stretch", key="mg_gitter")
    st.caption(f"**Vorgerechnet** (tools/vorrechnen.py, Variationsgleichung, drei Zufalls-Seeds, {int(voraus['mackey_glass']['t_mess'])} Zeiteinheiten je Lauf): λ über τ. "
               "Eine Messung in diesem Modell, nicht gegen eine Veröffentlichung geprüft.")

st.markdown("---")

# --- Experiment 6 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 6: Tab Lorenz (die zweite Art, chaotisch zu werden)")
st.caption(f"$\\sigma = {int(C.LOR_SIGMA)}$, $\\beta = 8/3$; $\\rho$ regelbar. Live: {C.LOR_N_START} Bahnen zugleich, RK4 mit $dt = {E.de(C.LOR_DT, 2)}$, QR-Verfahren. "
           "Die Summe der drei Exponenten muss die Spur $-(\\sigma + 1 + \\beta)$ sein: das ist die Gegenprobe.")
rho = st.slider("ρ", *bounds("rho_slider"), step=C.RHO_STEP, key="rho_slider")
with st.spinner("Rechnet das Lyapunov-Spektrum und die Zwei-Bahn-Divergenz …"):
    lor = _lorenz(int(rho), seed)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_lorenz_bahn(lor), width="stretch", key="lor_bahn")
with c2:
    st.plotly_chart(build_lorenz_abstand(lor), width="stretch", key="lor_abstand")
kind, text = E.verdict_lorenz(lor)
getattr(st, kind)(text)
st.caption(f"Rechenzeit dieser Messung beim ersten Aufruf: {E.de(lor.rechenzeit, 2)} s. Live gemessene Exponenten: "
           f"λ₁ = {E.de(lor.lam[0], 3, plus=True)}, λ₂ = {E.de(lor.lam[1], 3, plus=True)}, λ₃ = {E.de(lor.lam[2], 3, plus=True)}; "
           f"Streuung der Mittel λ₁ ± {E.de(lor.sem[0], 3)}. Die Bahn ist die des ersten Startwerts, ρ_H = {E.de(L.hopf_rho(), 3)} (Hopf der Fixpunkte).")
if voraus is not None:
    st.plotly_chart(build_lorenz_gitter(voraus["lorenz"], int(rho), lor.lam[0], L.hopf_rho()), width="stretch", key="lor_gitter")
    lg = voraus["lorenz"]
    st.caption(f"**Vorgerechnet** (tools/vorrechnen.py): λ₁ über ρ, {lg['n_start']} Bahnen mit je {int(lg['t_mess'])} Zeiteinheiten Messdauer, RK4 mit dt = {E.de(lg['dt'], 3)}. "
               "Zwischen ρ = 24,06 und ρ_H ist Chaos neben stabilen Fixpunkten bekannt, hier aber nicht gemessen.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Feste Verzögerung** (die Lieferzeit schwankt nicht) | Bei streuender Verzögerung (Erlang-Verteilung) verschiebt sich die Grenze; hier bleibt τ fest. | Rückkopplung mit streuender Lieferzeit |
| **Linearisierung am Gleichgewicht** (Hopf-Grenze aus $s + a e^{-s\\tau} = 0$) | Nur die Grenze folgt aus der Linearisierung. Die Amplitude dahinter ist hier eine Messung (Nullstelle der Geraden Spannweite² gegen r·τ bei etwa 1,569), kein Beweis des Wurzelverlaufs. | Messung hier |
| **Mackey–Glass nur bei τ = 17 geprüft** | Bei jedem anderen τ steht der Exponent ohne Referenz da. Er streut mit dem Zufalls-Seed um etwa ±6 % (Spanne über 20 Zufalls-Seeds, siehe README). | – |
| **Ein Modell je Gleichung** | Hutchinson, Mackey–Glass und Lorenz sind je eine Familie; andere Nichtlinearitäten können anders reagieren. | – |
| **Rechner-Zeitschritt klein genug** | Mit zu wenigen Schritten je τ rechnet schon die Rekursion instabil (Experiment 4). | [Schrittweite als Dynamik](https://github.com/sebastian-hanisch/schrittweite-dynamik-demo) |
| **Lorenz: Messdauer begrenzt** | Live sind es 160 kurze Bahnen: λ₁ streut um etwa ±0,01; Fenster bei ρ = 100 und 160 sind nur als λ₁ ≈ 0 festgestellt, nicht als Periode. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben sind die Verzögerung $\tau > 0$, die Stärke der Rückkopplung ($a$, $r$) und ein Anfangsverlauf auf $[-\tau, 0]$. Gesucht ist der weitere Verlauf $x(t)$ und die Frage, ob er abklingt, dauerhaft schwingt oder chaotisch wird.

| Symbol | Bedeutung |
|---|---|
| $x(t)$ | Zustand zur Zeit $t$ |
| $\tau$ | feste Verzögerung (in Experiment 1 bis 4 die Zeiteinheit) |
| $a$, $r$ | Stärke der Rückkopplung (lineare Gleichung, Hutchinson); die Experimente 1 bis 4 nutzen das Produkt $a\tau$ bzw. $r\tau$ |
| $s$, $W_k$ | Wurzel der Kennlinie; Zweig $k$ der Lambertschen W-Funktion |
| $\beta, \gamma, n$ | Parameter von Mackey–Glass ($0{,}2$; $0{,}1$; $10$) |
| $\sigma, \rho, \beta_L$ | Parameter von Lorenz ($\sigma = 10$, $\beta_L = 8/3$, $\rho$ regelbar) |
| $\lambda$ | Lyapunov-Exponent (Wachstumsrate eines kleinen Abstands) |
| $h = \tau/m$ | Zeitschritt, $m$ Schritte je Verzögerung |

**Lineare Gleichung** (mit konstanter Vorgeschichte $x \equiv 1$ auf $[-\tau, 0]$):
$$x'(t) = -a\,x(t-\tau).$$
Der Ansatz $x = e^{st}$ gibt die Kennlinie $s + a e^{-s\tau} = 0$ mit den Lösungen $s_k = W_k(-a\tau)/\tau$. Auf der imaginären Achse ($s = i\omega$) folgt aus $(i\omega\tau)\,e^{i\omega\tau} = -a\tau$ die Bedingung $\omega\tau = \pi/2$ und $a\tau = \pi/2$; die Periode ist dort $2\pi/\omega = 4\tau$.

**Verzögerte Logistik (Hutchinson)** (Vorgeschichte konstant $0{,}5$):
$$x'(t) = r\,x(t)\,\bigl(1 - x(t-\tau)\bigr).$$
Das Gleichgewicht $x = 1$ hat dieselbe Linearisierung $u' = -r\,u(t-\tau)$, also die Grenze $r\tau = \pi/2$.

**Mackey–Glass** (Vorgeschichte $1{,}2$ plus Normalrauschen der Standardabweichung $0{,}05$):
$$x'(t) = \frac{\beta\,x(t-\tau)}{1 + x(t-\tau)^n} - \gamma\,x(t).$$
Am Gleichgewicht $x^* = (\beta/\gamma - 1)^{1/n} = 1$ lautet die Linearisierung $u' = -\gamma u - b\,u(t-\tau)$ mit $b = 0{,}4$; die Hopf-Grenze liegt bei $\tau_c = \arccos(-\gamma/b)/\sqrt{b^2 - \gamma^2} = 4{,}708$.

**Lorenz** ($\sigma = 10$, $\beta_L = 8/3$):
$$x' = \sigma(y - x), \qquad y' = x(\rho - z) - y, \qquad z' = xy - \beta_L z.$$
Die Summe der drei Lyapunov-Exponenten ist die Spur der Jacobi-Matrix, $-(\sigma + 1 + \beta_L) = -41/3$. Die Fixpunkte $C_\pm$ haben ein Paar auf der imaginären Achse bei $\rho_H = \sigma(\sigma + \beta_L + 3)/(\sigma - \beta_L - 1) = 470/19$.

**Verfahren.** Heun: $x_{i+1} = x_i + \tfrac{h}{2}\,(k_1 + k_2)$ mit $k_1 = f(x_i, x_{i-m})$ und $k_2 = f(x_i + h k_1, x_{i-m+1})$. Innerhalb eines Blocks von $m$ Schritten ist der verzögerte Wert bekannt; die Gleichung ist dann in $x$ linear, und ein Block wird vektorisiert gerechnet (Summe, Produkt oder lineare Rekursion). Euler: $x_{i+1} = x_i + h\,f(x_i, x_{i-m})$. Der Lyapunov-Exponent von Mackey–Glass wird mit zwei Bahnen im Abstand $10^{-8}$ (Abstand nach jedem Block auf $10^{-8}$ zurückgesetzt) und mit der Variationsgleichung gemessen, der von Lorenz mit dem QR-Verfahren.

*Modellgrenze:* Dies ist die vollständige Beschreibung der vier Gleichungen und ihrer Parameter, wie die Demo sie rechnet; Beweise der Hopf-Amplitude, der Chaos-Aussagen und der Literaturwerte stehen hier nicht.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)

sync_query_params({
    "at_slider": at, "stuecke_slider": int(stuecke), "m_select": int(m), "verfahren_radio": verfahren,
    "mg_tau_slider": float(mg_tau), "rho_slider": int(rho), "seed_input": seed,
})
