"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Alle Parameter der Gleichungen stehen hier; kein Regler hat eine Zweierpotenz als Voreinstellung (Rechner-Artefakt): a*tau = 1,45; Mackey-Glass tau = 12 und 17;
rho = 20 und 28. Ausnahme ist das Preset „Von Hand“ mit a*tau = 1,0, dem Handwert des Beispiels x(1) = 0, x(2) = -1/2, x(3) = -1/6."""

import math

# --- Reglergrenzen und Voreinstellungen ---------------------------------------------------------------------------------------------------
AT_MIN, AT_MAX, AT_STEP, DEFAULT_AT = 0.10, 3.00, 0.01, 1.45     # a*tau (lineare Gleichung) und r*tau (Hutchinson), Experimente 1 bis 4
STUECKE_MIN, STUECKE_MAX, DEFAULT_STUECKE = 3, 8, 4                # Zahl der Stücke der Methode der Schritte (Experiment 1)
M_OPTIONEN = (1, 2, 3, 5, 7, 10, 20, 50, 100)                       # Schritte je Verzögerung (Experiment 4)
DEFAULT_M = 5
VERFAHREN = ("Euler", "Heun")
DEFAULT_VERFAHREN = "Euler"
MG_TAU_MIN, MG_TAU_MAX, MG_TAU_STEP, DEFAULT_MG_TAU = 6.0, 30.0, 0.5, 12.0
RHO_MIN, RHO_MAX, RHO_STEP, DEFAULT_RHO = 10, 160, 1, 20
SEED_MAX, DEFAULT_SEED = 999999, 35

# --- Lineare Gleichung und Hutchinson ------------------------------------------------------------------------------------------------------
M_STANDARD = 400          # Heun-Schritte je Verzögerung bei Hutchinson und bei der Wachstumsmessung
M_HAND = 1000             # Heun-Schritte je Verzögerung in Experiment 1 (Vergleich mit den exakten Stücken)
HUT_START = 0.5           # Vorgeschichte der verzögerten Logistik (konstant)
HUT_N_TAU = 2000          # Länge der Bahn in Verzögerungen (Experiment 3)
HUT_TAIL_TAU = 300        # Spannweite und Periode aus den letzten 300 Verzögerungen
HUT_FIT_RS = (1.58, 1.60, 1.62, 1.65, 1.70)   # Punkte der Geraden Spannweite^2 gegen r (Nähe der Grenze)
HUT_FIT_N_TAU = 3000
HUT_SPAN_NULL = 1e-3      # Spannweite unter dieser Schwelle gilt als „keine Dauerschwingung“
HUT_DRIFT = 0.02          # Ändert sich die Spannweite von einem Fenster der letzten 300 Verzögerungen zum vorigen um mehr als 2 %, ist die Bahn noch nicht eingeschwungen
EULER_UNSTABIL_AT = 1.45  # Wert aus dem Plan (Experiment 4); als Vergleichsstelle in der README

# --- Mackey-Glass ---------------------------------------------------------------------------------------------------------------------------
MG_BETA, MG_GAMMA, MG_N = 0.2, 0.1, 10.0
MG_H = 0.1                 # Zielschritt (Zeiteinheiten); je tau wird m = round(tau / h) ganzzahlig, der Schritt tau / m
MG_T_BURN = 5000.0         # Einschwingzeit vor Messung und Bild
MG_T_ZEIGE = 400.0         # Länge des gezeigten Zeitverlaufs
MG_T_MESS = 60000.0        # Messdauer des Exponenten live (Zeiteinheiten)
MG_TAU_STANDARD = 17.0     # der Wert des Plans (Messung V7b)
MG_LAMBDA_NULL = 0.0010    # Exponent unter dieser Schwelle gilt als „kein positiver Exponent messbar“ (Streuung der Messung, Abschnitt Befunde der README)

# --- Lorenz ---------------------------------------------------------------------------------------------------------------------------------
LOR_SIGMA, LOR_BETA = 10.0, 8.0 / 3.0
LOR_N_START, LOR_T_BURN, LOR_T_MESS, LOR_DT, LOR_RENORM, LOR_T_QR = 160, 60.0, 25.0, 0.01, 10, 10.0   # live: 160 Bahnen zugleich
LOR_N_DIV, LOR_D0, LOR_T_DIV_MAX, LOR_DT_DIV, LOR_T_BURN_DIV = 30, 1e-9, 40.0, 0.005, 60.0
LOR_PFAD_JEDE, LOR_T_PFAD_MIN = 4, 25.0
LOR_LAMBDA_CHAOS = 0.05    # lambda_1 darüber gilt als positiv (die Live-Messung streut um etwa 0,01)
LOR_RHO_BAND = (0.89, 0.91)   # Band von lambda_1 bei rho = 28 aus der Vormessung (L4), in der README als Messung des Plans

PRESETS_ZEIT = math.pi / 2     # Hopf-Grenze a*tau

# --- Farben ---------------------------------------------------------------------------------------------------------------------------------
COLOR_MAIN = "#2F6B65"
COLOR_ALT = "#c0504d"
COLOR_LINE = "#4c78a8"
COLOR_GREY = "rgba(130,130,130,0.8)"
COLOR_BAND = "rgba(76,120,168,0.18)"

# --- Presets (jedes legt alle Regler fest; fehlende Schlüssel meinen den Standardwert) -----------------------------------------------------
PRESETS = {
    "Unter der Grenze (Voreinstellung)": {
        "at": DEFAULT_AT, "stuecke": DEFAULT_STUECKE, "m": DEFAULT_M, "verfahren": DEFAULT_VERFAHREN,
        "mg_tau": DEFAULT_MG_TAU, "rho": DEFAULT_RHO, "seed": DEFAULT_SEED,
    },
    "Von Hand": {
        "at": 1.0, "stuecke": 4, "m": DEFAULT_M, "verfahren": DEFAULT_VERFAHREN,
        "mg_tau": DEFAULT_MG_TAU, "rho": DEFAULT_RHO, "seed": DEFAULT_SEED,
    },
    "Hinter der Grenze": {
        "at": 1.80, "stuecke": DEFAULT_STUECKE, "m": DEFAULT_M, "verfahren": DEFAULT_VERFAHREN,
        "mg_tau": DEFAULT_MG_TAU, "rho": DEFAULT_RHO, "seed": DEFAULT_SEED,
    },
    "Mackey–Glass": {
        "at": DEFAULT_AT, "stuecke": DEFAULT_STUECKE, "m": DEFAULT_M, "verfahren": DEFAULT_VERFAHREN,
        "mg_tau": MG_TAU_STANDARD, "rho": DEFAULT_RHO, "seed": DEFAULT_SEED,
    },
    "Lorenz": {
        "at": DEFAULT_AT, "stuecke": DEFAULT_STUECKE, "m": DEFAULT_M, "verfahren": DEFAULT_VERFAHREN,
        "mg_tau": DEFAULT_MG_TAU, "rho": 28, "seed": DEFAULT_SEED,
    },
}
PRESET_HELP = {
    "Unter der Grenze (Voreinstellung)": "a·τ = 1,45 liegt unter π/2 = 1,571: die lineare Gleichung klingt ab, die verzögerte Logistik schwingt nicht. Mackey–Glass mit τ = 12 und Lorenz mit ρ = 20 sind noch ruhig.",
    "Von Hand": "a·τ = 1,0: der Handwert der Methode der Schritte, x(1) = 0, x(2) = −1/2, x(3) = −1/6.",
    "Hinter der Grenze": "a·τ = 1,8 liegt über π/2: die lineare Gleichung wächst, die verzögerte Logistik schwingt dauerhaft (Spannweite etwa 2,1).",
    "Mackey–Glass": "τ = 17: der Standardfall der Mackey–Glass-Gleichung, bei dem die Messung einen kleinen positiven Exponenten findet.",
    "Lorenz": "ρ = 28: der klassische Wert des Lorenz-Systems mit positivem größtem Exponenten.",
}
