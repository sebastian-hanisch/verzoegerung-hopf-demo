"""Kennzahlen und Verdikte der sechs Experimente. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text).

Zahlen im Text: deutsche Schreibweise (Komma, Minuszeichen „−“), siehe `de`."""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import vzh_algorithm as A
import vzh_constants as C
import vzh_lorenz as L

VORGERECHNET = Path(__file__).resolve().parent / "vzh_vorgerechnet.json"


def de(x: float, nd: int = 2, plus: bool = False) -> str:
    """Zahl mit Dezimalkomma und echtem Minuszeichen; `plus`: Vorzeichen immer zeigen."""
    s = f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"
    if float(s.replace("+", "").replace("-", "")) == 0.0:       # „−0,00“ nie ausgeben
        s = s.lstrip("+-")
        s = ("+" + s) if plus else s
    return s.replace(".", ",").replace("-", "−")


@dataclass(frozen=True)
class Settings:
    at: float = C.DEFAULT_AT
    stuecke: int = C.DEFAULT_STUECKE
    m: int = C.DEFAULT_M
    verfahren: str = C.DEFAULT_VERFAHREN
    mg_tau: float = C.DEFAULT_MG_TAU
    rho: int = C.DEFAULT_RHO
    seed: int = C.DEFAULT_SEED


# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: Methode der Schritte
# ---------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hand:
    a: float
    stuecke: int
    werte: tuple          # x(0), x(1), ..., x(stuecke), exakt (Methode der Schritte)
    heun_werte: tuple     # dieselben Zeitpunkte, Heun mit M_HAND Schritten je Verzögerung
    abweichung: float     # größter Betrag der Differenz an diesen Zeitpunkten
    sigma: float          # Re s der rechtesten Kennlinienwurzel
    omega: float          # Im s
    t: tuple              # Stützstellen der Stücke für das Bild
    x: tuple


def analyse_hand(a: float, stuecke: int) -> Hand:
    pieces, values = A.methode_der_schritte(a, stuecke)
    heun = A.linear_dde(a, C.M_HAND, stuecke)
    heun_werte = tuple(float(heun[k * C.M_HAND]) for k in range(stuecke + 1))
    abw = max(abs(u - v) for u, v in zip(values, heun_werte))
    s = A.rechtester_wurzel(a)
    ts, xs = [], []
    for k, p in enumerate(pieces):
        tk = np.linspace(k, k + 1, 61)
        ts.extend(tk.tolist())
        xs.extend(p(tk).tolist())
    return Hand(a, stuecke, tuple(values), heun_werte, float(abw), float(s.real), float(abs(s.imag)), tuple(ts), tuple(xs))


def verdict_hand(h: Hand) -> tuple[str, str]:
    """Verdikt Experiment 1: die ersten Stützwerte, die Wachstumsrate der Hüllkurve und ob der Verlauf monoton ist (a <= 1/e)."""
    n = min(3, h.stuecke)
    stuetz = ", ".join(f"x({k}) = {de(h.werte[k], 3)}" for k in range(1, n + 1))
    if abs(h.a - A.hopf_grenze_linear()) < 0.005:
        rate = f"die Hüllkurve ist an der Grenze π/2 = 1,571 konstant (Rate {de(h.sigma, 3)}), die Schwingung hat die Periode 4 τ"
        kind = "info"
    elif h.sigma < 0:
        rate = f"die Hüllkurve fällt mit Rate {de(-h.sigma, 3)} je τ"
        kind = "success"
    else:
        rate = f"die Hüllkurve wächst mit Rate {de(h.sigma, 3)} je τ"
        kind = "warning"
    if h.a <= 1.0 / math.e:
        form = f"Der Verlauf ist monoton, denn a·τ = {de(h.a, 2)} liegt unter 1/e = {de(1.0 / math.e, 3)}."
    else:
        form = (f"Der Verlauf wechselt das Vorzeichen, denn a·τ = {de(h.a, 2)} liegt über 1/e = {de(1.0 / math.e, 3)}; "
                f"die Periode der Schwingung ist {de(2 * math.pi / h.omega, 2)} τ.")
    return kind, f"{stuetz}; {rate} (a·τ = {de(h.a, 2)}). {form}"


# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Kennlinie
# ---------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Kennlinie:
    a: float
    wurzeln: tuple        # komplexe Wurzeln s_k, k = -2 .. 2
    rechts: complex
    gemessen: float       # aus der Simulation gemessene Wachstumsrate (Heun, M_STANDARD)
    kurve_a: tuple
    kurve_re: tuple


def analyse_kennlinie(a: float) -> Kennlinie:
    w = A.kennlinie_wurzeln(a, ks=range(-2, 3))
    grid = np.linspace(C.AT_MIN, C.AT_MAX, 120)
    kurve = [A.rechtester_wurzel(float(g)).real for g in grid]
    return Kennlinie(a, tuple(complex(v) for v in w), A.rechtester_wurzel(a), A.gemessene_wachstumsrate(a, C.M_STANDARD, "heun"),
                     tuple(grid.tolist()), tuple(kurve))


def verdict_kennlinie(k: Kennlinie) -> tuple[str, str]:
    """Verdikt Experiment 2: größtes Re s und die Lage von a·τ zu π/2; Periode aus Im s."""
    grenze = A.hopf_grenze_linear()
    kopf = f"a·τ = {de(k.a, 2)}: größtes Re s = {de(k.rechts.real, 4)} (Simulation: {de(k.gemessen, 4)})"
    if abs(k.a - grenze) < 0.005:
        return "info", f"{kopf}; an der Grenze π/2 = {de(grenze, 3)}: Re s ≈ 0, die Periode ist 4 τ."
    periode = f", Periode {de(2 * math.pi / abs(k.rechts.imag), 2)} τ" if abs(k.rechts.imag) > 1e-12 else ", keine Schwingung (reelle Wurzel)"
    if k.a < grenze:
        return "success", f"{kopf}; unter π/2 = {de(grenze, 3)} abklingend{periode}."
    return "warning", f"{kopf}; über π/2 = {de(grenze, 3)} anwachsend{periode}."


# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: Hutchinson hinter der Hopf-Grenze
# ---------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hut:
    r: float
    span: float
    period: float
    mean: float
    t: tuple             # Zeitverlauf der letzten 40 Verzögerungen (ausgedünnt)
    x: tuple
    x_jetzt: tuple       # x(t) und x(t - 1) über die letzten 60 Verzögerungen (ausgedünnt), für das Phasenbild
    x_zuvor: tuple
    span_zuvor: float = float("nan")   # Spannweite des Fensters davor (Verzögerung 1400 bis 1700 von 2000); nan: unbekannt

    @property
    def eingeschwungen(self) -> bool:
        """Spannweite ändert sich von einem Fenster zum vorigen um höchstens 2 % (oder das Fenster davor ist unbekannt)."""
        if self.span_zuvor != self.span_zuvor or self.span <= 0.0:       # nan oder keine Schwingung
            return True
        return abs(self.span - self.span_zuvor) <= C.HUT_DRIFT * self.span


@dataclass(frozen=True)
class HutFit:
    rs: tuple
    spans: tuple
    c1: float
    c0: float
    r0: float            # Nullstelle der Geraden Spannweite^2 = c1 r + c0


def analyse_hutchinson(r: float) -> Hut:
    m = C.M_STANDARD
    x = A.hutchinson(r, m, C.HUT_N_TAU)
    kz = A.hutchinson_kennzahlen(x, m)
    davor = A.hutchinson_kennzahlen(x[: -C.HUT_TAIL_TAU * m], m)
    tail = x[-40 * m:]
    step = 8
    t = (np.arange(len(tail))[::step] / m).tolist()
    ph = x[-60 * m:]
    return Hut(r, kz["span"], kz["period"], kz["mean"], tuple(t), tuple(tail[::step].tolist()),
               tuple(ph[m:][::step].tolist()), tuple(ph[:-m][::step].tolist()), davor["span"])


def analyse_hutchinson_fit() -> HutFit:
    spans = []
    for r in C.HUT_FIT_RS:
        x = A.hutchinson(r, C.M_STANDARD, C.HUT_FIT_N_TAU)
        spans.append(A.hutchinson_kennzahlen(x, C.M_STANDARD)["span"])
    c1, c0, r0 = A.fit_spannweite_quadrat(C.HUT_FIT_RS, spans)
    return HutFit(tuple(C.HUT_FIT_RS), tuple(spans), c1, c0, r0)


def verdict_hutchinson(h: Hut, fit: HutFit) -> tuple[str, str]:
    """Verdikt Experiment 3: Spannweite und Periode der letzten 300 Verzögerungen; die gemessene Nullstelle der Geraden gegen π/2."""
    grenze = A.hopf_grenze_linear()
    nullst = (f"Die Gerade durch die Spannweiten bei r·τ = {de(fit.rs[0], 2)} bis {de(fit.rs[-1], 2)} (Spannweite² gegen r·τ) schneidet die Nullachse bei "
              f"{de(fit.r0, 3)}, gegen π/2 = {de(grenze, 3)}.")
    if h.span < C.HUT_SPAN_NULL:
        return "info", (f"r·τ = {de(h.r, 2)}: keine Dauerschwingung, die Spannweite der letzten {C.HUT_TAIL_TAU} Verzögerungen ist {de(h.span, 3)} "
                        f"(die Bahn läuft in den Fixpunkt 1). {nullst}")
    if not h.eingeschwungen:
        return "info", (f"r·τ = {de(h.r, 2)}: noch nicht entschieden, langsames Abklingen oder Anwachsen. Die Spannweite ändert sich von einem Fenster von {C.HUT_TAIL_TAU} Verzögerungen "
                        f"zum vorigen ({de(h.span_zuvor, 3)} auf {de(h.span, 3)}), die Bahn ist nach {C.HUT_N_TAU} Verzögerungen noch nicht eingeschwungen; nahe π/2 = {de(grenze, 3)} "
                        f"ist die Spannweite der letzten {C.HUT_TAIL_TAU} Verzögerungen kein Beleg für eine Dauerschwingung. {nullst}")
    return "warning", (f"r·τ = {de(h.r, 2)}: Dauerschwingung mit Spannweite {de(h.span, 3)} und Periode {de(h.period, 3)} τ "
                       f"(Mittel der letzten {C.HUT_TAIL_TAU} Verzögerungen). {nullst}")


# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: Zeitschritt
# ---------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Zeitschritt:
    a: float
    m: int
    verfahren: str
    exakt: float          # Re s der Gleichung
    diskret: float        # Rate der Rekursion aus den Wurzeln ihres Polynoms
    gemessen: float       # Rate aus der Simulation
    grenze_euler: float   # 2 m sin(pi/(4 m + 2))
    kurve_m: tuple
    kurve_euler: tuple
    kurve_heun: tuple


def analyse_zeitschritt(a: float, m: int, verfahren: str) -> Zeitschritt:
    meth = "euler" if verfahren == "Euler" else "heun"
    return Zeitschritt(
        a, m, verfahren, A.rechtester_wurzel(a).real, A.diskrete_wachstumsrate(a, m, meth), A.gemessene_wachstumsrate(a, m, meth), A.euler_grenze(m),
        tuple(C.M_OPTIONEN), tuple(A.diskrete_wachstumsrate(a, mm, "euler") for mm in C.M_OPTIONEN),
        tuple(A.diskrete_wachstumsrate(a, mm, "heun") for mm in C.M_OPTIONEN),
    )


def verdict_zeitschritt(z: Zeitschritt) -> tuple[str, str]:
    """Verdikt Experiment 4: stimmt das Vorzeichen der Wachstumsrate der Rechnung mit dem der Gleichung überein?"""
    zeile = (f"Rate je τ: Gleichung {de(z.exakt, 4, plus=True)}, {z.verfahren} mit m = {z.m} Schritten je τ {de(z.diskret, 4, plus=True)} "
             f"(Simulation {de(z.gemessen, 4, plus=True)})")
    euler = f" Euler-Grenze für m = {z.m}: 2m·sin(π/(4m+2)) = {de(z.grenze_euler, 3)}."
    if z.exakt < 0 < z.diskret:
        return "warning", (f"Bei m = {z.m} Schritten je τ rechnet {z.verfahren} die Gleichung bei a·τ = {de(z.a, 2)} als instabil, obwohl sie stabil ist. "
                           f"{zeile}.{euler}")
    if z.diskret < 0 < z.exakt:
        return "warning", (f"Bei m = {z.m} Schritten je τ rechnet {z.verfahren} die Gleichung bei a·τ = {de(z.a, 2)} als stabil, obwohl sie instabil ist. "
                           f"{zeile}.{euler}")
    return "success", (f"Bei m = {z.m} Schritten je τ stimmt das Vorzeichen der Rate bei a·τ = {de(z.a, 2)} mit der Gleichung überein "
                       f"(Abweichung {de(abs(z.diskret - z.exakt), 4)}). {zeile}.{euler}")


# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 5: Mackey-Glass
# ---------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MackeyGlass:
    tau: float
    seed: int
    lam_zweibahn: float
    lam_tangente: float
    t_mess: float
    x_min: float
    x_max: float
    rechenzeit: float     # Sekunden der ersten Berechnung (zwei Verfahren zusammen)
    hopf_tau: float
    wurzel_re: float      # größtes Re s der Linearisierung am Gleichgewicht bei diesem tau
    t: tuple
    x: tuple
    x_zuvor: tuple        # x(t - tau)
    x_jetzt: tuple        # x(t)


def analyse_mackey_glass(tau: float, seed: int) -> MackeyGlass:
    t0 = time.perf_counter()
    zb = A.mg_lyapunov(tau, C.MG_T_MESS, C.MG_T_BURN, seed, "zweibahn")
    tg = A.mg_lyapunov(tau, C.MG_T_MESS, C.MG_T_BURN, seed, "tangente")
    zeit = time.perf_counter() - t0
    t, x, m, h = A.mg_bahn(tau, seed, C.MG_T_BURN, C.MG_T_ZEIGE)
    step = max(1, int(round(0.4 / h)))
    sel = slice(m, None, step)
    ph_x = x[: len(x) - m][::step]
    ph_y = x[m:][::step]
    ts, xs = t[sel], x[sel]
    return MackeyGlass(
        float(tau), int(seed), zb["lambda"], tg["lambda"], zb["t_mess"], zb["min"], zb["max"], zeit, A.mg_hopf_tau(), A.mg_groesste_wurzel(tau).real,
        tuple(ts.tolist()), tuple(xs.tolist()), tuple(ph_x.tolist()), tuple(ph_y.tolist()),
    )


def verdict_mackey_glass(g: MackeyGlass) -> tuple[str, str]:
    """Verdikt Experiment 5: Exponent (Zweibahn) und Vorhersagezeit 1/λ; bei jedem anderen τ als 17 der Zusatz „nicht gegen eine Referenz geprüft“."""
    lam = g.lam_zweibahn
    hopf = f"Das Gleichgewicht verliert die Stabilität bei τ_c = {de(g.hopf_tau, 3)} (geschlossen)."
    zusatz = "" if abs(g.tau - C.MG_TAU_STANDARD) < 1e-9 else " Bei diesem τ nicht gegen eine Referenz geprüft."
    if lam > C.MG_LAMBDA_NULL:
        return "warning", (f"τ = {de(g.tau, 1)}: λ = {de(lam, 4, plus=True)} je Zeiteinheit (Zweibahn; Variationsgleichung {de(g.lam_tangente, 4, plus=True)}), "
                           f"Vorhersagezeit 1/λ rund {de(1.0 / lam, 0)} Zeiteinheiten.{zusatz} {hopf}")
    return "info", (f"τ = {de(g.tau, 1)}: kein positiver Exponent über der Schwelle {de(C.MG_LAMBDA_NULL, 4)} messbar (λ = {de(lam, 4, plus=True)} je Zeiteinheit): "
                    f"nach dieser Messung periodisch, ein Gleichgewicht oder ein Exponent unter der Schwelle.{zusatz} {hopf}")


# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 6: Lorenz
# ---------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Lorenz:
    rho: int
    seed: int
    lam: tuple            # lambda_1 >= lambda_2 >= lambda_3 (Mittel über die Bahnen)
    sem: tuple
    summe: float
    spur: float
    mittel_div: float     # mittlere Zeit bis Abstand 1 (nan: keine Bahn erreicht ihn)
    min_div: float
    max_div: float
    nicht_erreicht: int
    n_div: int
    rechenzeit: float
    pfad_x: tuple
    pfad_z: tuple
    abstand_t: tuple
    abstand: tuple


def analyse_lorenz(rho: int, seed: int) -> Lorenz:
    t0 = time.perf_counter()
    sp = L.spektrum(float(rho), seed)
    dv = L.divergenz(float(rho), seed)
    zeit = time.perf_counter() - t0
    z = dv["zeiten"]
    ok = z[~np.isnan(z)]
    pf = dv["pfad"]
    return Lorenz(
        int(rho), int(seed), tuple(float(v) for v in sp["lam"]), tuple(float(v) for v in sp["sem"]), float(sp["summe"]), L.spur(),
        float(ok.mean()) if len(ok) else float("nan"), float(ok.min()) if len(ok) else float("nan"), float(ok.max()) if len(ok) else float("nan"),
        int(np.isnan(z).sum()), int(len(z)), zeit,
        tuple(pf[:, 0].tolist()), tuple(pf[:, 2].tolist()),
        tuple((np.arange(len(dv["abstand"])) * dv["dt_pfad"]).tolist()), tuple(dv["abstand"].tolist()),
    )


def verdict_lorenz(lo: Lorenz) -> tuple[str, str]:
    """Verdikt Experiment 6: λ₁ und die Summe der Exponenten gegen die Spur; Vorhersagehorizont bei positivem λ₁."""
    l1 = lo.lam[0]
    summe = f"Summe der drei Exponenten {de(lo.summe, 3)} gegen die Spur {de(lo.spur, 3)}"
    rho_h = L.hopf_rho()
    if l1 > C.LOR_LAMBDA_CHAOS:
        if lo.nicht_erreicht == 0:
            hor = f"Zwei Bahnen im Anfangsabstand 1e-9 erreichen den Abstand 1 nach im Mittel {de(lo.mittel_div, 1)} Zeiteinheiten ({de(lo.min_div, 1)} bis {de(lo.max_div, 1)} bei {lo.n_div} Startwerten)"
        else:
            hor = f"Von {lo.n_div} Paaren im Anfangsabstand 1e-9 erreichen {lo.n_div - lo.nicht_erreicht} den Abstand 1 innerhalb von {de(C.LOR_T_DIV_MAX, 0)} Zeiteinheiten"
        band = ""
        if lo.rho == 28:
            band = f" (Band der Vormessung {de(C.LOR_RHO_BAND[0], 2)} bis {de(C.LOR_RHO_BAND[1], 2)})"
        return "warning", f"ρ = {lo.rho}: λ₁ = {de(l1, 2, plus=True)} ± {de(lo.sem[0], 2)}{band}; {hor}; {summe}."
    lage = (f"ρ liegt unter ρ_H = {de(rho_h, 3)}: die Fixpunkte sind linear stabil" if lo.rho < rho_h
            else f"ρ liegt über ρ_H = {de(rho_h, 3)}: die Fixpunkte sind linear instabil")
    return "info", (f"ρ = {lo.rho}: kein Chaos (λ₁ = {de(l1, 2, plus=True)} ± {de(lo.sem[0], 2)}, Schwelle {de(C.LOR_LAMBDA_CHAOS, 2)}); {lage}; "
                    f"{summe}.")


# ---------------------------------------------------------------------------------------------------------------------------------------
# Vorgerechnete Reihen
# ---------------------------------------------------------------------------------------------------------------------------------------


def lade_vorgerechnet(pfad: Path = VORGERECHNET) -> dict | None:
    """Die eingefrorenen Messreihen (tools/vorrechnen.py); None, wenn die Datei fehlt."""
    try:
        return json.loads(pfad.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
