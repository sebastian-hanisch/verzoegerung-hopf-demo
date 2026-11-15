"""Kern: Differenzialgleichungen mit fester Verzögerung (numpy, scipy).

Drei Gleichungen, alle mit dem Heun-Verfahren (Euler als Vergleich bei der linearen Gleichung):

* lineare Gleichung      x'(t) = -a x(t - tau)               (Zeit in Einheiten der Verzögerung, tau = 1, also a steht für a*tau)
* Hutchinson (verzögerte Logistik)  x'(t) = r x(t) (1 - x(t - tau))   (ebenfalls tau = 1, r steht für r*tau)
* Mackey-Glass           x'(t) = beta x(t - tau) / (1 + x(t - tau)^n) - gamma x(t)   (Zeit in Zeiteinheiten, tau regelbar)

**Blockweise statt Schritt für Schritt.** Im Block der Länge tau (m Schritte) ist der verzögerte Wert aus dem vorigen Block bekannt. Dann ist jede
dieser Gleichungen in x(t) allein linear, und ein Heun-Schritt ist eine feste lineare Vorschrift: Summe (lineare Gleichung: Trapezregel),
Produkt (Hutchinson: x_{i+1} = g_i x_i) oder lineare Rekursion x_{i+1} = A x_i + b_i (Mackey-Glass, mit `scipy.signal.lfilter`). Das ergibt dieselben Zahlen wie
der Schritt-für-Schritt-Heun (`tests/test_algorithm.py` rechnet beides) und ist um den Faktor 50 bis 100 schneller als eine Python-Schleife.

Indexkonvention der Fenster: ein Fenster hat m + 1 Werte, die Zeiten t - tau, ..., t (Spaltenindex k <-> t - tau + k h)."""

import math

import numpy as np
from scipy.signal import lfilter
from scipy.special import lambertw

import vzh_constants as C
from vzh_rng import SplitMix64

# ---------------------------------------------------------------------------------------------------------------------------------------
# Lineare Gleichung x' = -a x(t - 1), Vorgeschichte konstant 1
# ---------------------------------------------------------------------------------------------------------------------------------------


def linear_dde(a: float, m: int, n_tau: int, method: str = "heun", hist: float = 1.0) -> np.ndarray:
    """Lösung von x'(t) = -a x(t - 1) auf t = 0 .. n_tau (Einheiten der Verzögerung), m Schritte je Verzögerung.

    Rückgabe: Array der Länge n_tau * m + 1 (t = i / m). `method`: "heun" oder "euler"."""
    if m < 1 or n_tau < 1:
        raise ValueError("m und n_tau müssen mindestens 1 sein")
    if method not in ("heun", "euler"):
        raise ValueError(f"unbekanntes Verfahren: {method}")
    h = 1.0 / m
    x = np.empty((n_tau + 1) * m + 1)
    x[: m + 1] = hist                        # Vorgeschichte auf [-1, 0] (Index m ist t = 0)
    for j in range(n_tau):
        i0 = m * (j + 1)
        u = x[i0 - m: i0 + 1]
        if method == "heun":
            inc = -a * h * 0.5 * (u[:-1] + u[1:])
        else:
            inc = -a * h * u[:-1]
        x[i0 + 1: i0 + m + 1] = x[i0] + np.cumsum(inc)
    return x[m:]


def linear_dde_stepwise(a: float, m: int, n_tau: int, method: str = "heun", hist: float = 1.0) -> np.ndarray:
    """Dasselbe Schritt für Schritt (Schleife): Bezugsrechnung für die Blockfassung, nur für Tests und kleine n_tau gedacht."""
    h = 1.0 / m
    x = np.full((n_tau + 1) * m + 1, hist, dtype=float)
    for i in range(m, (n_tau + 1) * m):
        k1 = -a * x[i - m]
        if method == "heun":
            k2 = -a * x[i - m + 1]
            x[i + 1] = x[i] + 0.5 * h * (k1 + k2)
        else:
            x[i + 1] = x[i] + h * k1
    return x[m:]


def kennlinie_wurzeln(a: float, ks=range(-3, 4)) -> np.ndarray:
    """Wurzeln s_k der Kennlinie s + a e^{-s} = 0 (tau = 1): s_k = W_k(-a) (Lambertsche W-Funktion, Zweige k)."""
    return np.array([complex(lambertw(-a, k)) for k in ks])


def rechtester_wurzel(a: float) -> complex:
    """Die Wurzel mit dem größten Realteil (Zweig k = 0 genügt; über k = -3 .. 3 abgesichert)."""
    w = kennlinie_wurzeln(a)
    return complex(max(w, key=lambda z: (round(z.real, 12), z.imag)))      # bei gleichem Realteil (Paar): die mit positivem Im s


def hopf_grenze_linear() -> float:
    """a*tau, bei dem die Kennlinie ein Paar auf der imaginären Achse hat: s = i pi/2 gibt (i pi/2) e^{i pi/2} = -pi/2, also a*tau = pi/2."""
    return math.pi / 2


def euler_grenze(m: int) -> float:
    """Grenze von a*tau, ab der explizites Euler mit m Schritten je Verzögerung die lineare Gleichung instabil rechnet: 2 m sin(pi / (4 m + 2))."""
    return 2.0 * m * math.sin(math.pi / (4.0 * m + 2.0))


def diskrete_wachstumsrate(a: float, m: int, method: str = "heun") -> float:
    """Wachstumsrate je Verzögerung der diskreten Rekursion aus den Wurzeln ihres charakteristischen Polynoms (anderer Rechenweg als die Simulation).

    Heun:  x_{i+1} = x_i - (a h / 2)(x_{i-m} + x_{i-m+1})  ->  z^{m+1} - z^m + (a h / 2)(z + 1) = 0
    Euler: x_{i+1} = x_i - a h x_{i-m}                      ->  z^{m+1} - z^m + a h = 0
    Rate je Verzögerung = m ln(größter Betrag der Wurzeln)."""
    h = 1.0 / m
    poly = np.zeros(m + 2)
    poly[0] = 1.0
    poly[1] -= 1.0
    if method == "heun":
        c = 0.5 * a * h
        poly[m] += c
        poly[m + 1] += c
    elif method == "euler":
        poly[m + 1] += a * h
    else:
        raise ValueError(f"unbekanntes Verfahren: {method}")
    z = np.roots(poly)
    return float(m * math.log(float(np.max(np.abs(z)))))


def _extrema_envelope(x: np.ndarray, h: float, t_lo: float):
    """Zeiten und Beträge der lokalen Extrema (parabolisch zwischen drei Stützstellen verfeinert) für t > t_lo."""
    y0, y1, y2 = x[:-2], x[1:-1], x[2:]
    is_ext = ((y1 > y0) & (y1 >= y2)) | ((y1 < y0) & (y1 <= y2))
    idx = np.where(is_ext)[0] + 1
    if idx.size == 0:
        return np.array([]), np.array([])
    a0, a1, a2 = x[idx - 1], x[idx], x[idx + 1]
    den = a0 - 2.0 * a1 + a2
    ok = np.abs(den) > 0
    shift = np.zeros(idx.size)
    value = a1.copy()
    shift[ok] = 0.5 * (a0[ok] - a2[ok]) / den[ok]
    value[ok] = a1[ok] - (a2[ok] - a0[ok]) ** 2 / (8.0 * den[ok])
    t = (idx + shift) * h
    sel = (t > t_lo) & (np.abs(value) > 1e-250)
    return t[sel], np.abs(value[sel])


def gemessene_wachstumsrate(a: float, m: int = C.M_STANDARD, method: str = "heun") -> float:
    """Aus der Simulation gemessene Wachstumsrate je Verzögerung: Steigung von log|x| über die Extrema der Lösung (Hüllkurve).

    Ohne Extrema (a < 1/e, monotones Abklingen) über die Beträge bei ganzzahligen Vielfachen von tau. Laufzeit der Messung 300 Verzögerungen;
    wird dabei die Zahl der verwertbaren Punkte zu klein (Abklingen unter 1e-250), nimmt die Messung kürzere Läufe."""
    h = 1.0 / m
    for n_tau in (300, 150, 80, 40):
        with np.errstate(over="ignore", invalid="ignore"):
            x = linear_dde(a, m, n_tau, method)
        if not np.all(np.isfinite(x)):
            continue
        t, v = _extrema_envelope(x, h, 0.4 * n_tau)
        if len(t) >= 6:
            return float(np.polyfit(t, np.log(v), 1)[0])
        # monoton: Beträge bei t = 0, 1, 2, ... (nur gleiches Vorzeichen, über 1e-250)
        ti = np.arange(n_tau + 1, dtype=float)
        xi = x[:: m][: n_tau + 1]
        sel = (ti > 0.4 * n_tau) & (np.abs(xi) > 1e-250)
        if sel.sum() >= 6 and np.all(np.sign(xi[sel]) == np.sign(xi[sel][0])):
            return float(np.polyfit(ti[sel], np.log(np.abs(xi[sel])), 1)[0])
    return float("nan")


# Methode der Schritte (exakte Stücke als Polynome)


def methode_der_schritte(a: float, n_stuecke: int):
    """Exakte Lösung von x' = -a x(t - 1), Vorgeschichte 1, auf [0, n_stuecke] als Polynome je Stück (Methode der Schritte).

    Auf [k, k+1] ist x(t) = x(k) - a * Integral von k bis t über x(s - 1) ds, wobei x(s - 1) das Polynom des vorigen Stücks ist.
    Rückgabe: (Liste der Polynome, Liste der Stützwerte x(0), x(1), ..., x(n_stuecke))."""
    from numpy.polynomial import Polynomial as Poly

    if n_stuecke < 1:
        raise ValueError("mindestens ein Stück")
    t = Poly([0.0, 1.0])
    prev = Poly([1.0])                         # Vorgeschichte auf [-1, 0]
    values = [1.0]
    pieces = []
    for k in range(n_stuecke):
        shifted = prev(t - 1.0)                 # x(s - 1) als Polynom in s
        integ = shifted.integ()
        cur = values[-1] - a * (integ - integ(float(k)))
        pieces.append(cur)
        values.append(float(cur(float(k + 1))))
        prev = cur
    return pieces, values


# ---------------------------------------------------------------------------------------------------------------------------------------
# Hutchinson x' = r x (1 - x(t - 1))
# ---------------------------------------------------------------------------------------------------------------------------------------


def hutchinson(r: float, m: int = C.M_STANDARD, n_tau: int = 2000, x0: float = C.HUT_START) -> np.ndarray:
    """Lösung der verzögerten Logistik (Heun, m Schritte je Verzögerung), t = 0 .. n_tau; Vorgeschichte konstant x0.

    Im Block ist x_{i+1} = g_i x_i mit g_i = 1 + h/2 (a_i + a_{i+1} (1 + h a_i)), a_i = r (1 - x_{i-m}): das ist ein Heun-Schritt, nur nach x ausgeklammert."""
    h = 1.0 / m
    x = np.empty((n_tau + 1) * m + 1)
    x[: m + 1] = x0
    for j in range(n_tau):
        i0 = m * (j + 1)
        u = x[i0 - m: i0 + 1]
        a = r * (1.0 - u)
        g = 1.0 + 0.5 * h * (a[:-1] + a[1:] * (1.0 + h * a[:-1]))
        x[i0 + 1: i0 + m + 1] = x[i0] * np.cumprod(g)
    return x[m:]


def hutchinson_stepwise(r: float, m: int, n_tau: int, x0: float = C.HUT_START) -> np.ndarray:
    """Schritt-für-Schritt-Heun der verzögerten Logistik (Bezugsrechnung, nur für Tests)."""
    h = 1.0 / m
    x = np.full((n_tau + 1) * m + 1, x0, dtype=float)
    for i in range(m, (n_tau + 1) * m):
        k1 = r * x[i] * (1.0 - x[i - m])
        xp = x[i] + h * k1
        k2 = r * xp * (1.0 - x[i - m + 1])
        x[i + 1] = x[i] + 0.5 * h * (k1 + k2)
    return x[m:]


def hutchinson_kennzahlen(x: np.ndarray, m: int, tail_tau: int = C.HUT_TAIL_TAU) -> dict:
    """Spannweite (Maximum minus Minimum) und Periode (in Verzögerungen) der Dauerschwingung aus den letzten `tail_tau` Verzögerungen.

    Periode: mittlerer Abstand der Aufwärtsdurchgänge durch den Mittelwert; nan, wenn es weniger als drei Durchgänge gibt."""
    tail = x[-tail_tau * m:]
    span = float(tail.max() - tail.min())
    mean = float(tail.mean())
    up = np.where((tail[:-1] < mean) & (tail[1:] >= mean))[0]
    period = float(np.mean(np.diff(up))) / m if len(up) > 2 and span > 1e-6 else float("nan")
    return {"span": span, "period": period, "mean": mean}


def fit_spannweite_quadrat(rs, spans):
    """Gerade Spannweite^2 = c1 r + c0 durch die Punkte (kleinste Quadrate) und ihre Nullstelle r0 = -c0/c1."""
    c = np.polyfit(np.asarray(rs, dtype=float), np.asarray(spans, dtype=float) ** 2, 1)
    return float(c[0]), float(c[1]), float(-c[1] / c[0])


# ---------------------------------------------------------------------------------------------------------------------------------------
# Mackey-Glass
# ---------------------------------------------------------------------------------------------------------------------------------------


def mg_gleichgewicht(beta=C.MG_BETA, gamma=C.MG_GAMMA, n=C.MG_N) -> float:
    """x* mit beta x / (1 + x^n) = gamma x, also x* = (beta/gamma - 1)^(1/n)."""
    return (beta / gamma - 1.0) ** (1.0 / n)


def mg_linearisierung(beta=C.MG_BETA, gamma=C.MG_GAMMA, n=C.MG_N) -> tuple[float, float]:
    """Linearisierung u' = -gamma u - b u(t - tau) am Gleichgewicht: (gamma, b) mit b = -beta (1 + x*^n - n x*^n) / (1 + x*^n)^2."""
    xs = mg_gleichgewicht(beta, gamma, n)
    d = beta * (1.0 + xs**n - n * xs**n) / (1.0 + xs**n) ** 2
    return gamma, -d


def mg_hopf_tau(beta=C.MG_BETA, gamma=C.MG_GAMMA, n=C.MG_N) -> float:
    """Verzögerung der Hopf-Grenze des Gleichgewichts: tau_c = arccos(-gamma/b) / sqrt(b^2 - gamma^2) (geschlossen; nur für b > gamma)."""
    g, b = mg_linearisierung(beta, gamma, n)
    if b <= g:
        return float("inf")
    return math.acos(-g / b) / math.sqrt(b * b - g * g)


def mg_groesste_wurzel(tau: float, beta=C.MG_BETA, gamma=C.MG_GAMMA, n=C.MG_N) -> complex:
    """Wurzel der Kennlinie s + gamma + b e^{-s tau} = 0 mit größtem Realteil: s = W_k(-b tau e^{gamma tau}) / tau - gamma (k = -3 .. 3)."""
    g, b = mg_linearisierung(beta, gamma, n)
    arg = -b * tau * math.exp(g * tau)
    best = None
    for k in range(-3, 4):
        s = complex(lambertw(arg, k)) / tau - g
        if best is None or s.real > best.real:
            best = s
    return best


def mg_start_fenster(m: int, seed: int) -> np.ndarray:
    """Startverlauf: 1,2 plus Normalrauschen der Standardabweichung 0,05 auf m + 1 Stützstellen (SplitMix64)."""
    rng = SplitMix64(int(seed))
    return np.array([1.2 + 0.05 * rng.normal() for _ in range(m + 1)])


class _MG:
    """Rechenvorschrift eines Blocks mit festem tau, h = tau/m."""

    def __init__(self, tau: float, h_ziel: float = C.MG_H, beta=C.MG_BETA, gamma=C.MG_GAMMA, n=C.MG_N):
        self.m = max(2, int(round(tau / h_ziel)))
        self.h = tau / self.m
        self.tau, self.beta, self.gamma, self.n = tau, beta, gamma, n
        h = self.h
        self.A = 1.0 - h * gamma + 0.5 * h * h * gamma * gamma

    def g(self, x):
        return self.beta * x / (1.0 + x**self.n)

    def dg(self, x):
        xn = x**self.n
        return self.beta * (1.0 + xn - self.n * xn) / (1.0 + xn) ** 2

    def advance(self, w: np.ndarray) -> np.ndarray:
        """Fenster der Zeiten t - tau .. t -> Fenster t .. t + tau (Heun, blockweise linear in x)."""
        h, gamma = self.h, self.gamma
        c = self.g(w)
        b = 0.5 * h * (c[:-1] + c[1:]) - 0.5 * h * h * gamma * c[:-1]
        y = lfilter([1.0], [1.0, -self.A], b, zi=np.array([self.A * w[-1]]))[0]
        out = np.empty(self.m + 1)
        out[0] = w[-1]
        out[1:] = y
        return out

    def tangent(self, w: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Tangentialfenster nach einem Block: die Ableitung des Heun-Blocks nach dem Fenster (Variationsgleichung, g' von Hand)."""
        h, gamma = self.h, self.gamma
        gp = self.dg(w)
        b = 0.5 * h * (1.0 - h * gamma) * gp[:-1] * d[:-1] + 0.5 * h * gp[1:] * d[1:]
        y = lfilter([1.0], [1.0, -self.A], b, zi=np.array([self.A * d[-1]]))[0]
        out = np.empty(self.m + 1)
        out[0] = d[-1]
        out[1:] = y
        return out


def mg_bahn(tau: float, seed: int, t_burn: float = C.MG_T_BURN, t_zeige: float = C.MG_T_ZEIGE, h_ziel: float = C.MG_H):
    """Bahn der Mackey-Glass-Gleichung nach der Einschwingzeit: (t, x) über `t_zeige` Zeiteinheiten und das Fenster x(t - tau) dazu (für das Phasenbild)."""
    mg = _MG(tau, h_ziel)
    w = mg_start_fenster(mg.m, seed)
    for _ in range(int(math.ceil(t_burn / tau))):
        w = mg.advance(w)
    n_blocks = int(math.ceil(t_zeige / tau)) + 1
    wins = [w]
    for _ in range(n_blocks):
        w = mg.advance(w)
        wins.append(w)
    x = np.concatenate([wins[0]] + [v[1:] for v in wins[1:]])
    t = (np.arange(len(x)) - mg.m) * mg.h
    return t, x, mg.m, mg.h


def mg_lyapunov(tau: float, t_mess: float, t_burn: float, seed: int, verfahren: str = "zweibahn", h_ziel: float = C.MG_H, d0: float = 1e-8) -> dict:
    """Größter Lyapunov-Exponent je Zeiteinheit.

    "zweibahn": Zwei Bahnen im Abstand d0 (Euklid-Norm des Fensters aus m + 1 Werten), nach jedem Block (Länge tau) Abstand zurück auf d0, Mittel von log(Abstand/d0)/tau.
    "tangente": Variationsgleichung (Ableitung des Verzögerungsterms von Hand) längs derselben Bahn, Tangentialfenster nach jedem Block auf Länge 1 gesetzt.
    Beide messen dieselbe Größe auf verschiedenen Wegen; Messdauer = Zahl der Blöcke x tau."""
    mg = _MG(tau, h_ziel)
    w = mg_start_fenster(mg.m, seed)
    for _ in range(int(math.ceil(t_burn / tau))):
        w = mg.advance(w)
    n_blocks = max(1, int(round(t_mess / tau)))
    acc = 0.0
    lo, hi = float(w.min()), float(w.max())
    if verfahren == "zweibahn":
        v = w + d0 / math.sqrt(mg.m + 1)
        for _ in range(n_blocks):
            w = mg.advance(w)
            v = mg.advance(v)
            diff = v - w
            dist = float(np.linalg.norm(diff))
            acc += math.log(dist / d0)
            v = w + diff * (d0 / dist)
            lo, hi = min(lo, float(w.min())), max(hi, float(w.max()))
    elif verfahren == "tangente":
        d = np.full(mg.m + 1, 1.0 / math.sqrt(mg.m + 1))
        for _ in range(n_blocks):
            w_neu = mg.advance(w)
            d = mg.tangent(w, d)
            nd = float(np.linalg.norm(d))
            acc += math.log(nd)
            d = d / nd
            w = w_neu
            lo, hi = min(lo, float(w.min())), max(hi, float(w.max()))
    else:
        raise ValueError(f"unbekanntes Verfahren: {verfahren}")
    t_gesamt = n_blocks * tau
    lam = acc / t_gesamt
    return {"lambda": lam, "t_mess": t_gesamt, "m": mg.m, "h": mg.h, "min": lo, "max": hi}
