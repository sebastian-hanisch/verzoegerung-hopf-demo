"""Tab Lorenz: das Lorenz-System als zweite Art, in stetiger Zeit chaotisch zu werden (drei Größen statt einer Verzögerung).

x' = sigma (y - x),  y' = x (rho - z) - y,  z' = x y - beta z   mit sigma = 10 und beta = 8/3 fest, rho regelbar.

Der Rechenaufwand liegt in der Zahl der Zeitschritte, nicht in der Größe des Systems. Damit die Live-Rechnung unter etwa zwei Sekunden bleibt, laufen
**viele kurze Bahnen gleichzeitig** (ein Feld mit einer Zeile je Startwert, RK4 über alle Zeilen auf einmal) statt einer langen Bahn nacheinander;
das Spektrum ist dann das Mittel über die Startwerte, mit der Streuung zwischen ihnen als Fehlerangabe.

Lyapunov-Spektrum nach dem QR-Verfahren (RK4 für den Zustand und drei Tangentialvektoren zugleich, alle `renorm` Schritte eine QR-Zerlegung)."""

import math

import numpy as np

import vzh_constants as C
from vzh_rng import SplitMix64

SIGMA, BETA = C.LOR_SIGMA, C.LOR_BETA


def spur() -> float:
    """Spur der Jacobi-Matrix, -(sigma + 1 + beta): Summe der drei Lyapunov-Exponenten (Volumenschrumpfung)."""
    return -(SIGMA + 1.0 + BETA)


def hopf_rho() -> float:
    """rho_H = sigma (sigma + beta + 3) / (sigma - beta - 1): ab hier haben die Fixpunkte C+ und C- ein Paar auf der imaginären Achse (Routh-Hurwitz)."""
    return SIGMA * (SIGMA + BETA + 3.0) / (SIGMA - BETA - 1.0)


def hopf_frequenz() -> float:
    """Frequenz des Paares bei rho_H: sqrt(beta (sigma + rho_H))."""
    return math.sqrt(BETA * (SIGMA + hopf_rho()))


def jacobi(s, rho: float) -> np.ndarray:
    x, y, z = s
    return np.array([[-SIGMA, SIGMA, 0.0], [rho - z, -1.0, -x], [y, x, -BETA]])


def fixpunkt_c(rho: float) -> np.ndarray:
    """Fixpunkt C+ = (c, c, rho - 1) mit c = sqrt(beta (rho - 1)); existiert für rho > 1."""
    c = math.sqrt(BETA * (rho - 1.0))
    return np.array([c, c, rho - 1.0])


def rechte_seite(S, rho):
    """Lorenz-Feld für ein Feld von Zuständen der Form (..., 3)."""
    x, y, z = S[..., 0], S[..., 1], S[..., 2]
    return np.stack([SIGMA * (y - x), x * (rho - z) - y, x * y - BETA * z], axis=-1)


def rk4(S, rho, dt):
    k1 = rechte_seite(S, rho)
    k2 = rechte_seite(S + 0.5 * dt * k1, rho)
    k3 = rechte_seite(S + 0.5 * dt * k2, rho)
    k4 = rechte_seite(S + dt * k3, rho)
    return S + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _ableitung(S, Q, rho):
    """Zustandsfeld und Tangentialfeld J(S) Q für Felder S (K, 3), Q (K, 3, 3) ohne die Matrix J aufzubauen."""
    x, y, z = S[:, 0], S[:, 1], S[:, 2]
    dS = np.stack([SIGMA * (y - x), x * (rho - z) - y, x * y - BETA * z], axis=1)
    q0, q1, q2 = Q[:, 0, :], Q[:, 1, :], Q[:, 2, :]
    dQ = np.stack(
        [SIGMA * (q1 - q0), (rho - z)[:, None] * q0 - q1 - x[:, None] * q2, y[:, None] * q0 + x[:, None] * q1 - BETA * q2],
        axis=1,
    )
    return dS, dQ


def _rk4_mit_tangente(S, Q, rho, dt):
    a1, b1 = _ableitung(S, Q, rho)
    a2, b2 = _ableitung(S + 0.5 * dt * a1, Q + 0.5 * dt * b1, rho)
    a3, b3 = _ableitung(S + 0.5 * dt * a2, Q + 0.5 * dt * b2, rho)
    a4, b4 = _ableitung(S + dt * a3, Q + dt * b3, rho)
    return S + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4), Q + dt / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)


def startwerte(rho: float, n: int, seed: int) -> np.ndarray:
    """n zufällige Startwerte (SplitMix64): x, y gleichverteilt in [-15, 15], z in [0, max(30, rho)]."""
    rng = SplitMix64(int(seed))
    zmax = max(30.0, rho)
    pts = np.empty((n, 3))
    for i in range(n):
        pts[i] = (-15.0 + 30.0 * rng.random(), -15.0 + 30.0 * rng.random(), zmax * rng.random())
    return pts


def einschwingen(S, rho: float, t: float, dt: float) -> np.ndarray:
    for _ in range(int(round(t / dt))):
        S = rk4(S, rho, dt)
    return S


def spektrum(rho: float, seed: int, n_starts: int = C.LOR_N_START, t_burn: float = C.LOR_T_BURN, t_mess: float = C.LOR_T_MESS,
             dt: float = C.LOR_DT, renorm: int = C.LOR_RENORM, t_qr: float = C.LOR_T_QR) -> dict:
    """Lyapunov-Spektrum (lambda_1 >= lambda_2 >= lambda_3) als Mittel über `n_starts` gleichzeitig gerechnete Bahnen.

    `t_burn`: Einschwingzeit des Zustands; `t_qr`: Zeit, in der die Tangentialvektoren sich auf die Richtungen des Spektrums einstellen, ohne mitgezählt zu werden.
    Rückgabe: {"lam": Mittel (3,), "sem": Streuung des Mittels (3,), "summe": Summe der drei Mittelwerte, "je_start": Exponenten je Bahn (n, 3),
    "t_gesamt": Summe der Messzeiten aller Bahnen}."""
    S = einschwingen(startwerte(rho, n_starts, seed), rho, t_burn, dt)
    Q = np.tile(np.eye(3), (n_starts, 1, 1))
    acc = np.zeros((n_starts, 3))
    n_steps = int(round(t_mess / dt))
    n_qr = int(round(t_qr / dt))
    for k in range(n_qr + n_steps):
        S, Q = _rk4_mit_tangente(S, Q, rho, dt)
        if (k + 1) % renorm == 0:
            Q, R = np.linalg.qr(Q)
            if k >= n_qr:
                acc += np.log(np.abs(np.diagonal(R, axis1=1, axis2=2)))
    per = acc / (n_steps * dt)
    per = np.sort(per, axis=1)[:, ::-1]
    return {
        "lam": per.mean(axis=0),
        "sem": per.std(axis=0, ddof=1) / math.sqrt(n_starts) if n_starts > 1 else np.zeros(3),
        "summe": float(per.mean(axis=0).sum()),
        "je_start": per,
        "t_gesamt": n_starts * n_steps * dt,
        "n_starts": n_starts,
    }


def divergenz(rho: float, seed: int, n: int = C.LOR_N_DIV, d0: float = C.LOR_D0, t_max: float = C.LOR_T_DIV_MAX, dt: float = C.LOR_DT_DIV,
              t_burn: float = C.LOR_T_BURN_DIV) -> dict:
    """Zwei Bahnen im Anfangsabstand d0 (zufällige Richtung): Zeit bis zum Abstand 1, für n Startwerte auf dem Attraktor.

    Rückgabe: {"zeiten": Zeit bis Abstand 1 je Start (nan: nicht erreicht), "pfad": Bahn des ersten Startwerts (Zeilen x, y, z), "abstand": Abstand des ersten
    Paares über die Zeit}. Die Startwerte werden nach der Einschwingzeit `t_burn` aus zufälligen Zuständen genommen (n verschiedene Bahnen)."""
    A = einschwingen(startwerte(rho, n, seed + 7919), rho, t_burn, dt)
    rng = SplitMix64(int(seed) + 104729)
    d = np.array([[rng.normal(), rng.normal(), rng.normal()] for _ in range(n)])
    d = d0 * d / np.linalg.norm(d, axis=1)[:, None]
    B = A + d
    zeiten = np.full(n, np.nan)
    pfad = [A[0].copy()]
    abstand = [float(np.linalg.norm(A[0] - B[0]))]
    for k in range(1, int(round(t_max / dt)) + 1):
        A = rk4(A, rho, dt)
        B = rk4(B, rho, dt)
        dist = np.linalg.norm(A - B, axis=1)
        neu = np.isnan(zeiten) & (dist > 1.0)
        zeiten[neu] = k * dt
        if k % C.LOR_PFAD_JEDE == 0:
            pfad.append(A[0].copy())
            abstand.append(float(dist[0]))
        if not np.isnan(zeiten).any() and k * dt > 0:
            # alle Paare haben den Abstand 1 erreicht: den Pfad nur noch fertig aufzeichnen, solange er kürzer als 20 Zeiteinheiten ist
            if k * dt >= C.LOR_T_PFAD_MIN:
                break
    return {"zeiten": zeiten, "pfad": np.array(pfad), "abstand": np.array(abstand), "dt_pfad": dt * C.LOR_PFAD_JEDE}
