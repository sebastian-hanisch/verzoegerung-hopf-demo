"""Orakel-Tests (ORACLE-PLAYBOOK §2, §5, §6): jede Kernroutine gegen einen anderen Rechenweg.

* Wachstumsrate der Simulation  gegen  Lambertsche W-Funktion (scipy.special.lambertw), und gegen die Wurzeln des charakteristischen Polynoms der Rekursion (numpy.roots)
* Euler-Grenze 2m·sin(π/(4m+2))  gegen die Wurzeln dieses Polynoms
* Methode der Schritte (numpy-Polynome in Gleitkomma)  gegen eine eigene Bruchrechnung (fractions)
* Heun: Ordnung 2 (Fehlerverhältnis bei halbem Schritt)
* Hutchinson (nichtlinear)  gegen die Linearisierung (Kleinstabweichung wächst mit Re W0(-r·τ))
* Mackey-Glass: Blockfassung gegen die Schleife; Zweibahn-Exponent gegen die Variationsgleichung; Variationsgleichung gegen zentrale Differenz
* Lorenz: Summe der Exponenten gegen die Spur; Jacobi-Eigenwerte gegen die Wurzeln des charakteristischen Polynoms; RK4 gegen scipy.integrate.solve_ivp (DOP853)
Alle Läufe unter 10 Sekunden; große Läufe (200 Zufallsinstanzen) liefen im Scratch-Ordner, nicht im Repo."""

import math
from fractions import Fraction

import numpy as np
import pytest

import vzh_algorithm as A
import vzh_constants as C
import vzh_lorenz as L

lambertw = pytest.importorskip("scipy.special").lambertw
solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp


# --- lineare Gleichung ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("a", [0.5, 1.0, 1.4, 1.55, 1.6, 1.7, 2.0])
def test_gemessene_wachstumsrate_gleich_lambertw(a):
    """Messung (Heun, 400 Schritte je τ) gegen Re W0(-a). Der Plan nannte bei a = 1,7 einen Ausreißer (0,0007): er kam von den Spitzenwerten auf dem Gitter."""
    soll = complex(lambertw(-a, 0)).real
    assert A.gemessene_wachstumsrate(a, 400, "heun") == pytest.approx(soll, abs=1e-4)


@pytest.mark.parametrize("method", ["euler", "heun"])
@pytest.mark.parametrize("m", [1, 2, 3, 5, 7, 10, 20])
@pytest.mark.parametrize("a", [0.5, 1.45, 1.7])
def test_simulation_gleich_wurzeln_des_charakteristischen_polynoms(a, m, method):
    assert A.gemessene_wachstumsrate(a, m, method) == pytest.approx(A.diskrete_wachstumsrate(a, m, method), abs=2e-3)


@pytest.mark.parametrize("m", [1, 2, 3, 5, 10, 20, 50])
def test_euler_grenze_ist_die_stabilitaetsgrenze_der_rekursion(m):
    grenze = A.euler_grenze(m)
    assert A.diskrete_wachstumsrate(grenze * 0.999, m, "euler") < 0 < A.diskrete_wachstumsrate(grenze * 1.001, m, "euler")


def test_heun_rate_naehert_sich_der_gleichung_mit_ordnung_zwei():
    a = 1.45
    exakt = complex(lambertw(-a, 0)).real
    f1 = abs(A.diskrete_wachstumsrate(a, 50, "heun") - exakt)
    f2 = abs(A.diskrete_wachstumsrate(a, 100, "heun") - exakt)
    assert 3.5 < f1 / f2 < 4.5                      # halber Schritt: Fehler durch 4


def test_methode_der_schritte_gegen_eigene_bruchrechnung():
    """Unabhängiger Weg: Polynome als Koeffizientenlisten aus Brüchen; x(k+1) = x(k) - a ∫ x(s-1) ds über den vorigen Abschnitt (a = 11/10, kein Vielfaches von 2)."""
    a = Fraction(11, 10)

    def verschiebe(coeffs, d):
        """p(t - d) als Koeffizientenliste in t (binomische Entwicklung)."""
        out = [Fraction(0)] * len(coeffs)
        for i, c in enumerate(coeffs):
            for j in range(i + 1):
                out[j] += c * math.comb(i, j) * (-d) ** (i - j)
        return out

    def integriere_ab(coeffs, start):
        """Stammfunktion mit Wert 0 bei `start`."""
        s = [Fraction(0)] + [c / (i + 1) for i, c in enumerate(coeffs)]
        s[0] = -sum(c * start**i for i, c in enumerate(s))
        return s

    prev = [Fraction(1)]
    xk = Fraction(1)
    werte = [xk]
    for k in range(5):
        schiebung = verschiebe(prev, 1)                    # x(s - 1) als Polynom in s
        integ = integriere_ab(schiebung, Fraction(k))
        cur = [-a * c for c in integ]
        cur[0] += xk
        xk = sum(c * Fraction(k + 1) ** i for i, c in enumerate(cur))
        werte.append(xk)
        prev = cur
    _, num = A.methode_der_schritte(1.1, 5)
    assert num == pytest.approx([float(w) for w in werte], abs=1e-12)
    # zusätzlich der von Hand integrierte Abschnitt 1: x(1) = 1 - a
    assert werte[1] == 1 - a                           # von Hand: x(1) = 1 - a * Integral von 0 bis 1 über 1


def test_methode_der_schritte_gegen_heun_bei_beliebigem_a():
    for a in (0.3, 1.1, 1.9):
        _, vals = A.methode_der_schritte(a, 6)
        x = A.linear_dde(a, 2000, 6)
        for k in range(7):
            assert vals[k] == pytest.approx(float(x[k * 2000]), abs=2e-6), (a, k)


def test_skalieren_der_vorgeschichte_aendert_die_wachstumsrate_nicht_und_skaliert_die_loesung():
    x1 = A.linear_dde(1.45, 100, 30, hist=1.0)
    x3 = A.linear_dde(1.45, 100, 30, hist=3.0)
    assert np.allclose(x3, 3.0 * x1, rtol=1e-12, atol=1e-12)


# --- verzögerte Logistik gegen die Linearisierung --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("r, amplitude", [(1.2, 1e-5), (1.4, 1e-5), (1.7, 1e-9)])
def test_hutchinson_kleine_abweichung_wie_re_w0(r, amplitude):
    """Konstanter Anfangsverlauf 1 + amplitude: solange die Abweichung klein bleibt (1e-12 < |u| < 3e-5), ist sie die lineare Gleichung u' = -r u(t - 1);
    Hüllkurvensteigung der Extrema gegen Re W0(-r). Unter 1e-12 ist x - 1 Rundungsrauschen (x ≈ 1 hat nur 16 Stellen)."""
    m = 400
    x = A.hutchinson(r, m, 220, x0=1.0 + amplitude)
    u = x - 1.0
    t, v = A._extrema_envelope(u, 1.0 / m, 10.0)
    sel = (v < 3e-5) & (v > 1e-12)
    assert sel.sum() > 20
    steigung = np.polyfit(t[sel], np.log(v[sel]), 1)[0]
    assert steigung == pytest.approx(complex(lambertw(-r, 0)).real, abs=3e-4)


# --- Mackey-Glass ---------------------------------------------------------------------------------------------------------------------------


def test_mg_wurzel_gegen_newton_aus_vielen_startwerten():
    """Unabhängig von der W-Funktion: Newton auf s + γ + b e^{-sτ} = 0 aus einem Gitter von Startwerten, größter Realteil."""
    gamma, b = A.mg_linearisierung()
    tau = 17.0
    best = None
    for re0 in np.linspace(-0.5, 0.5, 9):
        for im0 in np.linspace(0.0, 3.0, 40):
            s = complex(re0, im0)
            for _ in range(60):
                f = s + gamma + b * np.exp(-s * tau)
                s -= f / (1 - b * tau * np.exp(-s * tau))
            if abs(s + gamma + b * np.exp(-s * tau)) < 1e-10 and (best is None or s.real > best.real):
                best = s
    assert best.real == pytest.approx(A.mg_groesste_wurzel(tau).real, abs=1e-8)
    assert best.real == pytest.approx(0.0413, abs=5e-5)         # Handbeispiel des Plans (Newton)


def test_mg_variationsgleichung_gegen_zentrale_differenz():
    """Die Ableitung des Heun-Blocks nach dem Fenster: Variationsgleichung (g' von Hand) gegen (F(w + e d) - F(w - e d)) / 2e."""
    mg = A._MG(17.0)
    w = A.mg_start_fenster(mg.m, 3)
    for _ in range(40):
        w = mg.advance(w)
    rng = np.random.default_rng(1)
    d = rng.standard_normal(mg.m + 1)
    eps = 1e-6
    fd = (mg.advance(w + eps * d) - mg.advance(w - eps * d)) / (2 * eps)
    tan = mg.tangent(w, d)
    assert np.max(np.abs(fd - tan)) < 1e-7 * max(1.0, np.max(np.abs(tan)))


def test_mg_zweibahn_gegen_variationsgleichung():
    """Zwei verschiedene Wege zur selben Größe: gleiche Bahn, gleiche Norm; der Unterschied ist nur das endliche d0 = 1e-8."""
    for seed in (3, 5):
        zb = A.mg_lyapunov(17.0, 12000, 5000, seed, "zweibahn")["lambda"]
        tg = A.mg_lyapunov(17.0, 12000, 5000, seed, "tangente")["lambda"]
        assert zb == pytest.approx(tg, abs=2e-6)


def test_mg_exponent_ist_bei_tau_6_und_12_null_und_bei_17_positiv():
    """Periodische Bahnen (τ = 6, 12) und der chaotische Fall (τ = 17)."""
    assert abs(A.mg_lyapunov(6.0, 12000, 5000, 3, "tangente")["lambda"]) < 1e-3
    assert abs(A.mg_lyapunov(12.0, 12000, 5000, 3, "tangente")["lambda"]) < 1e-3
    assert A.mg_lyapunov(17.0, 12000, 5000, 3, "tangente")["lambda"] > 3e-3


def test_mg_gleicher_zufalls_seed_gleiche_ausgabe_anderer_seed_andere_bahn():
    a = A.mg_lyapunov(17.0, 3000, 1000, 8, "zweibahn")["lambda"]
    b = A.mg_lyapunov(17.0, 3000, 1000, 8, "zweibahn")["lambda"]
    c = A.mg_lyapunov(17.0, 3000, 1000, 9, "zweibahn")["lambda"]
    assert a == b and a != c


# --- Lorenz ---------------------------------------------------------------------------------------------------------------------------------


def test_lorenz_jacobi_eigenwerte_gegen_charakteristisches_polynom():
    """λ³ + (σ+β+1) λ² + β(σ+ρ) λ + 2σβ(ρ-1) = 0 (von Hand hergeleitet) gegen die Eigenwerte der Jacobi-Matrix am Fixpunkt."""
    sg, be = C.LOR_SIGMA, C.LOR_BETA
    for rho in (10.0, 20.0, 28.0, 60.0):
        wurzeln = np.roots([1.0, sg + be + 1, be * (sg + rho), 2 * sg * be * (rho - 1)])
        ev = np.linalg.eigvals(L.jacobi(L.fixpunkt_c(rho), rho))
        assert sorted(wurzeln, key=lambda z: (z.real, z.imag)) == pytest.approx(sorted(ev, key=lambda z: (z.real, z.imag)), abs=1e-8)


def test_lorenz_hopf_rho_aus_den_eigenwerten_per_bisektion():
    def re_paar(rho):
        ev = np.linalg.eigvals(L.jacobi(L.fixpunkt_c(rho), rho))
        return max(e.real for e in ev if abs(e.imag) > 1e-9)

    lo, hi = 20.0, 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if re_paar(mid) < 0:
            lo = mid
        else:
            hi = mid
    assert 0.5 * (lo + hi) == pytest.approx(L.hopf_rho(), abs=1e-6)


def test_lorenz_rk4_gegen_dop853():
    """Eine Zeiteinheit mit dt = 0,001 gegen das Bibliotheksverfahren DOP853 (rtol 1e-13); der Fehler von RK4 liegt bei etwa 3e-8."""
    s0 = np.array([1.0, 2.0, 20.0])
    ref = solve_ivp(lambda t, s: L.rechte_seite(s, 28.0), (0, 1.0), s0, method="DOP853", rtol=1e-13, atol=1e-13).y[:, -1]
    S = s0.copy()
    for _ in range(1000):
        S = L.rk4(S, 28.0, 0.001)
    assert np.max(np.abs(S - ref)) < 1e-6


def test_lorenz_rk4_hat_ordnung_vier():
    s0 = np.array([1.0, 2.0, 20.0])
    ref = solve_ivp(lambda t, s: L.rechte_seite(s, 28.0), (0, 1.0), s0, method="DOP853", rtol=1e-13, atol=1e-13).y[:, -1]

    def fehler(n):
        S = s0.copy()
        for _ in range(n):
            S = L.rk4(S, 28.0, 1.0 / n)
        return np.max(np.abs(S - ref))

    verhaeltnis = fehler(800) / fehler(1600)
    assert 12 < verhaeltnis < 20                      # Ordnung 4: Faktor 16 (gemessen 15,0)


def test_lorenz_symmetrie_x_y_minus():
    s = np.array([3.0, -2.0, 15.0])
    spiegel = np.array([-3.0, 2.0, 15.0])
    a = L.rk4(s, 28.0, 0.01)
    b = L.rk4(spiegel, 28.0, 0.01)
    assert np.allclose(b, [-a[0], -a[1], a[2]], atol=1e-13)


@pytest.mark.parametrize("rho", [10.0, 20.0, 28.0, 40.0, 60.0])
def test_lorenz_summe_der_exponenten_gleich_spur(rho):
    sp = L.spektrum(rho, 3, n_starts=40)
    assert sp["summe"] == pytest.approx(L.spur(), abs=1e-3)


def test_lorenz_exponent_in_flussrichtung_ist_null_bei_chaos():
    sp = L.spektrum(28.0, 3, n_starts=100, t_mess=40.0)
    assert abs(sp["lam"][1]) < 0.02                  # etwa das Vierfache der Streuung des Mittels (0,003)


def test_lorenz_unterhalb_hopf_bahn_geht_zum_fixpunkt():
    """1 < ρ < ρ_H: die Bahnen laufen nach C+ oder C- (x = y = ±c, z = ρ-1)."""
    rho = 20.0
    S = L.einschwingen(L.startwerte(rho, 12, 5), rho, 80.0, 0.01)
    c = math.sqrt(C.LOR_BETA * (rho - 1))
    assert np.allclose(np.abs(S[:, 0]), c, atol=5e-3) and np.allclose(S[:, 2], rho - 1, atol=5e-3)


def test_lorenz_unter_eins_geht_alles_gegen_null():
    S = L.einschwingen(L.startwerte(0.5, 10, 2), 0.5, 60.0, 0.01)
    assert np.max(np.abs(S)) < 1e-6


def test_lorenz_gleicher_zufalls_seed_gleiche_ausgabe():
    a = L.spektrum(28.0, 3, n_starts=8, t_burn=10, t_mess=5)["lam"]
    b = L.spektrum(28.0, 3, n_starts=8, t_burn=10, t_mess=5)["lam"]
    assert np.array_equal(a, b)


def test_lorenz_divergenz_bei_rho_28_liegt_im_erwarteten_band():
    """ln(1e9)/λ₁ = 23,1 aus λ₁ = 0,90; gemessen in Mittel um 21 (Band 15 bis 25)."""
    dv = L.divergenz(28.0, 3)
    z = dv["zeiten"]
    assert not np.isnan(z).any() and 18.0 < float(np.mean(z)) < 24.0 and 12.0 < float(z.min()) and float(z.max()) < 28.0
