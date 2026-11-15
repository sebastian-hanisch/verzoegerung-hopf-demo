"""Einheiten des Kerns an Handbeispielen (DEMO-PLAYBOOK §1b): Methode der Schritte, Kennlinie, Blockfassung gegen Schritt-für-Schritt, Mackey-Glass-Linearisierung, Lorenz.

Der Erwartungswert steht je Test als Rechnung (Brüche), nicht als abgeschriebene Ausgabe des Codes."""

import math
from fractions import Fraction

import numpy as np
import pytest

import vzh_algorithm as A
import vzh_constants as C
import vzh_lorenz as L
from vzh_rng import SplitMix64


def test_methode_der_schritte_handwerte(hand):
    pieces, values = A.methode_der_schritte(hand["a"], 4)
    assert values == pytest.approx(hand["werte"], abs=1e-12)
    assert len(pieces) == 4
    # Stück 1: x(t) = 1 - t
    assert pieces[0](0.25) == pytest.approx(0.75)
    # Stück 2: x(t) = (t - 1)(t - 3)/2
    assert pieces[1](1.5) == pytest.approx((0.5) * (-1.5) / 2)


def test_heun_trifft_die_handwerte(hand):
    x = A.linear_dde(hand["a"], 1000, 4)
    got = [float(x[k * 1000]) for k in range(5)]
    assert got == pytest.approx(hand["werte"], abs=1e-5)
    x100 = A.linear_dde(hand["a"], 100, 3)
    assert float(x100[300]) == pytest.approx(-1.0 / 6.0, abs=1e-5)


def test_kennlinie_bei_pi_halbe_ist_rein_imaginaer():
    s = A.rechtester_wurzel(math.pi / 2)
    assert abs(s.real) < 1e-12 and s.imag == pytest.approx(math.pi / 2, abs=1e-12)
    # von Hand: (i pi/2) e^{i pi/2} = (i pi/2) i = -pi/2
    assert abs(1j * math.pi / 2 * np.exp(1j * math.pi / 2) + math.pi / 2) < 1e-15
    assert 2 * math.pi / s.imag == pytest.approx(4.0, abs=1e-12)


@pytest.mark.parametrize("a", [0.3, 1.4, 1.7])
def test_wurzeln_erfuellen_die_kennlinie(a):
    for s in A.kennlinie_wurzeln(a):
        assert abs(s + a * np.exp(-s)) < 1e-10


def test_bei_a_gleich_eins_ist_die_rechteste_wurzel_gedaempft():
    s = A.rechtester_wurzel(1.0)
    assert s.real == pytest.approx(-0.31813, abs=1e-5) and abs(s.imag) == pytest.approx(1.33724, abs=1e-5)


@pytest.mark.parametrize("m", [1, 2, 5])
def test_euler_grenze_handwerte(m):
    # Handrechnung 2 m sin(pi/(4m+2)): m = 1: 2 sin(pi/6) = 1; m = 2: 4 sin(pi/10); m = 5: 10 sin(pi/22)
    erwartet = {1: 2 * 0.5, 2: 4 * math.sin(math.pi / 10), 5: 10 * math.sin(math.pi / 22)}[m]
    assert A.euler_grenze(m) == pytest.approx(erwartet)
    assert A.euler_grenze(m) < math.pi / 2


def test_euler_grenze_strebt_gegen_pi_halbe():
    werte = [A.euler_grenze(m) for m in (1, 2, 5, 10, 100, 1000)]
    assert werte == sorted(werte)
    assert werte[-1] == pytest.approx(math.pi / 2, abs=1e-3)


@pytest.mark.parametrize("method", ["heun", "euler"])
@pytest.mark.parametrize("a", [0.5, 1.4, 1.7])
def test_blockfassung_gleich_schritt_fuer_schritt_linear(a, method):
    x1 = A.linear_dde(a, 50, 20, method)
    x2 = A.linear_dde_stepwise(a, 50, 20, method)
    assert np.max(np.abs(x1 - x2)) < 1e-12


@pytest.mark.parametrize("r", [1.4, 1.7, 2.2])
def test_blockfassung_gleich_schritt_fuer_schritt_hutchinson(r):
    x1 = A.hutchinson(r, 50, 60)
    x2 = A.hutchinson_stepwise(r, 50, 60)
    assert np.max(np.abs(x1 - x2)) < 1e-11


def test_hutchinson_gleichgewicht_bleibt_stehen_und_positivitaet():
    assert np.allclose(A.hutchinson(1.7, 100, 50, x0=1.0), 1.0, atol=1e-14)
    for r in (1.4, 2.2, 3.0):
        assert np.all(A.hutchinson(r, 100, 300) > 0)


def test_hutchinson_kennzahlen_an_einer_bekannten_schwingung():
    """Handkonstruiert: x = 1 + 0,5 sin(2 pi t/4) hat Spannweite 1 und Periode 4 (m = 400 Schritte je Einheit)."""
    m = 400
    t = np.arange(400 * m) / m
    x = 1.0 + 0.5 * np.sin(2 * np.pi * t / 4.0)
    kz = A.hutchinson_kennzahlen(x, m, tail_tau=300)
    assert kz["span"] == pytest.approx(1.0, abs=1e-6) and kz["period"] == pytest.approx(4.0, abs=2e-3)


def test_fit_der_spannweite_findet_die_nullstelle_einer_exakten_geraden():
    rs = np.array([1.2, 1.4, 1.6])
    spans = np.sqrt(3.0 * rs - 3.0)           # Spannweite² = 3 r - 3, Nullstelle 1
    c1, c0, r0 = A.fit_spannweite_quadrat(rs, spans)
    assert (c1, c0, r0) == pytest.approx((3.0, -3.0, 1.0))


def test_mackey_glass_handwerte_der_linearisierung():
    # x* = (beta/gamma - 1)^(1/n) = (2 - 1)^(1/10) = 1;  b = -0,2 (1 + 1 - 10)/(2)^2 = 0,4
    assert A.mg_gleichgewicht() == pytest.approx(1.0)
    gamma, b = A.mg_linearisierung()
    assert (gamma, b) == pytest.approx((0.1, 0.4))
    # tau_c = arccos(-0,25)/sqrt(0,16 - 0,01)
    assert A.mg_hopf_tau() == pytest.approx(math.acos(-0.25) / math.sqrt(0.15), rel=1e-12)
    assert A.mg_hopf_tau() == pytest.approx(4.708196, abs=1e-6)


def test_mackey_glass_kennlinie_bei_hopf_grenze_ist_rein_imaginaer():
    gamma, b = A.mg_linearisierung()
    tc = A.mg_hopf_tau()
    w = math.sqrt(b * b - gamma * gamma)
    s = 1j * w
    assert abs(s + gamma + b * np.exp(-s * tc)) < 1e-12
    # größte Wurzel knapp unter und knapp über tau_c: Vorzeichen des Realteils wechselt
    assert A.mg_groesste_wurzel(tc - 0.05).real < 0 < A.mg_groesste_wurzel(tc + 0.05).real


def test_mackey_glass_wurzel_bei_tau_17_erfuellt_die_kennlinie():
    gamma, b = A.mg_linearisierung()
    s = A.mg_groesste_wurzel(17.0)
    assert abs(s + gamma + b * np.exp(-s * 17.0)) < 1e-10
    assert s.real > 0


def test_mackey_glass_block_gleich_schritt_fuer_schritt():
    """Blockfassung (lfilter) gegen die Schleife des Heun-Schritts: x' = F(x, x(t - tau)), F(x, xd) = beta xd/(1 + xd^n) - gamma x."""
    mg = A._MG(5.0, 0.1)
    w = A.mg_start_fenster(mg.m, 4)
    h, beta, gamma, n = mg.h, C.MG_BETA, C.MG_GAMMA, C.MG_N

    def f(x, xd):
        return beta * xd / (1 + xd**n) - gamma * x

    ref = [w[-1]]
    for k in range(mg.m):
        xi = ref[-1]
        k1 = f(xi, w[k])
        k2 = f(xi + h * k1, w[k + 1])
        ref.append(xi + 0.5 * h * (k1 + k2))
    assert np.max(np.abs(mg.advance(w) - np.array(ref))) < 1e-12


def test_mackey_glass_gleichgewicht_bleibt_stehen():
    mg = A._MG(17.0)
    w = np.ones(mg.m + 1)
    assert np.allclose(mg.advance(w), 1.0, atol=1e-13)


def test_lorenz_spur_und_hopf_in_brueche():
    sg, be = Fraction(10), Fraction(8, 3)
    assert -(sg + 1 + be) == Fraction(-41, 3) and L.spur() == pytest.approx(-41 / 3)
    rho_h = sg * (sg + be + 3) / (sg - be - 1)
    assert rho_h == Fraction(470, 19) and L.hopf_rho() == pytest.approx(470 / 19)
    # Routh-Hurwitz: a2 a1 = a0 bei rho_H, exakt
    assert (sg + be + 1) * be * (sg + rho_h) == 2 * sg * be * (rho_h - 1)


def test_lorenz_eigenwerte_bei_hopf_rho_sind_minus_13_667_und_imaginaer():
    ev = np.linalg.eigvals(L.jacobi(L.fixpunkt_c(L.hopf_rho()), L.hopf_rho()))
    realteile = sorted(ev.real)
    assert realteile[0] == pytest.approx(-41 / 3, abs=1e-9)
    assert abs(realteile[1]) < 1e-9 and abs(realteile[2]) < 1e-9
    assert max(abs(ev.imag)) == pytest.approx(9.62453, abs=1e-5) == pytest.approx(L.hopf_frequenz(), abs=1e-5)


def test_lorenz_fixpunkt_ist_nullstelle_des_feldes():
    for rho in (10.0, 24.0, 28.0):
        c = L.fixpunkt_c(rho)
        assert np.allclose(L.rechte_seite(c, rho), 0.0, atol=1e-12)


def test_splitmix64_bekannter_erster_wert():
    """Referenzwert des SplitMix64 mit Zufalls-Seed 0 (Vigna, öffentliche Referenzimplementierung): 0xE220A8397B1DCDAF."""
    assert SplitMix64(0).next_u64() == 0xE220A8397B1DCDAF
    assert 0.0 <= SplitMix64(7).random() < 1.0


def test_startverlauf_hat_die_vorgeschriebene_lage_und_ist_vom_seed_abhaengig():
    w1, w2 = A.mg_start_fenster(170, 5), A.mg_start_fenster(170, 5)
    assert np.array_equal(w1, w2) and not np.array_equal(w1, A.mg_start_fenster(170, 6))
    assert abs(w1.mean() - 1.2) < 0.02 and abs(w1.std() - 0.05) < 0.01
