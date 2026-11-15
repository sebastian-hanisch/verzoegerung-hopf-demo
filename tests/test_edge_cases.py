"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Schrittzahl, Grenzwerte, ungültige Eingaben, Ein-Bahn-Fälle."""

import math

import numpy as np
import pytest

import vzh_algorithm as A
import vzh_constants as C
import vzh_evaluation as E
import vzh_lorenz as L


def test_kleinster_fall_ein_schritt_je_verzoegerung_und_ein_abschnitt():
    """m = 1, n_tau = 1 (Euler und Heun): Länge 2 (t = 0 und 1), Startwert 1. Von Hand: Euler x(1) = 1 - a, Heun x(1) = 1 - a (u_0 = u_1 = 1)."""
    for method in ("euler", "heun"):
        x = A.linear_dde(1.45, 1, 1, method)
        assert len(x) == 2 and x[0] == 1.0 and x[1] == pytest.approx(1 - 1.45)


def test_genau_zwei_abschnitte_methode_der_schritte():
    pieces, values = A.methode_der_schritte(1.0, 2)
    assert len(pieces) == 2 and values == pytest.approx([1.0, 0.0, -0.5])


def test_ungueltige_eingaben_werden_abgelehnt():
    with pytest.raises(ValueError):
        A.linear_dde(1.0, 0, 5)
    with pytest.raises(ValueError):
        A.linear_dde(1.0, 5, 0)
    with pytest.raises(ValueError):
        A.linear_dde(1.0, 5, 5, "rk4")
    with pytest.raises(ValueError):
        A.methode_der_schritte(1.0, 0)
    with pytest.raises(ValueError):
        A.diskrete_wachstumsrate(1.0, 5, "rk4")
    with pytest.raises(ValueError):
        A.mg_lyapunov(17.0, 100, 100, 1, "unbekannt")


def test_ohne_rueckkopplung_bleibt_die_loesung_konstant():
    assert np.all(A.linear_dde(0.0, 20, 10) == 1.0)
    assert A.rechtester_wurzel(1e-9).real == pytest.approx(0.0, abs=1e-8)


def test_euler_bei_m_1_und_a_1_liegt_genau_auf_der_grenze():
    """Euler-Grenze für m = 1 ist 2·sin(π/6) = 1: z² - z + 1 = 0 hat |z| = 1, die Wachstumsrate ist 0 (inklusive Grenze)."""
    assert A.euler_grenze(1) == pytest.approx(1.0, abs=1e-15)
    assert abs(A.diskrete_wachstumsrate(1.0, 1, "euler")) < 1e-9


def test_a_gleich_pi_halbe_nur_im_band():
    """Genau π/2 nie auf Gleitkomma-Gleichheit: Re s liegt im Band 1e-6, die Simulation im Band 1e-3."""
    a = math.pi / 2
    assert abs(A.rechtester_wurzel(a).real) < 1e-12
    assert abs(A.gemessene_wachstumsrate(a, 400, "heun")) < 1e-3


def test_monotones_abklingen_hat_keine_extrema_und_nutzt_den_betrag():
    """a = 0,1 < 1/e: die Wurzel ist reell und die Messung über die Beträge bei ganzen τ stimmt mit ihr überein."""
    a = 0.1
    assert abs(A.rechtester_wurzel(a).imag) < 1e-12
    assert A.gemessene_wachstumsrate(a, 400, "heun") == pytest.approx(A.rechtester_wurzel(a).real, abs=1e-4)


def test_vorgeschichte_gleich_gleichgewicht_hat_spannweite_null():
    x = A.hutchinson(1.8, 100, 400, x0=1.0)
    assert A.hutchinson_kennzahlen(x, 100)["span"] < 1e-12


def test_verzoegerung_groesser_als_die_messdauer():
    """t_mess < τ: es gibt trotzdem einen Block und ein endliches Ergebnis."""
    r = A.mg_lyapunov(17.0, 5.0, 100.0, 1, "zweibahn")
    assert math.isfinite(r["lambda"]) and r["t_mess"] == 17.0


def test_mackey_glass_tau_extreme_der_regler_rechnen():
    for tau in (C.MG_TAU_MIN, C.MG_TAU_MAX):
        g = E.analyse_mackey_glass(tau, 1)
        assert math.isfinite(g.lam_zweibahn) and len(g.t) == len(g.x)
        assert len(g.x_zuvor) == len(g.x_jetzt)


def test_lorenz_bei_rho_h_hat_ein_paar_auf_der_imaginaeren_achse():
    ev = np.linalg.eigvals(L.jacobi(L.fixpunkt_c(L.hopf_rho()), L.hopf_rho()))
    assert min(abs(ev.real)) < 1e-9


def test_lorenz_eine_einzelne_bahn_hat_streuung_null_statt_fehler():
    sp = L.spektrum(28.0, 1, n_starts=1, t_burn=5, t_mess=3)
    assert np.all(sp["sem"] == 0.0) and len(sp["lam"]) == 3


def test_lorenz_rho_grenzen_der_regler_rechnen():
    for rho in (C.RHO_MIN, C.RHO_MAX):
        lo = E.analyse_lorenz(rho, 1)
        assert all(math.isfinite(v) for v in lo.lam)


def test_de_schreibt_nie_minus_null():
    assert E.de(-0.00001, 2) == "0,00" and E.de(-0.00001, 2, plus=True) == "+0,00"
    assert E.de(-1.5, 1) == "−1,5" and E.de(2.25, 2, plus=True) == "+2,25"
