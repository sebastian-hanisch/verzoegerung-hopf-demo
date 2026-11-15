"""Claims-Tests: jede Zahl der README steht hier als Band oder Zählung, mit der Zahl der Läufe. Gliederung wie die Befund-Tabelle der README (Zeilen B1 bis B12).

Zufall nur über die festen Zufalls-Seeds unten; „Läufe“ nennt die Zahl der Zufalls-Seeds bzw. Startverläufe. Nie eine Zweierpotenz als a·τ, r·τ oder ρ."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import vzh_algorithm as A
import vzh_constants as C
import vzh_evaluation as E
import vzh_lorenz as L
from vzh_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent
VOR = json.loads((ROOT / "vzh_vorgerechnet.json").read_text(encoding="utf-8"))
LOR = {r["rho"]: r for r in VOR["lorenz"]["rows"]}
MG = {r["tau"]: r for r in VOR["mackey_glass"]["rows"]}


# --- B1: lineare Gleichung ---------------------------------------------------------------------------------------------------------------


def test_b1_methode_der_schritte_handwerte():
    _, v = A.methode_der_schritte(1.0, 4)
    assert v == pytest.approx([1.0, 0.0, -1 / 2, -1 / 6, 5 / 24], abs=1e-12)


def test_b1_wachstumsrate_gleich_lambertw_in_7_von_7_werten_hoechstens_2e_6():
    """README: 7 Werte a·τ (0,5; 1,0; 1,4; 1,55; 1,6; 1,7; 2,0), größter Unterschied 1,2e-6 (Heun, 400 Schritte je τ)."""
    devs = [abs(A.gemessene_wachstumsrate(a, 400, "heun") - A.rechtester_wurzel(a).real) for a in (0.5, 1.0, 1.4, 1.55, 1.6, 1.7, 2.0)]
    assert len(devs) == 7 and max(devs) < 2e-6


def test_b1_periode_bei_pi_halbe_ist_vier_tau():
    s = A.rechtester_wurzel(math.pi / 2)
    assert 2 * math.pi / s.imag == pytest.approx(4.0, abs=1e-9)


def test_b1_wurzeln_bei_a_1_und_a_2():
    s1, s2 = A.rechtester_wurzel(1.0), A.rechtester_wurzel(2.0)
    assert (s1.real, s1.imag) == pytest.approx((-0.31813, 1.33724), abs=1e-5)
    assert (s2.real, s2.imag) == pytest.approx((0.17282, 1.67369), abs=1e-5)
    assert 2 * math.pi / s1.imag == pytest.approx(4.6986, abs=1e-4)


def test_b1_grenze_ein_durch_e_trennt_monoton_und_schwingend():
    assert abs(A.rechtester_wurzel(0.36).imag) < 1e-12 and abs(A.rechtester_wurzel(0.38).imag) > 0.01


# --- B2: Hutchinson ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("r, span", [(1.60, 0.786), (1.65, 1.276), (1.8, 2.101), (2.2, 3.503), (3.0, 7.582)])
def test_b2_spannweiten_hinter_der_grenze(r, span):
    """README: Heun, 400 Schritte je τ, Vorgeschichte 0,5, 2000 Verzögerungen, Spannweite der letzten 300: Bänder ±0,002."""
    assert E.analyse_hutchinson(r).span == pytest.approx(span, abs=2e-3)


@pytest.mark.parametrize("r", [1.40, 1.55])
def test_b2_unter_der_grenze_ist_die_spannweite_null(r):
    assert E.analyse_hutchinson(r).span < 1e-6


def _hist(m, k):
    rng = SplitMix64(1000 + k)
    return np.array([1.0 + 0.4 * (rng.random() - 0.5) for _ in range(m + 1)])


def test_b2_gegenprobe_60_laeufe_bei_1_55_und_1_60():
    """20 Anfangsverläufe (1 + 0,4 (u - 0,5), u gleichverteilt je Stützstelle) und je 3 Schrittweiten (m = 200, 400, 800), 2000 Verzögerungen:
    bei r·τ = 1,55 ist die Spannweite in 60 von 60 Läufen höchstens 1e-6 (gemessen 3,5e-8), bei 1,60 in 60 von 60 zwischen 0,785 und 0,787."""
    s155, s160 = [], []
    for m in (200, 400, 800):
        for k in range(20):
            h = _hist(m, k)
            s155.append(A.hutchinson_kennzahlen(A.hutchinson(1.55, m, 2000, x0=h), m)["span"])
            s160.append(A.hutchinson_kennzahlen(A.hutchinson(1.60, m, 2000, x0=h), m)["span"])
    assert len(s155) == 60 and max(s155) < 1e-6
    assert len(s160) == 60 and 0.785 < min(s160) and max(s160) < 0.787


def test_b2_gerade_spannweite_quadrat_nullstelle_und_perioden():
    fit = E.analyse_hutchinson_fit()
    assert fit.r0 == pytest.approx(1.5691, abs=5e-4) and abs(fit.r0 - math.pi / 2) < 3e-3
    assert fit.c1 == pytest.approx(19.894, abs=0.05) and fit.c0 == pytest.approx(-31.216, abs=0.1)
    _, _, r0_vier = A.fit_spannweite_quadrat(fit.rs[:4], fit.spans[:4])
    assert r0_vier == pytest.approx(1.5700, abs=5e-4)
    perioden = [A.hutchinson_kennzahlen(A.hutchinson(r, 400, 3000), 400)["period"] for r in C.HUT_FIT_RS]
    assert perioden == pytest.approx([4.006, 4.020, 4.035, 4.057, 4.096], abs=2e-3)
    assert perioden == sorted(perioden)


# --- B3: Zeitschritt --------------------------------------------------------------------------------------------------------------------


def test_b3_euler_grenzen_der_readme():
    assert [round(A.euler_grenze(m), 3) for m in (1, 2, 5, 10, 100)] == [1.000, 1.236, 1.423, 1.495, 1.563]


def test_b3_bei_a_1_45_rechnet_euler_bis_m_5_instabil_und_ab_m_7_stabil():
    raten = {m: A.diskrete_wachstumsrate(1.45, m, "euler") for m in (1, 2, 3, 5, 7, 10)}
    assert [raten[m] > 0 for m in (1, 2, 3, 5)] == [True] * 4 and [raten[m] < 0 for m in (7, 10)] == [True] * 2
    assert raten[5] == pytest.approx(0.0121, abs=5e-5) and raten[7] == pytest.approx(-0.0061, abs=5e-5)
    assert raten[1] == pytest.approx(0.1858, abs=5e-4)
    assert A.rechtester_wurzel(1.45).real == pytest.approx(-0.0568, abs=5e-5)


def test_b3_heun_ist_bei_a_1_45_schon_ab_m_1_stabil():
    raten = [A.diskrete_wachstumsrate(1.45, m, "heun") for m in (1, 2, 3, 5, 7, 10)]
    assert all(r < 0 for r in raten)
    assert raten[0] == pytest.approx(-0.1608, abs=5e-4) and raten[3] == pytest.approx(-0.0621, abs=5e-4)
    assert abs(raten[0] / A.rechtester_wurzel(1.45).real - 2.83) < 0.01


# --- B4: Mackey-Glass -------------------------------------------------------------------------------------------------------------------


def test_b4_hopf_grenze_und_wurzel_bei_tau_17():
    assert A.mg_hopf_tau() == pytest.approx(4.7082, abs=5e-5)
    assert A.mg_groesste_wurzel(17.0).real == pytest.approx(0.0413, abs=5e-5)


def test_b4_exponent_bei_tau_17_ueber_20_zufalls_seeds_t_12000():
    """README: T = 12 000 (5000 Einschwingzeit), 20 Zufalls-Seeds (1 bis 20): Mittel 0,0052, Streuung 0,0003, Spanne 0,0047 bis 0,0057."""
    lam = [A.mg_lyapunov(17.0, 12000, 5000, s, "zweibahn")["lambda"] for s in range(1, 21)]
    assert len(lam) == 20
    assert float(np.mean(lam)) == pytest.approx(0.00517, abs=2e-4) and float(np.std(lam, ddof=1)) == pytest.approx(0.00033, abs=2e-4)
    assert 0.0045 < min(lam) and max(lam) < 0.0060


def test_b4_exponent_bei_tau_17_ueber_20_zufalls_seeds_t_60000():
    """README: T = 60 000 (die Live-Messung der App), 20 Zufalls-Seeds: Mittel 0,0051, Streuung 0,0001, Spanne 0,0049 bis 0,0054; Vorhersagezeit 1/λ im Mittel 195."""
    lam = [A.mg_lyapunov(17.0, 60000, 5000, s, "zweibahn")["lambda"] for s in range(1, 21)]
    assert float(np.mean(lam)) == pytest.approx(0.00514, abs=1.5e-4) and float(np.std(lam, ddof=1)) == pytest.approx(0.00014, abs=1e-4)
    assert 0.0047 < min(lam) and max(lam) < 0.0057
    assert 1.0 / float(np.mean(lam)) == pytest.approx(195.0, abs=8.0)
    mittel = float(np.mean(lam))
    assert (mittel - min(lam)) / mittel < 0.08 and (max(lam) - mittel) / mittel < 0.08       # „etwa ±6 %“ in App und README (gemessen −5 % und +6 %)


def test_b4_wertebereich_der_loesung_bei_tau_17():
    r = A.mg_lyapunov(17.0, 12000, 5000, 3, "zweibahn")
    assert 0.40 < r["min"] < 0.44 and 1.31 < r["max"] < 1.33


def test_b4_vorgerechnetes_gitter_in_tau():
    """README (vorgerechnet, Variationsgleichung, 3 Zufalls-Seeds, 40 000 Zeiteinheiten je Lauf): τ = 5 bis 16 alle |λ| < 1e-4 (12 von 12),
    τ = 17 bis 30: 13 von 14 Werten > 0,0005, nur τ = 19 nicht."""
    ruhig = [abs(MG[t]["lam"]) for t in range(5, 17)]
    assert len(ruhig) == 12 and max(ruhig) < 1e-4
    positiv = [t for t in range(17, 31) if MG[t]["lam"] > 5e-4]
    assert len(positiv) == 13 and 19 not in positiv
    assert MG[17]["lam"] == pytest.approx(0.0051, abs=2e-4) and MG[24]["lam"] == pytest.approx(0.0104, abs=2e-4)
    assert max(r["lam"] for r in MG.values()) == MG[24]["lam"]
    assert 0.0006 < MG[21]["min"] and MG[21]["max"] < 0.0016 and MG[21]["lam"] == pytest.approx(0.0010, abs=2e-4)   # je Zufalls-Seed 0,0007 bis 0,0015


# --- B5: Lorenz --------------------------------------------------------------------------------------------------------------------------


def test_b5_hopf_schwelle_und_spur():
    assert L.hopf_rho() == pytest.approx(24.7368, abs=5e-5) and L.hopf_frequenz() == pytest.approx(9.6245, abs=5e-5)
    assert L.spur() == pytest.approx(-13.6667, abs=5e-5)


def test_b5_vorgerechnete_summe_gleich_spur_in_22_von_22_werten():
    assert len(LOR) == 22 and all(abs(r["summe"] + 41 / 3) < 5e-4 for r in LOR.values())


def test_b5_vorgerechnetes_lambda1_bei_rho_28_und_zeitschritt():
    """README: 100 Bahnen, 100 Zeiteinheiten Messdauer, RK4 dt = 0,005: λ₁(28) = 0,90 ± 0,002; dt = 0,01 / 0,005 / 0,0025: 0,904 / 0,908 / 0,903 (alle im Band 0,89 bis 0,92)."""
    assert LOR[28]["lam"][0] == pytest.approx(0.902, abs=3e-3)
    assert [z["lam1"] for z in VOR["lorenz"]["zeitschritt_rho28"]] == pytest.approx([0.9035, 0.9079, 0.9029], abs=1e-4)
    assert all(0.89 < z["lam1"] < 0.92 for z in VOR["lorenz"]["zeitschritt_rho28"])


def test_b5_vorgerechnetes_lambda1_gitter_chaos_und_fenster():
    """22 Werte ρ: λ₁ > 0,05 in 15 (ρ = 25, 28, 30, 35, 40, 45, 50, 60, 70, 80, 90, 110, 120, 130, 140); nicht bei ρ = 5, 10, 15, 20 (Fixpunkte) und 100, 150, 160."""
    chaos = sorted(r for r, v in LOR.items() if v["lam"][0] > 0.05)
    assert chaos == [25, 28, 30, 35, 40, 45, 50, 60, 70, 80, 90, 110, 120, 130, 140]
    ruhig = sorted(set(LOR) - set(chaos))
    assert ruhig == [5, 10, 15, 20, 100, 150, 160]
    assert LOR[100]["lam"][0] == pytest.approx(-0.0025, abs=2e-3) and LOR[150]["lam"][0] == pytest.approx(0.0014, abs=2e-3) and LOR[160]["lam"][0] == pytest.approx(0.0013, abs=2e-3)
    assert all(abs(LOR[r]["lam"][1]) < 0.003 for r in chaos)           # Exponent in Flussrichtung
    assert LOR[40]["lam"][0] == pytest.approx(1.14, abs=0.02) and LOR[60]["lam"][0] == pytest.approx(1.41, abs=0.02)


def test_b5_kein_chaos_bei_100_150_160_in_100_von_100_bahnen():
    """Absolutaussage mit Gegenprobe (ORACLE §6): 100 Bahnen mit Einschwingzeit 50 und Messdauer 100 (dt = 0,01): in 300 von 300 Bahnen λ₁ ≤ 0,03 (gemessen höchstens 0,019)."""
    for rho in (100.0, 150.0, 160.0):
        sp = L.spektrum(rho, 2026, n_starts=100, t_burn=50.0, t_mess=100.0, dt=0.01, t_qr=15.0)
        assert float(sp["je_start"][:, 0].max()) <= 0.03, rho


def test_b5_live_lambda1_bei_rho_28_ueber_5_zufalls_seeds():
    """README: Live-Messung (160 Bahnen zugleich): λ₁(28) = 0,896 bis 0,917 über die Zufalls-Seeds 1 bis 5 (Mittel 0,906, Streuung 0,010)."""
    lam = [L.spektrum(28.0, s)["lam"][0] for s in (1, 2, 3, 4, 5)]
    assert 0.885 < min(lam) and max(lam) < 0.925
    assert float(np.mean(lam)) == pytest.approx(0.906, abs=0.01)


def test_b5_vorhersagehorizont_bei_rho_28_ueber_5_zufalls_seeds():
    """README: je 30 Startwerte auf dem Attraktor, Anfangsabstand 1e-9: Mittel 21,3 bis 22,4 je Zufalls-Seed, einzelne Paare 16 bis 29; ln(10⁹)/0,90 = 23,0."""
    mittel, alle = [], []
    for seed in (1, 2, 3, 35, 36):
        z = L.divergenz(28.0, seed)["zeiten"]
        assert not np.isnan(z).any()
        mittel.append(float(np.mean(z)))
        alle.extend(z.tolist())
    assert 21.0 < min(mittel) and max(mittel) < 22.8
    assert 15.5 < min(alle) and max(alle) < 29.0
    assert math.log(1e9) / 0.90 == pytest.approx(23.0, abs=0.05)


def test_b5_zwei_bahnen_bei_rho_20_gehen_nicht_auseinander():
    z = L.divergenz(20.0, 1)["zeiten"]
    assert np.isnan(z).all() and len(z) == 30


# --- B6: App-Texte ---------------------------------------------------------------------------------------------------------------------


def test_b6_live_lorenz_hat_summe_gleich_spur_auf_1e_3_bei_den_acht_reglerwerten_der_readme():
    for rho in (10.0, 20.0, 28.0, 40.0, 60.0, 100.0, 150.0, 160.0):
        assert L.spektrum(rho, 1, n_starts=40)["summe"] == pytest.approx(L.spur(), abs=2e-3), rho


def test_b5_kurze_einschwingzeit_taeuscht_bei_rho_100():
    """README: Einschwingzeit 30: λ₁ live bei ρ = 100 in 3 Zufalls-Seeds (35, 36, 37) +0,016 bis +0,042; Einschwingzeit 60: +0,011 bis +0,016."""
    kurz = [float(L.spektrum(100.0, s, t_burn=30.0)["lam"][0]) for s in (35, 36, 37)]
    lang = [float(L.spektrum(100.0, s, t_burn=60.0)["lam"][0]) for s in (35, 36, 37)]
    assert 0.012 < min(kurz) and 0.030 < max(kurz) < 0.050
    assert 0.008 < min(lang) and max(lang) < 0.020


def test_b5_vorhersagehorizont_bei_rho_100_nur_ein_paar_von_150():
    erreicht = []
    for seed in (1, 2, 3, 35, 36):
        z = L.divergenz(100.0, seed)["zeiten"]
        erreicht.extend(z[~np.isnan(z)].tolist())
    assert len(erreicht) <= 3                         # gemessen 1 von 150 Paaren
