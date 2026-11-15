"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Ergebniszeilen, nicht Zufalls-Seeds. Die Zahlen im Text sind in deutscher Schreibweise (Komma, „−“)."""

import math

import pytest

import vzh_constants as C
import vzh_evaluation as E


def _hand(a, sigma, omega, werte=(1.0, 0.0, -0.5, -1.0 / 6.0, 5.0 / 24.0), stuecke=4):
    return E.Hand(a, stuecke, tuple(werte), tuple(werte), 0.0, sigma, omega, (0.0, 1.0), (1.0, 0.0))


def test_de_formatiert_komma_und_minuszeichen():
    assert E.de(0.3181, 3) == "0,318" and E.de(-0.5, 3) == "−0,500" and E.de(0.0053, 4, plus=True) == "+0,0053"


def test_verdikt_hand_abklingend_mit_stuetzwerten_und_rate():
    kind, text = E.verdict_hand(_hand(1.0, -0.31813, 1.33724))
    assert kind == "success"
    assert "x(1) = 0,000, x(2) = −0,500, x(3) = −0,167" in text and "fällt mit Rate 0,318 je τ" in text
    assert "(a·τ = 1,00)" in text and "über 1/e = 0,368" in text and "Periode der Schwingung ist 4,70 τ" in text


def test_verdikt_hand_monoton_unter_ein_durch_e():
    kind, text = E.verdict_hand(_hand(0.3, -0.489, 0.0))
    assert kind == "success" and "Der Verlauf ist monoton, denn a·τ = 0,30 liegt unter 1/e = 0,368" in text


def test_verdikt_hand_schwelle_ein_durch_e_ist_inklusive():
    kind, text = E.verdict_hand(_hand(1.0 / math.e, -1.0, 0.0))
    assert "monoton" in text and "wechselt" not in text


def test_verdikt_hand_wachsend_und_an_der_grenze():
    kind, text = E.verdict_hand(_hand(1.8, 0.1, 1.6))
    assert kind == "warning" and "wächst mit Rate 0,100 je τ" in text
    kind, text = E.verdict_hand(_hand(math.pi / 2, 1e-17, math.pi / 2))
    assert kind == "info" and "an der Grenze π/2 = 1,571" in text and "Periode 4 τ" in text
    # Schwelle der Grenze: 0,005 um π/2
    assert E.verdict_hand(_hand(math.pi / 2 + 0.004, 0.0003, 1.57))[0] == "info"
    assert E.verdict_hand(_hand(math.pi / 2 + 0.006, 0.0003, 1.57))[0] == "warning"


def _kenn(a, re, im, gemessen):
    return E.Kennlinie(a, (complex(re, im),), complex(re, im), gemessen, (0.1,), (0.0,))


def test_verdikt_kennlinie_drei_lagen():
    kind, text = E.verdict_kennlinie(_kenn(1.45, -0.0568, 1.53, -0.0568))
    assert kind == "success" and "a·τ = 1,45: größtes Re s = −0,0568 (Simulation: −0,0568)" in text and "unter π/2 = 1,571 abklingend, Periode 4,11 τ" in text
    kind, text = E.verdict_kennlinie(_kenn(1.8, 0.1, 1.6, 0.1))
    assert kind == "warning" and "über π/2 = 1,571 anwachsend" in text
    kind, text = E.verdict_kennlinie(_kenn(1.5708, 0.0, math.pi / 2, 0.0))
    assert kind == "info" and "an der Grenze π/2 = 1,571: Re s ≈ 0, die Periode ist 4 τ" in text


def test_verdikt_kennlinie_reelle_wurzel_hat_keine_schwingung():
    kind, text = E.verdict_kennlinie(_kenn(0.2, -0.26, 0.0, -0.26))
    assert kind == "success" and "keine Schwingung (reelle Wurzel)" in text


def _hut(r, span, period, zuvor=float("nan")):
    return E.Hut(r, span, period, 1.0, (0.0,), (1.0,), (1.0,), (1.0,), zuvor)


_FIT = E.HutFit((1.58, 1.60, 1.62, 1.65, 1.70), (0.44, 0.79, 1.01, 1.28, 1.61), 19.894, -31.216, 1.5691)


def test_verdikt_hutchinson_mit_und_ohne_schwingung():
    kind, text = E.verdict_hutchinson(_hut(1.45, 0.0, float("nan")), _FIT)
    assert kind == "info" and "keine Dauerschwingung" in text and "ist 0,000" in text and "schneidet die Nullachse bei 1,569, gegen π/2 = 1,571" in text
    kind, text = E.verdict_hutchinson(_hut(1.65, 1.2757, 4.0571), _FIT)
    assert kind == "warning" and "Spannweite 1,276 und Periode 4,057 τ" in text and "bei r·τ = 1,58 bis 1,70" in text


def test_verdikt_hutchinson_nahe_der_grenze_ist_noch_nicht_entschieden():
    """Fund der unabhängigen Prüfung: bei r·τ = 1,57 (unter π/2) meldete die App „Dauerschwingung mit Spannweite 0,083“, obwohl es eine Transiente ist (Re s = −0,00036).
    Ändert sich die Spannweite von einem Fenster der letzten 300 Verzögerungen zum vorigen um mehr als 2 %, entscheidet das Verdikt nicht."""
    kind, text = E.verdict_hutchinson(_hut(1.57, 0.0833, 4.001, zuvor=0.0977), _FIT)
    assert kind == "info" and "noch nicht entschieden" in text and "Dauerschwingung mit" not in text and "0,098 auf 0,083" in text and "kein Beleg für eine Dauerschwingung" in text
    assert E.verdict_hutchinson(_hut(1.58, 0.444, 4.006, zuvor=0.444), _FIT)[0] == "warning"
    assert E.verdict_hutchinson(_hut(1.58, 0.444, 4.006, zuvor=0.444 * 1.019), _FIT)[0] == "warning"            # Grenze 2 %
    assert E.verdict_hutchinson(_hut(1.58, 0.444, 4.006, zuvor=0.444 * 1.03), _FIT)[0] == "info"
    assert E.verdict_hutchinson(_hut(1.58, 0.444, 4.006), _FIT)[0] == "warning"                                  # Fenster davor unbekannt


def test_verdikt_hutchinson_mit_dem_echten_kern_bei_1_57_und_bei_1_58():
    """Echte Rechnung (Heun, 400 Schritte je τ, 2000 Verzögerungen): bei 1,57 ist die Spannweite 0,083 gegen 0,098 im Fenster davor, bei 1,58 sind beide 0,444."""
    h = E.analyse_hutchinson(1.57)
    assert h.span == pytest.approx(0.0833, abs=5e-4) and h.span_zuvor == pytest.approx(0.0977, abs=5e-4) and not h.eingeschwungen
    kind, text = E.verdict_hutchinson(h, E.analyse_hutchinson_fit())
    assert kind == "info" and "noch nicht entschieden" in text and "Dauerschwingung mit" not in text
    h = E.analyse_hutchinson(1.58)
    assert h.eingeschwungen and h.span == pytest.approx(0.444, abs=5e-4)
    assert E.verdict_hutchinson(h, E.analyse_hutchinson_fit())[0] == "warning"


def test_verdikt_hutchinson_schwelle_der_spannweite():
    assert E.verdict_hutchinson(_hut(1.5, C.HUT_SPAN_NULL * 0.99, float("nan")), _FIT)[0] == "info"
    assert E.verdict_hutchinson(_hut(1.5, C.HUT_SPAN_NULL * 1.01, 4.0), _FIT)[0] == "warning"


def _zeit(exakt, diskret, m=5, verfahren="Euler", a=1.45):
    return E.Zeitschritt(a, m, verfahren, exakt, diskret, diskret, 1.4231, (m,), (diskret,), (diskret,))


def test_verdikt_zeitschritt_instabil_obwohl_stabil():
    kind, text = E.verdict_zeitschritt(_zeit(-0.0568, 0.0121))
    assert kind == "warning" and "Bei m = 5 Schritten je τ rechnet Euler die Gleichung bei a·τ = 1,45 als instabil, obwohl sie stabil ist" in text
    assert "Gleichung −0,0568, Euler mit m = 5 Schritten je τ +0,0121" in text and "= 1,423" in text


def test_verdikt_zeitschritt_stabil_obwohl_instabil_und_gleiches_vorzeichen():
    kind, text = E.verdict_zeitschritt(_zeit(0.17, -0.01, m=1, a=2.2))
    assert kind == "warning" and "als stabil, obwohl sie instabil ist" in text
    kind, text = E.verdict_zeitschritt(_zeit(-0.0568, -0.0412, m=10, verfahren="Heun"))
    assert kind == "success" and "Vorzeichen der Rate bei a·τ = 1,45 mit der Gleichung überein (Abweichung 0,0156)" in text and "Heun mit m = 10" in text


def _mg(tau, lam, lam_t=None):
    return E.MackeyGlass(tau, 1, lam, lam if lam_t is None else lam_t, 60000.0, 0.4, 1.3, 0.1, 4.7082, 0.04, (0.0,), (1.0,), (1.0,), (1.0,))


def test_verdikt_mackey_glass_chaos_bei_17_ohne_referenzzusatz():
    kind, text = E.verdict_mackey_glass(_mg(17.0, 0.0053, 0.0052))
    assert kind == "warning" and "τ = 17,0: λ = +0,0053 je Zeiteinheit (Zweibahn; Variationsgleichung +0,0052)" in text
    assert "Vorhersagezeit 1/λ rund 189 Zeiteinheiten" in text and "nicht gegen eine Referenz" not in text and "τ_c = 4,708" in text


def test_verdikt_mackey_glass_anderes_tau_hat_den_referenzzusatz():
    kind, text = E.verdict_mackey_glass(_mg(24.0, 0.0104))
    assert kind == "warning" and "Bei diesem τ nicht gegen eine Referenz geprüft." in text


def test_verdikt_mackey_glass_schwelle_und_periodischer_fall():
    kind, text = E.verdict_mackey_glass(_mg(12.0, 0.00003))
    assert kind == "info" and "kein positiver Exponent über der Schwelle 0,0010 messbar (λ = +0,0000 je Zeiteinheit)" in text
    assert "nach dieser Messung periodisch, ein Gleichgewicht oder ein Exponent unter der Schwelle" in text and "die Bahn ist periodisch oder ein Gleichgewicht" not in text
    assert E.verdict_mackey_glass(_mg(17.0, C.MG_LAMBDA_NULL * 1.01))[0] == "warning"
    assert E.verdict_mackey_glass(_mg(17.0, C.MG_LAMBDA_NULL * 0.99))[0] == "info"


def _lor(rho, lam1, sem1=0.006, mittel=21.1, nicht=0):
    return E.Lorenz(rho, 1, (lam1, 0.0, -14.57), (sem1, 0.0, 0.0), -13.6666, -41 / 3, mittel, 17.9, 25.1, nicht, 30, 1.0, (0.0,), (0.0,), (0.0,), (1e-9,))


def test_verdikt_lorenz_chaos_bei_28_mit_band_und_horizont():
    kind, text = E.verdict_lorenz(_lor(28, 0.90))
    assert kind == "warning" and "ρ = 28: λ₁ = +0,90 ± 0,01 (Band der Vormessung 0,89 bis 0,91)" in text
    assert "nach im Mittel 21,1 Zeiteinheiten (17,9 bis 25,1 bei 30 Startwerten)" in text
    assert "Summe der drei Exponenten −13,667 gegen die Spur −13,667" in text


def test_verdikt_lorenz_chaos_bei_anderem_rho_ohne_band_und_unvollstaendiger_horizont():
    kind, text = E.verdict_lorenz(_lor(60, 1.40, mittel=14.2, nicht=4))
    assert kind == "warning" and "Band der Vormessung" not in text and "Von 30 Paaren im Anfangsabstand 1e-9 erreichen 26 den Abstand 1 innerhalb von 40 Zeiteinheiten" in text


def test_verdikt_lorenz_kein_chaos_und_lage_zu_rho_h():
    kind, text = E.verdict_lorenz(_lor(20, -0.14, mittel=float("nan"), nicht=30))
    assert kind == "info" and "kein Chaos (λ₁ = −0,14 ± 0,01, Schwelle 0,05)" in text and "ρ liegt unter ρ_H = 24,737: die Fixpunkte sind linear stabil" in text
    kind, text = E.verdict_lorenz(_lor(100, 0.0, mittel=float("nan"), nicht=30))
    assert kind == "info" and "ρ liegt über ρ_H = 24,737: die Fixpunkte sind linear instabil" in text


def test_verdikt_lorenz_schwelle_des_chaos():
    assert E.verdict_lorenz(_lor(100, C.LOR_LAMBDA_CHAOS * 1.01))[0] == "warning"
    assert E.verdict_lorenz(_lor(100, C.LOR_LAMBDA_CHAOS * 0.99, mittel=float("nan"), nicht=30))[0] == "info"


def test_jedes_preset_nennt_seine_hauptaussage_im_verdikt():
    """Preset-Fazits sind Behauptungen (DEMO-PLAYBOOK §11a Punkt 7): jedes Preset durchlaufen und die erwartete Verdikt-Zeile prüfen."""
    unter = E.Settings(**C.PRESETS["Unter der Grenze (Voreinstellung)"])
    hinter = E.Settings(**C.PRESETS["Hinter der Grenze"])
    # Unter der Grenze: Gleichung klingt ab, Hutchinson schwingt nicht, Euler mit m = 5 rechnet instabil
    assert E.verdict_kennlinie(E.analyse_kennlinie(unter.at))[0] == "success"
    assert E.verdict_hutchinson(E.analyse_hutchinson(unter.at), E.analyse_hutchinson_fit())[0] == "info"
    assert E.verdict_zeitschritt(E.analyse_zeitschritt(unter.at, unter.m, unter.verfahren))[0] == "warning"
    # Hinter der Grenze: Gleichung wächst, Hutchinson schwingt
    assert E.verdict_kennlinie(E.analyse_kennlinie(hinter.at))[0] == "warning"
    assert E.verdict_hutchinson(E.analyse_hutchinson(hinter.at), E.analyse_hutchinson_fit())[0] == "warning"
    # Mackey–Glass: bei τ = 17 ein positiver Exponent, bei der Voreinstellung τ = 12 keiner; Lorenz: ρ = 28 chaotisch, ρ = 20 nicht
    mg = E.Settings(**C.PRESETS["Mackey–Glass"])
    assert E.verdict_mackey_glass(E.analyse_mackey_glass(mg.mg_tau, mg.seed))[0] == "warning"
    assert E.verdict_mackey_glass(E.analyse_mackey_glass(unter.mg_tau, unter.seed))[0] == "info"
    lz = E.Settings(**C.PRESETS["Lorenz"])
    assert E.verdict_lorenz(E.analyse_lorenz(lz.rho, lz.seed))[0] == "warning"
    assert E.verdict_lorenz(E.analyse_lorenz(unter.rho, unter.seed))[0] == "info"
    # Von Hand: die Zeile nennt x(2) = −0,500
    hand = E.Settings(**C.PRESETS["Von Hand"])
    assert "x(2) = −0,500" in E.verdict_hand(E.analyse_hand(hand.at, hand.stuecke))[1]
