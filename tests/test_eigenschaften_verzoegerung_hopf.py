"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für die Verfahren gelten müssen, ohne dass man das richtige Ergebnis kennt.

Simulationen: gleicher Zufalls-Seed gleiche Ausgabe; Fortsetzbarkeit (ein längerer Lauf beginnt mit dem kürzeren); Skalieren der Vorgeschichte der **linearen** Gleichung skaliert die
Lösung; die Lorenz-Symmetrie (x, y) -> (-x, -y); Positivität der verzögerten Logistik; Heun nähert die Gleichung mit wachsendem m monoton an."""

import numpy as np
import pytest

import vzh_algorithm as A
import vzh_lorenz as L


@pytest.mark.parametrize("method", ["heun", "euler"])
def test_fortsetzbarkeit_langer_lauf_beginnt_mit_dem_kurzen(method):
    kurz = A.linear_dde(1.45, 50, 10, method)
    lang = A.linear_dde(1.45, 50, 25, method)
    assert np.array_equal(lang[: len(kurz)], kurz)


def test_fortsetzbarkeit_hutchinson():
    assert np.array_equal(A.hutchinson(1.8, 100, 80)[: 100 * 30 + 1], A.hutchinson(1.8, 100, 30))


@pytest.mark.parametrize("c", [0.3, 2.7])
def test_skalieren_der_vorgeschichte_skaliert_die_loesung_der_linearen_gleichung(c):
    assert np.allclose(A.linear_dde(1.45, 40, 20, hist=c), c * A.linear_dde(1.45, 40, 20, hist=1.0), rtol=1e-12, atol=1e-12)


def test_skalieren_gilt_nicht_fuer_hutchinson():
    """Gegenprobe: die verzögerte Logistik ist nichtlinear, also ändert die Skalierung der Vorgeschichte die Bahn (sonst wäre die Eigenschaft oben leer)."""
    a = A.hutchinson(1.8, 40, 20, x0=0.5)
    b = A.hutchinson(1.8, 40, 20, x0=1.0)
    assert not np.allclose(b, 2.0 * a, rtol=1e-3)


@pytest.mark.parametrize("r", [1.4, 2.2, 3.0])
def test_hutchinson_bleibt_positiv_fuer_positive_vorgeschichte(r):
    rng = np.random.default_rng(3)
    hist = rng.uniform(0.2, 1.8, size=101)
    assert np.all(A.hutchinson(r, 100, 300, x0=hist) > 0)


def test_heun_naehert_die_gleichung_mit_wachsendem_m_monoton_an():
    exakt = A.rechtester_wurzel(1.45).real
    fehler = [abs(A.diskrete_wachstumsrate(1.45, m, "heun") - exakt) for m in (2, 5, 10, 20, 50, 100)]
    assert fehler == sorted(fehler, reverse=True) and fehler[-1] < 1e-4


def test_euler_grenze_waechst_mit_m_und_liegt_unter_pi_halbe():
    werte = [A.euler_grenze(m) for m in range(1, 60)]
    assert werte == sorted(werte) and max(werte) < np.pi / 2


def test_lorenz_symmetrie_bleibt_ueber_viele_schritte_erhalten():
    s = np.array([3.0, -2.0, 15.0])
    spiegel = np.array([-3.0, 2.0, 15.0])
    for _ in range(100):
        s, spiegel = L.rk4(s, 28.0, 0.01), L.rk4(spiegel, 28.0, 0.01)
    assert np.allclose(spiegel, [-s[0], -s[1], s[2]], atol=1e-9)


def test_gleicher_zufalls_seed_gleiche_ausgabe_mackey_glass_und_lorenz():
    a, b = A.mg_bahn(17.0, 4, 500.0, 50.0), A.mg_bahn(17.0, 4, 500.0, 50.0)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], A.mg_bahn(17.0, 5, 500.0, 50.0)[1])
    p, q = L.divergenz(28.0, 4, n=4, t_max=5.0), L.divergenz(28.0, 4, n=4, t_max=5.0)
    assert np.array_equal(p["pfad"], q["pfad"])
