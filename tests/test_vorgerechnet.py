"""Die eingefrorenen Messreihen (vzh_vorgerechnet.json, erzeugt von tools/vorrechnen.py): Aufbau, Kennzeichnung und Übereinstimmung mit der Live-Rechnung an je einem Punkt."""

import json
from pathlib import Path

import pytest

import vzh_algorithm as A
import vzh_evaluation as E
import vzh_lorenz as L

ROOT = Path(__file__).resolve().parent.parent
DATEN = json.loads((ROOT / "vzh_vorgerechnet.json").read_text(encoding="utf-8"))


def test_aufbau_und_parameter_der_vorgerechneten_reihen():
    assert DATEN["schnell"] is False
    lor, mg = DATEN["lorenz"], DATEN["mackey_glass"]
    assert [r["rho"] for r in lor["rows"]] == [5, 10, 15, 20, 25, 28, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    assert (lor["n_start"], lor["t_burn"], lor["t_qr"], lor["t_mess"], lor["dt"]) == (100, 50.0, 15.0, 100.0, 0.005)
    assert [r["tau"] for r in mg["rows"]] == list(range(5, 31)) and mg["seeds"] == [11, 12, 13] and mg["t_mess"] == 40000.0
    for r in mg["rows"]:
        assert r["min"] <= r["lam"] <= r["max"]


def test_lade_vorgerechnet_und_fehlende_datei():
    assert E.lade_vorgerechnet()["lorenz"]["seed"] == 2026
    assert E.lade_vorgerechnet(ROOT / "gibt_es_nicht.json") is None


def test_vorgerechnetes_lambda1_gleich_live_bei_rho_28_innerhalb_der_streuung():
    frozen = next(r for r in DATEN["lorenz"]["rows"] if r["rho"] == 28)["lam"][0]
    live = L.spektrum(28.0, 7)["lam"][0]
    assert abs(frozen - live) < 0.03


def test_vorgerechnetes_mackey_glass_gleich_live_bei_tau_17_und_24():
    for tau in (17, 24):
        frozen = next(r for r in DATEN["mackey_glass"]["rows"] if r["tau"] == tau)["lam"]
        live = A.mg_lyapunov(float(tau), 60000.0, 5000.0, 21, "zweibahn")["lambda"]
        assert abs(frozen - live) < 6e-4, tau
