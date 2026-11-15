"""Rechnet die schweren Messreihen vor und schreibt sie nach vzh_vorgerechnet.json (die App liest nur diese Datei).

Aufruf (im Repo-Ordner, Python des Test- oder Laufzeit-venv): python tools/vorrechnen.py [--schnell]
Dauer ohne --schnell auf einem Büro-Rechner etwa 10 Minuten (Lorenz-Gitter in rho dominiert). Das Ergebnis ist deterministisch: die Zufalls-Seeds stehen unten.

Inhalt:
* "lorenz": Lyapunov-Spektrum für rho = 5, 10, ..., 160 und rho = 28 (100 gleichzeitig gerechnete Bahnen, je 100 Zeiteinheiten Messdauer, RK4 dt = 0,005);
  dazu Zeitschritt und Messdauer bei rho = 28 als Streuungsprobe.
* "mackey_glass": größter Exponent (Variationsgleichung, drei Zufalls-Seeds, 40 000 Zeiteinheiten je Lauf) für tau = 5, 6, ..., 30. Eine Messung ohne Literaturvergleich.
Beide Reihen sind in der App als „vorgerechnet“ gekennzeichnet."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import vzh_algorithm as A  # noqa: E402
import vzh_lorenz as L  # noqa: E402

AUSGABE = ROOT / "vzh_vorgerechnet.json"
LOR_SEED = 2026
MG_SEEDS = (11, 12, 13)


def lorenz_reihe(schnell: bool) -> dict:
    rhos = [5, 10, 15, 20, 25, 28, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    if schnell:
        rhos = [10, 20, 28, 40, 100, 160]
    n_start, burn, qr, mess, dt = (40, 40.0, 10.0, 40.0, 0.01) if schnell else (100, 50.0, 15.0, 100.0, 0.005)
    rows = []
    for rho in rhos:
        t0 = time.time()
        r = L.spektrum(float(rho), LOR_SEED, n_starts=n_start, t_burn=burn, t_mess=mess, dt=dt, t_qr=qr)
        rows.append({"rho": rho, "lam": [round(float(v), 4) for v in r["lam"]], "sem": [round(float(v), 4) for v in r["sem"]], "summe": round(r["summe"], 4)})
        print(f"  rho = {rho}: {rows[-1]['lam']} (Summe {rows[-1]['summe']}) in {time.time() - t0:.1f} s", flush=True)
    schritt = []
    for dt_x in (0.01, 0.005, 0.0025):
        r = L.spektrum(28.0, LOR_SEED + 1, n_starts=n_start, t_burn=burn, t_mess=mess, dt=dt_x, t_qr=qr)
        schritt.append({"dt": dt_x, "lam1": round(float(r["lam"][0]), 4), "sem": round(float(r["sem"][0]), 4)})
        print(f"  rho = 28, dt = {dt_x}: lambda_1 = {schritt[-1]['lam1']} +- {schritt[-1]['sem']}", flush=True)
    return {"rows": rows, "zeitschritt_rho28": schritt, "n_start": n_start, "t_burn": burn, "t_qr": qr, "t_mess": mess, "dt": dt, "seed": LOR_SEED}


def mg_reihe(schnell: bool) -> dict:
    taus = list(range(5, 31))
    t_mess = 40000.0
    if schnell:
        taus, t_mess = [6, 12, 17, 24], 12000.0
    rows = []
    for tau in taus:
        werte = [A.mg_lyapunov(float(tau), t_mess, 5000.0, s, "tangente")["lambda"] for s in MG_SEEDS]
        rows.append({"tau": tau, "lam": round(float(np.mean(werte)), 5), "min": round(float(min(werte)), 5), "max": round(float(max(werte)), 5)})
        print(f"  tau = {tau}: {rows[-1]}", flush=True)
    return {"rows": rows, "seeds": list(MG_SEEDS), "t_mess": t_mess, "t_burn": 5000.0, "h": 0.1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schnell", action="store_true", help="kleine Fassung zum Ausprobieren (ändert die Zahlen; nur zum Testen des Skripts)")
    args = ap.parse_args()
    t0 = time.time()
    print("Lorenz ...", flush=True)
    lor = lorenz_reihe(args.schnell)
    print("Mackey-Glass ...", flush=True)
    mg = mg_reihe(args.schnell)
    daten = {"schnell": args.schnell, "lorenz": lor, "mackey_glass": mg, "dauer_s": round(time.time() - t0, 1)}
    ziel = AUSGABE if not args.schnell else ROOT / "vzh_vorgerechnet_schnell.json"
    ziel.write_text(json.dumps(daten, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"geschrieben: {ziel.name} nach {daten['dauer_s']} s")


if __name__ == "__main__":
    main()
