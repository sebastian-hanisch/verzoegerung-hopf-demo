"""Gemeinsame Fixtures: das Handbeispiel, an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand():
    """Lineare Gleichung x' = -x(t - 1), Vorgeschichte 1 auf [-1, 0] (Handwert a·τ = 1, ein Handwert, kein Voreinstellungswert der App).

    Von Hand nach der Methode der Schritte: auf [0, 1] ist x(t) = 1 - t, also x(1) = 0; auf [1, 2] ist x(t) = (t - 1)(t - 3)/2, also x(2) = -1/2;
    auf [2, 3] folgt x(3) = -1/6, dann x(4) = 5/24."""
    return {"a": 1.0, "werte": [1.0, 0.0, -1.0 / 2.0, -1.0 / 6.0, 5.0 / 24.0]}
