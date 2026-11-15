"""Eigener Zufallsgenerator (SplitMix64): gleiche Zufalls-Seeds geben auf jeder Python-Version und Plattform dieselben Folgen.

Herkunft: kopiert aus saal-abstand-demo/sab_rng.py (Stand 2026-11-14), nur Klassenkopf und Docstring gekürzt; die Demo importiert nichts aus
anderen Demos. Python-`random` und `hash()` sind für feste Zufalls-Seeds über Versionen und Prozesse nicht zugesichert."""

import math

MASK = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def random(self) -> float:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        return (self.next_u64() >> 11) / float(1 << 53)

    def normal(self) -> float:
        """Standardnormalverteilt (Box-Muller)."""
        u1 = max(self.random(), 1e-12)
        u2 = self.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
