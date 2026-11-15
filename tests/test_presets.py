"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Begrenzung."""

import vzh_constants as C
import vzh_evaluation as ev
import vzh_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            value = preset[key]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Unter der Grenze (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_die_presets_unterscheiden_sich_paarweise():
    """Sonst wäre die Hervorhebung des aktiven Presets mehrdeutig."""
    werte = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(werte)) == len(werte)


def test_keine_voreinstellung_ist_eine_zweierpotenz_ausser_dem_handwert():
    """Rechner-Artefakt (Zweierpotenzen): Voreinstellungen und Presets meiden sie; Ausnahme ist a·τ = 1,0 im Preset „Von Hand“ (der Handwert des Beispiels)."""
    def zweierpotenz(v):
        return v > 0 and float(v).is_integer() and (int(v) & (int(v) - 1)) == 0

    for name, preset in C.PRESETS.items():
        for key in ("at", "mg_tau", "rho"):
            if name == "Von Hand" and key == "at":
                continue
            assert not zweierpotenz(preset[key]), (name, key, preset[key])
    assert not zweierpotenz(C.DEFAULT_AT) and not zweierpotenz(C.DEFAULT_MG_TAU) and not zweierpotenz(C.DEFAULT_RHO)
    assert not zweierpotenz(C.DEFAULT_M) and C.DEFAULT_M == 5
