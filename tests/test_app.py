"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def _texte(at) -> str:
    return " ".join(e.value for kind in ("success", "info", "warning") for e in getattr(at, kind))


def test_jedes_preset_zeigt_seine_hauptaussage_in_der_app():
    """Preset-Fazits sind Behauptungen: je Preset die erwartete Zeile im sichtbaren Verdikt (Kernfrage und Experimente)."""
    import vzh_constants as C

    erwartet = {
        "Unter der Grenze (Voreinstellung)": ["unter π/2 = 1,571 abklingend", "als instabil, obwohl sie stabil ist", "keine Dauerschwingung"],
        "Von Hand": ["x(2) = −0,500", "fällt mit Rate 0,318 je τ"],
        "Hinter der Grenze": ["über π/2 = 1,571 anwachsend", "Dauerschwingung mit Spannweite 2,101"],
        "Mackey–Glass": ["τ = 17,0: λ = +0,00"],
        "Lorenz": ["ρ = 28: λ₁ = +0,9"],
    }
    assert set(erwartet) == set(C.PRESETS)
    for name, zeilen in erwartet.items():
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        text = _texte(at)
        for zeile in zeilen:
            assert zeile in text, (name, zeile)


def test_permalink_uebernimmt_werte_und_begrenzt_sie():
    at = AppTest.from_file(APP_PATH)
    at.query_params["at"] = "9.9"                  # über dem Maximum: wird auf 3,0 begrenzt
    at.query_params["verfahren"] = "Heun"
    at.query_params["m"] = "7"
    at.query_params["rho"] = "28"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="at_slider").value == 3.0
    assert at.radio(key="verfahren_radio").value == "Heun"
    assert at.select_slider(key="m_select").value == 7
    assert at.slider(key="rho_slider").value == 28


def test_permalink_ignoriert_unerlaubte_auswahlwerte():
    at = AppTest.from_file(APP_PATH)
    at.query_params["m"] = "6"                      # keine Option
    at.query_params["verfahren"] = "RK4"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.select_slider(key="m_select").value == 5 and at.radio(key="verfahren_radio").value == "Euler"
