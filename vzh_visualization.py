"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.

Deutsche Zahlenschreibweise in den Achsen (`separators=",."`)."""

import math

import numpy as np
import plotly.graph_objects as go

import vzh_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", separators=",.",
                      legend=dict(orientation="h", y=-0.22) if legend else dict(visible=False), showlegend=legend)
    return lock_axes(fig)


def build_stuecke(h):
    """Experiment 1: die exakten Stücke der Methode der Schritte, die Stützwerte an ganzen τ und die Hüllkurve ±e^(Re s·t) (nur die Steigung zählt, nicht die Höhe)."""
    fig = go.Figure()
    ts, xs = np.array(h.t), np.array(h.x)
    for k in range(h.stuecke):
        sel = (ts >= k) & (ts <= k + 1)
        fig.add_trace(go.Scatter(x=ts[sel], y=xs[sel], mode="lines", line=dict(color=C.COLOR_MAIN if k % 2 == 0 else C.COLOR_LINE, width=3),
                                 name=f"Abschnitt {k + 1}", hovertemplate="t = %{x:.2f}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(range(h.stuecke + 1)), y=list(h.werte), mode="markers", marker=dict(size=9, color="#111", symbol="diamond"),
                             name="Stützwerte x(0), x(1), …", hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"))
    tt = np.linspace(0, h.stuecke, 120)
    env = np.exp(h.sigma * tt)
    for sign in (1, -1):
        fig.add_trace(go.Scatter(x=tt, y=sign * env, mode="lines", line=dict(color=C.COLOR_GREY, dash="dot", width=1.5), name="Hüllkurve e^(Re s·t)",
                                 showlegend=sign == 1, hoverinfo="skip"))
    fig.update_xaxes(title_text="Zeit t (in Verzögerungen τ)")
    fig.update_yaxes(title_text="x(t)")
    return _base(fig)


def build_wurzeln(k):
    """Experiment 2, links: die Wurzeln s_k der Kennlinie in der komplexen Ebene; die rechteste ist markiert."""
    fig = go.Figure()
    w = np.array(k.wurzeln)
    fig.add_trace(go.Scatter(x=w.real, y=w.imag, mode="markers", marker=dict(size=9, color=C.COLOR_LINE), name="Wurzeln s_k, k = −2 … 2",
                             hovertemplate="Re s = %{x:.4f}, Im s = %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[k.rechts.real], y=[k.rechts.imag], mode="markers", marker=dict(size=14, color=C.COLOR_ALT, symbol="diamond-open", line=dict(width=3)),
                             name="rechteste Wurzel", hovertemplate="Re s = %{x:.4f}, Im s = %{y:.4f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=C.COLOR_GREY, dash="dash"))
    fig.update_xaxes(title_text="Re s")
    fig.update_yaxes(title_text="Im s")
    return _base(fig)


def build_kurve_wachstum(k):
    """Experiment 2, rechts: größtes Re s gegen a·τ mit π/2 und der aus der Simulation gemessenen Rate beim gewählten Wert."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(k.kurve_a), y=list(k.kurve_re), mode="lines", line=dict(color=C.COLOR_LINE, width=3), name="Re W₀(−a·τ) / τ",
                             hovertemplate="a·τ = %{x:.2f}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[k.a], y=[k.gemessen], mode="markers", marker=dict(size=12, color=C.COLOR_ALT, symbol="x", line=dict(width=3)),
                             name="gemessen (Simulation)", hovertemplate="a·τ = %{x:.2f}: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C.COLOR_GREY))
    fig.add_vline(x=math.pi / 2, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text="π/2", annotation_position="top")
    fig.update_xaxes(title_text="a·τ")
    fig.update_yaxes(title_text="Wachstumsrate je Verzögerung")
    return _base(fig)


def build_hut_zeit(h):
    fig = go.Figure(go.Scatter(x=list(h.t), y=list(h.x), mode="lines", line=dict(color=C.COLOR_MAIN, width=2), hovertemplate="t = %{x:.2f}: %{y:.4f}<extra></extra>",
                               name="x(t)"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_GREY, dash="dot"))
    fig.update_xaxes(title_text="Zeit der letzten 40 Verzögerungen")
    fig.update_yaxes(title_text="x(t)")
    return _base(fig, legend=False)


def build_phase(x_zuvor, x_jetzt, xtitle, ytitle, farbe=C.COLOR_LINE):
    """Phasenbild x(t) gegen die um die Verzögerung zurückliegende x(t − τ)."""
    fig = go.Figure(go.Scatter(x=list(x_zuvor), y=list(x_jetzt), mode="lines", line=dict(color=farbe, width=1.5), hoverinfo="skip", name="Bahn"))
    fig.update_xaxes(title_text=xtitle)
    fig.update_yaxes(title_text=ytitle)
    return _base(fig, legend=False)


def build_hut_fit(fit, r_jetzt, span_jetzt):
    """Spannweite² gegen r·τ nahe der Grenze: die fünf Messpunkte, die Gerade, ihre Nullstelle und π/2."""
    fig = go.Figure()
    xs = np.linspace(min(fit.r0, fit.rs[0]) - 0.01, fit.rs[-1] + 0.02, 40)
    fig.add_trace(go.Scatter(x=xs, y=fit.c1 * xs + fit.c0, mode="lines", line=dict(color=C.COLOR_GREY, dash="dash"), name="Gerade durch die Punkte",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=list(fit.rs), y=[s * s for s in fit.spans], mode="markers", marker=dict(size=10, color=C.COLOR_MAIN), name="Messpunkte",
                             hovertemplate="r·τ = %{x:.2f}: Spannweite² %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[fit.r0], y=[0], mode="markers", marker=dict(size=12, color=C.COLOR_ALT, symbol="x", line=dict(width=3)),
                             name=f"Nullstelle {fit.r0:.3f}".replace(".", ","), hovertemplate="Nullstelle %{x:.4f}<extra></extra>"))
    if fit.rs[0] - 0.05 <= r_jetzt <= fit.rs[-1] + 0.05:
        fig.add_trace(go.Scatter(x=[r_jetzt], y=[span_jetzt ** 2], mode="markers", marker=dict(size=11, color=C.COLOR_ALT, symbol="diamond-open", line=dict(width=3)),
                                 name="gewählter Wert", hovertemplate="r·τ = %{x:.2f}: Spannweite² %{y:.3f}<extra></extra>"))
    fig.add_vline(x=math.pi / 2, line=dict(color=C.COLOR_GREY, dash="dot"), annotation_text="π/2", annotation_position="top")
    fig.update_xaxes(title_text="r·τ")
    fig.update_yaxes(title_text="Spannweite²")
    return _base(fig)


def build_zeitschritt(z):
    """Experiment 4: Wachstumsrate der Rechnung gegen die Zahl der Schritte je τ (Euler und Heun) und die Rate der Gleichung."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(z.kurve_m), y=list(z.kurve_euler), mode="lines+markers", line=dict(color=C.COLOR_ALT, width=2.5), name="Euler",
                             hovertemplate="m = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(z.kurve_m), y=list(z.kurve_heun), mode="lines+markers", line=dict(color=C.COLOR_LINE, width=2.5), name="Heun",
                             hovertemplate="m = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=z.exakt, line=dict(color=C.COLOR_MAIN, dash="dash"), annotation_text="Gleichung (Re s)", annotation_position="bottom right")
    fig.add_hline(y=0, line=dict(color=C.COLOR_GREY))
    fig.add_trace(go.Scatter(x=[z.m], y=[z.gemessen], mode="markers", marker=dict(size=13, color="#111", symbol="x", line=dict(width=3)),
                             name=f"gemessen ({z.verfahren}, m = {z.m})", hovertemplate="m = %{x}: %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Schritte je Verzögerung m", type="log")
    fig.update_yaxes(title_text="Wachstumsrate je Verzögerung")
    return _base(fig)


def build_mg_zeit(g):
    fig = go.Figure(go.Scatter(x=list(g.t), y=list(g.x), mode="lines", line=dict(color=C.COLOR_MAIN, width=1.8), hovertemplate="t = %{x:.1f}: %{y:.4f}<extra></extra>",
                               name="x(t)"))
    fig.update_xaxes(title_text="Zeit (Zeiteinheiten, nach 5000 Einschwingzeit)")
    fig.update_yaxes(title_text="x(t)")
    return _base(fig, legend=False)


def build_mg_gitter(reihe, tau_jetzt, lam_jetzt):
    """Vorgerechnetes Gitter λ(τ) (Mittel über drei Zufalls-Seeds, Band Minimum bis Maximum) mit dem live gemessenen Punkt."""
    rows = reihe["rows"]
    taus = [r["tau"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=taus + taus[::-1], y=[r["max"] for r in rows] + [r["min"] for r in rows][::-1], fill="toself", fillcolor=C.COLOR_BAND,
                             line=dict(width=0), hoverinfo="skip", name="Minimum bis Maximum (3 Zufalls-Seeds)"))
    fig.add_trace(go.Scatter(x=taus, y=[r["lam"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_LINE, width=2.5), name="vorgerechnet (Mittel)",
                             hovertemplate="τ = %{x}: λ = %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[tau_jetzt], y=[lam_jetzt], mode="markers", marker=dict(size=12, color=C.COLOR_ALT, symbol="diamond-open", line=dict(width=3)),
                             name="live gemessen", hovertemplate="τ = %{x}: λ = %{y:.4f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C.COLOR_GREY))
    fig.update_xaxes(title_text="Verzögerung τ")
    fig.update_yaxes(title_text="λ je Zeiteinheit")
    return _base(fig)


def build_lorenz_bahn(lo):
    fig = go.Figure(go.Scatter(x=list(lo.pfad_x), y=list(lo.pfad_z), mode="lines", line=dict(color=C.COLOR_MAIN, width=1.5), hoverinfo="skip", name="Bahn"))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="z")
    return _base(fig, legend=False)


def build_lorenz_abstand(lo):
    """Abstand zweier Bahnen (Anfangsabstand 1e-9) über die Zeit, logarithmisch; Linie bei Abstand 1."""
    t, d = np.array(lo.abstand_t), np.maximum(np.array(lo.abstand), 1e-300)
    fig = go.Figure(go.Scatter(x=t, y=d, mode="lines", line=dict(color=C.COLOR_ALT, width=2), name="Abstand", hovertemplate="t = %{x:.1f}: %{y:.3g}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text="Abstand 1", annotation_position="bottom right")
    fig.update_xaxes(title_text="Zeit")
    fig.update_yaxes(title_text="Abstand der beiden Bahnen", type="log")
    return _base(fig, legend=False)


def build_lorenz_gitter(reihe, rho_jetzt, lam1_jetzt, rho_hopf):
    """Vorgerechnetes λ₁(ρ) mit dem live gemessenen Punkt und der Hopf-Schwelle ρ_H der Fixpunkte."""
    rows = reihe["rows"]
    rhos = [r["rho"] for r in rows]
    l1 = [r["lam"][0] for r in rows]
    err = [r["sem"][0] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rhos, y=l1, mode="lines+markers", error_y=dict(type="data", array=err, visible=True), line=dict(color=C.COLOR_LINE, width=2.5),
                             name="λ₁ vorgerechnet", hovertemplate="ρ = %{x}: λ₁ = %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[rho_jetzt], y=[lam1_jetzt], mode="markers", marker=dict(size=12, color=C.COLOR_ALT, symbol="diamond-open", line=dict(width=3)),
                             name="live gemessen", hovertemplate="ρ = %{x}: λ₁ = %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C.COLOR_GREY))
    fig.add_vline(x=rho_hopf, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text="ρ_H", annotation_position="top")
    fig.update_xaxes(title_text="ρ")
    fig.update_yaxes(title_text="größter Exponent λ₁")
    return _base(fig)
