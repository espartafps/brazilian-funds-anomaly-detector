"""
Chart Exporter
Exports each dashboard chart as a standalone PDF (vector) and PNG (high-res)
suitable for inclusion in LaTeX documents.

Outputs go to reports/figures/:
  chart_01_timeline.pdf / .png
  chart_02_market.pdf   / .png
  chart_03_features.pdf / .png
  chart_04_zscore.pdf   / .png
  chart_05_volratio.pdf / .png
  chart_06_fundtype.pdf / .png

Usage:
  python src/visualization/export_charts.py
"""

import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go

sys.stdout.reconfigure(encoding="utf-8")

PROCESSED_DIR = "data/processed"
REPORTS_DIR   = "reports/generated"
FIGURES_DIR   = "reports/figures"

# Dimensions for LaTeX export — full text-width figures
# 1 200 px @ scale 3 ≈ 3 600 px ≈ 300 DPI on a standard A4 column
FIG_WIDTH  = 1_200
FIG_HEIGHT = 520
FIG_SCALE  = 3          # multiplier for PNG; PDF is always vector

_C = {
    "anomaly":    "#e74c3c",
    "vol_regime": "#f39c12",
    "outflow":    "#9b59b6",
    "ibovespa":   "#27ae60",
    "usd_brl":    "#2980b9",
    "vix":        "#d35400",
    "normal_z":   "rgba(52,152,219,0.50)",
    "anomaly_z":  "rgba(231,76,60,0.65)",
    "vol_band":   "rgba(241,196,15,0.18)",
    "vol_line":   "#f1c40f",
    "bg":         "#0d1117",
    "paper":      "#161b22",
    "grid":       "#30363d",
    "text":       "#c9d1d9",
    "sub":        "#8b949e",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> dict:
    print("Loading data files...")
    data: dict = {}

    for key, filename in [
        ("anomaly_results", "anomaly_results.parquet"),
        ("anomaly_summary", "anomaly_summary.parquet"),
        ("anomaly_by_type", "anomaly_summary_by_type.parquet"),
        ("market_data",     "market_data.parquet"),
        ("signal_matrix",   "signal_matrix.parquet"),
    ]:
        path = os.path.join(PROCESSED_DIR, filename)
        try:
            df = pd.read_parquet(path)
            if key == "market_data":
                df = df.reset_index()
                first = df.columns[0]
                if first != "date":
                    df = df.rename(columns={first: "date"})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            data[key] = df
            print(f"  ✓ {key}: {len(df):,} rows")
        except FileNotFoundError:
            print(f"  ✗ {key}: not found")
            data[key] = pd.DataFrame()

    model_path = os.path.join(REPORTS_DIR, "model_results.json")
    try:
        with open(model_path) as fh:
            data["model_results"] = json.load(fh)
        print("  ✓ model_results: loaded")
    except FileNotFoundError:
        print("  ✗ model_results: not found")
        data["model_results"] = None

    return data


# ── Shared layout helpers ──────────────────────────────────────────────────────

def _layout(title: str = "", **extra) -> dict:
    base = dict(
        template="plotly_dark",
        paper_bgcolor=_C["paper"],
        plot_bgcolor=_C["bg"],
        font=dict(family="Inter, Arial, sans-serif", color=_C["text"], size=13),
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
        margin=dict(t=50 if title else 30, b=55, l=70, r=60),
        hovermode="x unified",
        barmode="overlay",
        legend=dict(
            bgcolor="rgba(22,27,34,0.90)",
            bordercolor=_C["grid"],
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    if title:
        base["title"] = dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color=_C["text"]),
            x=0.01,
        )
    base.update(extra)
    return base


def _axis(title: str = "", **kw) -> dict:
    base = dict(gridcolor=_C["grid"], zerolinecolor=_C["grid"], tickfont_size=11)
    if title:
        base["title"] = dict(text=title, font=dict(size=12))
    base.update(kw)
    return base


def _cumret(series: pd.Series) -> pd.Series:
    base = series.dropna().iat[0] if series.dropna().size else 1.0
    return (series / base - 1.0) * 100.0


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=f"<i>{msg}</i>",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=13, color=_C["sub"]),
    )
    fig.update_layout(**_layout())
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ── Individual figure builders ─────────────────────────────────────────────────

def fig_timeline(data: dict) -> go.Figure:
    summary = data["anomaly_summary"]
    if summary.empty:
        return _empty("Run anomaly_detector.py first")

    fig = go.Figure()
    for col, label, color, fill, fillcolor, dash in [
        ("pct_is_anomaly",        "Return Anomalies (%)",    _C["anomaly"],    "tozeroy", "rgba(231,76,60,0.12)",  "solid"),
        ("pct_vol_regime_change",  "Vol. Regime Changes (%)", _C["vol_regime"], None,      None,                   "solid"),
        ("pct_is_large_outflow",   "Large Outflows (%)",      _C["outflow"],    None,      None,                   "dot"),
    ]:
        if col not in summary.columns:
            continue
        kw: dict = dict(x=summary["date"], y=summary[col], name=label,
                        line=dict(color=color, width=2, dash=dash))
        if fill:
            kw["fill"] = fill
            kw["fillcolor"] = fillcolor
        fig.add_trace(go.Scatter(**kw))

    fig.update_layout(**_layout("Timeline de Anomalias"))
    fig.update_xaxes(**_axis())
    fig.update_yaxes(**_axis("% de Fundos"))
    return fig


def fig_market(data: dict) -> go.Figure:
    market  = data["market_data"]
    summary = data["anomaly_summary"]
    has_market  = not market.empty
    has_anomaly = not summary.empty and "pct_is_anomaly" in summary.columns

    if not has_market and not has_anomaly:
        return _empty("Run market_collector.py first")

    fig = go.Figure()
    if has_market:
        for ticker, label, color, dash in [
            ("ibovespa_close", "Ibovespa", _C["ibovespa"], "solid"),
            ("vix_close",      "VIX",      _C["vix"],      "dash"),
            ("usd_brl_close",  "USD/BRL",  _C["usd_brl"],  "solid"),
        ]:
            if ticker not in market.columns:
                continue
            fig.add_trace(go.Scatter(
                x=market["date"], y=_cumret(market[ticker]),
                name=label, yaxis="y",
                line=dict(color=color, width=2, dash=dash),
            ))

    if has_anomaly:
        fig.add_trace(go.Scatter(
            x=summary["date"], y=summary["pct_is_anomaly"],
            name="Anomaly Density (%)", yaxis="y2",
            line=dict(color=_C["anomaly"], width=1.5, dash="dot"),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.07)",
        ))

    fig.update_layout(
        **_layout("Anomalias vs. Indicadores de Mercado"),
        yaxis =_axis("Retorno Acum. (%)"),
        yaxis2=dict(**_axis("Anomaly Density (%)"),
                    overlaying="y", side="right", showgrid=False),
    )
    return fig


def fig_features(data: dict) -> go.Figure:
    model = data["model_results"]
    if not model or not model.get("top_features"):
        return _empty("Run predictive_model.py first")

    top15  = sorted(model["top_features"].items(), key=lambda kv: kv[1])[-15:]
    names  = [kv[0] for kv in top15]
    values = [kv[1] for kv in top15]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=values, colorscale="Teal", showscale=False, line_width=0),
        showlegend=False,
    ))
    fig.update_layout(**_layout("Top Features — Modelo Preditivo"))
    fig.update_xaxes(**_axis("Importância"))
    fig.update_yaxes(**_axis())
    return fig


def fig_zscore(data: dict) -> go.Figure:
    df = data["anomaly_results"]
    if df.empty or "z_score" not in df.columns:
        return _empty("Run anomaly_detector.py first")

    z   = df["z_score"].dropna().clip(-8, 8)
    fig = go.Figure()

    if "is_anomaly" in df.columns:
        mask = df.loc[z.index, "is_anomaly"]
        fig.add_trace(go.Histogram(x=z[~mask], name="Normal",
                                   nbinsx=120, bingroup="z",
                                   marker_color=_C["normal_z"]))
        fig.add_trace(go.Histogram(x=z[mask],  name="Anomalia",
                                   nbinsx=120, bingroup="z",
                                   marker_color=_C["anomaly_z"]))
    else:
        fig.add_trace(go.Histogram(x=z, name="Z-score", nbinsx=120,
                                   marker_color=_C["normal_z"]))

    fig.add_vline(x= 2.5, line_dash="dash", line_color=_C["anomaly"], line_width=1.5)
    fig.add_vline(x=-2.5, line_dash="dash", line_color=_C["anomaly"], line_width=1.5)

    fig.update_layout(**_layout("Distribuição de Z-scores"))
    fig.update_xaxes(**_axis("Z-score"))
    fig.update_yaxes(**_axis("Contagem"))
    return fig


def fig_volratio(data: dict) -> go.Figure:
    df = data["anomaly_results"]
    if df.empty or "vol_ratio" not in df.columns:
        df = data["signal_matrix"]
    if df.empty or "vol_ratio" not in df.columns or "date" not in df.columns:
        return _empty("Run anomaly_detector.py first")

    grp   = df.groupby("date")["vol_ratio"]
    daily = pd.concat([
        grp.median().rename("median"),
        grp.quantile(0.25).rename("q25"),
        grp.quantile(0.75).rename("q75"),
    ], axis=1).reset_index()

    x_band = pd.concat([daily["date"], daily["date"].iloc[::-1]], ignore_index=True)
    y_band = pd.concat([daily["q75"],  daily["q25"].iloc[::-1]],  ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_band, y=y_band, fill="toself",
        fillcolor=_C["vol_band"], line=dict(color="rgba(0,0,0,0)"),
        name="IQR (25–75%)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["median"],
        name="Median Vol. Ratio",
        line=dict(color=_C["vol_line"], width=2),
    ))
    fig.add_hline(y=1.5, line_dash="dash", line_color=_C["anomaly"], line_width=1.5,
                  annotation_text="Threshold 1.5",
                  annotation_font_color=_C["anomaly"],
                  annotation_position="bottom right")

    fig.update_layout(**_layout("Volatility Ratio ao Longo do Tempo"))
    fig.update_xaxes(**_axis())
    fig.update_yaxes(**_axis("Vol. Ratio"))
    return fig


def fig_fundtype(data: dict) -> go.Figure:
    df = data.get("anomaly_by_type")
    if df is None or df.empty or "fund_type" not in df.columns:
        return _empty("Re-run pipeline com fund_type → cvm_collector.py → anomaly_detector.py")
    if "pct_is_anomaly" not in df.columns:
        return _empty("anomaly_summary_by_type.parquet sem coluna pct_is_anomaly")

    colours = [
        "#e74c3c", "#27ae60", "#2980b9", "#f39c12",
        "#9b59b6", "#1abc9c", "#d35400", "#58a6ff",
    ]
    fund_types = sorted(df["fund_type"].dropna().unique())
    fig = go.Figure()
    for i, ft in enumerate(fund_types):
        subset = df[df["fund_type"] == ft].sort_values("date")
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["pct_is_anomaly"],
            name=ft, mode="lines",
            line=dict(color=colours[i % len(colours)], width=2),
        ))

    fig.update_layout(**_layout("Anomalias por Tipo de Fundo"))
    fig.update_xaxes(**_axis())
    fig.update_yaxes(**_axis("% de Fundos"))
    return fig


# ── Export ─────────────────────────────────────────────────────────────────────

CHARTS = [
    ("chart_01_timeline", fig_timeline),
    ("chart_02_market",   fig_market),
    ("chart_03_features", fig_features),
    ("chart_04_zscore",   fig_zscore),
    ("chart_05_volratio", fig_volratio),
    ("chart_06_fundtype", fig_fundtype),
]


def export_all(data: dict, fmt: list = ("pdf", "png")) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for name, builder in CHARTS:
        print(f"\nBuilding {name}...")
        fig = builder(data)

        for ext in fmt:
            path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
            if ext == "png":
                fig.write_image(path, width=FIG_WIDTH, height=FIG_HEIGHT, scale=FIG_SCALE)
            else:
                fig.write_image(path, width=FIG_WIDTH, height=FIG_HEIGHT)
            print(f"  ✓ {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Brazilian Funds — Chart Exporter")
    print("=" * 60)

    data = load_data()
    export_all(data, fmt=["pdf", "png"])

    print(f"\nAll figures saved to: {os.path.abspath(FIGURES_DIR)}/")
    print("\nLaTeX usage example:")
    print(r"  \includegraphics[width=\textwidth]{figures/chart_01_timeline.pdf}")
