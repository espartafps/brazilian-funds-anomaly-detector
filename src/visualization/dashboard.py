"""
Interactive Dashboard
Builds a single-page HTML dashboard from pipeline outputs using Plotly.
Run after completing the full analysis pipeline.
"""

import json
import os
import webbrowser

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PROCESSED_DIR = "data/processed"
REPORTS_DIR = "reports/generated"
OUTPUT_PATH = os.path.join(REPORTS_DIR, "dashboard.html")

# GitHub-dark colour palette
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


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> dict:
    """Load all pipeline outputs; returns empty frames for missing files."""
    print("Loading data files...")
    data: dict = {}

    parquet_map = [
        ("anomaly_results",  "anomaly_results.parquet"),
        ("anomaly_summary",  "anomaly_summary.parquet"),
        ("anomaly_by_type",  "anomaly_summary_by_type.parquet"),
        ("market_data",      "market_data.parquet"),
        ("signal_matrix",    "signal_matrix.parquet"),
    ]

    for key, filename in parquet_map:
        path = os.path.join(PROCESSED_DIR, filename)
        try:
            df = pd.read_parquet(path)
            if key == "market_data":
                # Saved with DatetimeIndex; restore as a plain 'date' column
                df = df.reset_index()
                first = df.columns[0]
                if first != "date":
                    df = df.rename(columns={first: "date"})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            data[key] = df
            print(f"  ✓ {key}: {len(df):,} rows")
        except FileNotFoundError:
            print(f"  ✗ {key}: not found — run the pipeline first")
            data[key] = pd.DataFrame()
        except Exception as exc:
            print(f"  ✗ {key}: {exc}")
            data[key] = pd.DataFrame()

    model_path = os.path.join(REPORTS_DIR, "model_results.json")
    try:
        with open(model_path) as fh:
            data["model_results"] = json.load(fh)
        print("  ✓ model_results: loaded")
    except FileNotFoundError:
        print("  ✗ model_results: not found — run predictive_model.py first")
        data["model_results"] = None

    return data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cumret(series: pd.Series) -> pd.Series:
    """Cumulative return (%) relative to the first valid observation."""
    base = series.dropna().iat[0] if series.dropna().size else 1.0
    return (series / base - 1.0) * 100.0


def _no_data(fig: go.Figure, row: int, col: int, msg: str = "") -> None:
    """Place a centred placeholder text when a panel has no data."""
    note = msg or "No data — run the pipeline first."
    fig.add_trace(
        go.Scatter(
            x=[0.5], y=[0.5],
            mode="text",
            text=[f"<i>{note}</i>"],
            textfont=dict(size=11, color=_C["sub"]),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row, col=col,
    )
    fig.update_xaxes(range=[0, 1], visible=False, row=row, col=col)
    fig.update_yaxes(range=[0, 1], visible=False, row=row, col=col)


# ── Panel builders ────────────────────────────────────────────────────────────

def _panel_1(fig: go.Figure, data: dict) -> None:
    """Timeline — % of funds flagged each day (anomaly / vol-regime / outflow)."""
    summary = data["anomaly_summary"]
    if summary.empty:
        _no_data(fig, 1, 1, "Run anomaly_detector.py → anomaly_summary.parquet")
        return

    series_cfg = [
        ("pct_is_anomaly",       "Return Anomalies (%)",    _C["anomaly"],    "tozeroy", "rgba(231,76,60,0.12)",  "solid"),
        ("pct_vol_regime_change", "Vol. Regime Changes (%)", _C["vol_regime"], None,      None,                   "solid"),
        ("pct_is_large_outflow",  "Large Outflows (%)",      _C["outflow"],    None,      None,                   "dot"),
    ]

    added = False
    for col_name, label, color, fill, fillcolor, dash in series_cfg:
        if col_name not in summary.columns:
            continue
        kwargs: dict = dict(
            x=summary["date"],
            y=summary[col_name],
            name=label,
            line=dict(color=color, width=1.5, dash=dash),
        )
        if fill:
            kwargs["fill"] = fill
            kwargs["fillcolor"] = fillcolor
        fig.add_trace(go.Scatter(**kwargs), row=1, col=1)
        added = True

    if not added:
        _no_data(fig, 1, 1, "anomaly_summary.parquet has no expected pct_* columns.")


def _panel_2(fig: go.Figure, data: dict) -> None:
    """Market indicators (cumul. %) on primary axis + anomaly density on secondary."""
    market  = data["market_data"]
    summary = data["anomaly_summary"]

    has_market  = not market.empty
    has_anomaly = (not summary.empty) and ("pct_is_anomaly" in summary.columns)

    if not has_market and not has_anomaly:
        _no_data(fig, 2, 1, "Run market_collector.py → market_data.parquet")
        return

    if has_market:
        market_series = [
            ("ibovespa_close", "Ibovespa", _C["ibovespa"], "solid"),
            ("vix_close",      "VIX",      _C["vix"],      "dash"),
            ("usd_brl_close",  "USD/BRL",  _C["usd_brl"],  "solid"),
        ]
        for ticker, label, color, dash in market_series:
            if ticker not in market.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=market["date"],
                    y=_cumret(market[ticker]),
                    name=label,
                    line=dict(color=color, width=1.5, dash=dash),
                ),
                row=2, col=1, secondary_y=False,
            )

    if has_anomaly:
        fig.add_trace(
            go.Scatter(
                x=summary["date"],
                y=summary["pct_is_anomaly"],
                name="Anomaly Density (%)",
                line=dict(color=_C["anomaly"], width=1.2, dash="dot"),
                fill="tozeroy",
                fillcolor="rgba(231,76,60,0.07)",
            ),
            row=2, col=1, secondary_y=True,
        )


def _panel_3(fig: go.Figure, data: dict) -> None:
    """Horizontal bar chart — top 15 Random Forest feature importances."""
    model = data["model_results"]
    if not model or not model.get("top_features"):
        _no_data(fig, 2, 2, "Run predictive_model.py → model_results.json")
        return

    top15 = sorted(model["top_features"].items(), key=lambda kv: kv[1])[-15:]
    names  = [kv[0] for kv in top15]
    values = [kv[1] for kv in top15]

    fig.add_trace(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            name="Importance",
            marker=dict(
                color=values,
                colorscale="Teal",
                showscale=False,
                line_width=0,
            ),
            showlegend=False,
        ),
        row=2, col=2,
    )


def _panel_4(fig: go.Figure, data: dict) -> None:
    """Overlapping histograms — Z-score distribution split by anomaly flag."""
    df = data["anomaly_results"]
    if df.empty or "z_score" not in df.columns:
        _no_data(fig, 3, 1, "Run anomaly_detector.py → anomaly_results.parquet")
        return

    z = df["z_score"].dropna().clip(-8, 8)

    if "is_anomaly" in df.columns:
        mask = df.loc[z.index, "is_anomaly"]
        fig.add_trace(
            go.Histogram(
                x=z[~mask], name="Normal",
                nbinsx=120, bingroup="z",
                marker_color=_C["normal_z"],
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=z[mask], name="Anomaly",
                nbinsx=120, bingroup="z",
                marker_color=_C["anomaly_z"],
            ),
            row=3, col=1,
        )
    else:
        fig.add_trace(
            go.Histogram(x=z, name="Z-score", nbinsx=120, marker_color=_C["normal_z"]),
            row=3, col=1,
        )

    # Detection threshold markers at ±2.5
    for x_val in (2.5, -2.5):
        fig.add_vline(
            x=x_val, row=3, col=1,
            line_dash="dash", line_color=_C["anomaly"], line_width=1,
        )


def _panel_5(fig: go.Figure, data: dict) -> None:
    """Volatility ratio over time — daily median with IQR shaded band."""
    df = data["anomaly_results"]
    if df.empty or "vol_ratio" not in df.columns:
        df = data["signal_matrix"]

    if df.empty or "vol_ratio" not in df.columns or "date" not in df.columns:
        _no_data(fig, 3, 2, "Run anomaly_detector.py → anomaly_results.parquet")
        return

    grp    = df.groupby("date")["vol_ratio"]
    median = grp.median().rename("median")
    q25    = grp.quantile(0.25).rename("q25")
    q75    = grp.quantile(0.75).rename("q75")
    daily  = pd.concat([median, q25, q75], axis=1).reset_index()

    # IQR shaded band (polygon: upper edge L→R, lower edge R→L)
    x_band = pd.concat([daily["date"], daily["date"].iloc[::-1]], ignore_index=True)
    y_band = pd.concat([daily["q75"],  daily["q25"].iloc[::-1]],  ignore_index=True)

    fig.add_trace(
        go.Scatter(
            x=x_band, y=y_band,
            fill="toself",
            fillcolor=_C["vol_band"],
            line=dict(color="rgba(0,0,0,0)"),
            name="IQR (25–75%)",
            hoverinfo="skip",
        ),
        row=3, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["median"],
            name="Median Vol. Ratio",
            line=dict(color=_C["vol_line"], width=1.5),
        ),
        row=3, col=2,
    )

    # Regime-change threshold line at 1.5
    fig.add_hline(
        y=1.5, row=3, col=2,
        line_dash="dash", line_color=_C["anomaly"], line_width=1,
        annotation_text="Threshold 1.5",
        annotation_font_color=_C["anomaly"],
        annotation_position="bottom right",
    )


def _panel_6(fig: go.Figure, data: dict) -> None:
    """Anomaly rate over time segmented by fund type."""
    df = data.get("anomaly_by_type")
    if df is None or df.empty or "fund_type" not in df.columns:
        _no_data(fig, 4, 1, "Re-run pipeline com fund_type → cvm_collector.py → anomaly_detector.py")
        return

    if "pct_is_anomaly" not in df.columns:
        _no_data(fig, 4, 1, "anomaly_summary_by_type.parquet sem coluna pct_is_anomaly")
        return

    # Palette: cycle through distinct colours
    colours = [
        "#e74c3c", "#27ae60", "#2980b9", "#f39c12",
        "#9b59b6", "#1abc9c", "#d35400", "#58a6ff",
    ]
    fund_types = sorted(df["fund_type"].dropna().unique())

    for i, ft in enumerate(fund_types):
        subset = df[df["fund_type"] == ft].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["pct_is_anomaly"],
                name=ft,
                line=dict(color=colours[i % len(colours)], width=1.5),
                mode="lines",
            ),
            row=4, col=1,
        )


# ── Figure assembly ───────────────────────────────────────────────────────────

def _build_figure(data: dict) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "<b>1 · Timeline de Anomalias</b>",                  "",
            "<b>2 · Anomalias vs. Indicadores de Mercado</b>",
            "<b>3 · Top Features — Modelo Preditivo</b>",
            "<b>4 · Distribuição de Z-scores</b>",
            "<b>5 · Volatility Ratio ao Longo do Tempo</b>",
            "<b>6 · Anomalias por Tipo de Fundo</b>",            "",
        ],
        specs=[
            [{"colspan": 2}, None],
            [{"secondary_y": True}, {}],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        row_heights=[0.18, 0.32, 0.27, 0.23],
        vertical_spacing=0.08,
        horizontal_spacing=0.07,
    )

    _panel_1(fig, data)
    _panel_2(fig, data)
    _panel_3(fig, data)
    _panel_4(fig, data)
    _panel_5(fig, data)
    _panel_6(fig, data)

    return fig


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        title=dict(
            text="<b>Brazilian Funds Anomaly Detector</b> — Dashboard Interativo",
            font=dict(size=20, color=_C["text"], family="Inter, Arial, sans-serif"),
            x=0.5, xanchor="center", y=0.99,
        ),
        template="plotly_dark",
        paper_bgcolor=_C["paper"],
        plot_bgcolor=_C["bg"],
        font=dict(family="Inter, Arial, sans-serif", color=_C["text"], size=11),
        height=1500,
        margin=dict(t=70, b=40, l=60, r=40),
        hovermode="x unified",
        barmode="overlay",
        legend=dict(
            bgcolor="rgba(22,27,34,0.90)",
            bordercolor=_C["grid"],
            borderwidth=1,
            font=dict(size=10),
            tracegroupgap=4,
        ),
    )

    fig.update_xaxes(gridcolor=_C["grid"], zerolinecolor=_C["grid"], tickfont_size=10)
    fig.update_yaxes(gridcolor=_C["grid"], zerolinecolor=_C["grid"], tickfont_size=10)

    # Per-panel axis labels
    fig.update_yaxes(title_text="% de Fundos",     title_font_size=10, row=1, col=1)
    fig.update_yaxes(title_text="Retorno Acum. (%)", title_font_size=10, secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Anomaly Density (%)", title_font_size=10, secondary_y=True,  row=2, col=1)
    fig.update_xaxes(title_text="Importância",     title_font_size=10, row=2, col=2)
    fig.update_xaxes(title_text="Z-score",         title_font_size=10, row=3, col=1)
    fig.update_yaxes(title_text="Contagem",         title_font_size=10, row=3, col=1)
    fig.update_yaxes(title_text="Vol. Ratio",       title_font_size=10, row=3, col=2)
    fig.update_yaxes(title_text="% de Fundos",     title_font_size=10, row=4, col=1)

    # Subplot title styling
    for ann in fig.layout.annotations:
        if ann.text.startswith("<b>"):
            ann.font.color = _C["text"]
            ann.font.size  = 12

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Brazilian Funds — Dashboard Builder")
    print("=" * 60)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    data = load_data()

    print("\nBuilding dashboard...")
    fig = _build_figure(data)
    fig = _apply_theme(fig)

    fig.write_html(
        OUTPUT_PATH,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )

    abs_path = os.path.abspath(OUTPUT_PATH).replace("\\", "/")
    print(f"\nSaved: {OUTPUT_PATH}")
    webbrowser.open(f"file:///{abs_path}")
    print("Opening in browser...")


if __name__ == "__main__":
    main()
