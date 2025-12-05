#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from dateutil import parser as dateparser
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash  # for callback_context

# -------- CLI --------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--truflation-path", default=str(Path("data/truflation/cpi_us_yoy_frozen.csv")))
    p.add_argument("--bls-path", default=str(Path("data/bls/bls_cpi_yoy.csv")))
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--shift-min", type=int, default=-240)
    p.add_argument("--shift-max", type=int, default=240)
    p.add_argument("--shift-step", type=int, default=1)
    p.add_argument("--shift-default", type=int, default=0)
    p.add_argument("--line-opacity", type=float, default=0.50)
    return p.parse_args()


# -------- Loaders --------
def _parse_any_date(s) -> pd.Timestamp:
    return pd.to_datetime(dateparser.parse(str(s)).date())


def load_truflation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"Date", "CPI_YoY"}.issubset(df.columns), "Truflation CSV must have columns: Date,CPI_YoY"
    df = df.rename(columns={"Date": "date", "CPI_YoY": "truflation"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df[["date", "truflation"]]


def _forward_fill_daily_from_releases(releases: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    rel = releases.sort_values("date").reset_index(drop=True)
    idx = pd.date_range(rel["date"].iloc[0], end_date, freq="D")
    s = pd.Series(index=idx, dtype=float)
    for _, r in rel.iterrows():
        s.loc[pd.Timestamp(r["date"])] = float(r["value"])
    s = s.ffill()
    return pd.DataFrame({"date": s.index, "bls": s.values})


def load_bls_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower() for c in df.columns}

    # Case 1: Already daily (date, bls)
    if {"date", "bls"}.issubset(cols):
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df["date"] = pd.to_datetime(df["date"].apply(_parse_any_date))
        df["bls"] = pd.to_numeric(df["bls"], errors="coerce")
        df = df.dropna(subset=["date", "bls"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
        return df[["date", "bls"]]

    # Case 2: Release dates (release_date, cpi_yoy)
    if {"release_date", "cpi_yoy"}.issubset(cols):
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df["date"] = pd.to_datetime(df["release_date"].apply(_parse_any_date))

        def _to_float(x):
            x = str(x).strip()
            if x.endswith("%"):
                x = x[:-1]
            return float(x)

        df["value"] = df["cpi_yoy"].apply(_to_float)
        df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
        end = pd.Timestamp.today().normalize()
        return _forward_fill_daily_from_releases(df[["date", "value"]], end)

    raise ValueError(
        f"BLS file must be daily (date,BLS) or releases (release_date,cpi_yoy). Got: {list(df.columns)}"
    )


# -------- Full grid --------
def build_full_grid(tru: pd.DataFrame, bls: pd.DataFrame, shift_min: int, shift_max: int) -> pd.DataFrame:
    tmin, tmax = tru["date"].min(), tru["date"].max()
    bmin, bmax = bls["date"].min(), bls["date"].max()
    xmin = min(tmin + pd.Timedelta(days=shift_min), bmin)
    xmax = max(tmax + pd.Timedelta(days=shift_max), bmax)
    idx = pd.date_range(xmin, xmax, freq="D")

    base = pd.DataFrame({"date": idx})
    base = base.merge(tru, on="date", how="left")
    base = base.merge(bls, on="date", how="left")
    return base


def shift_truflation_on_grid(base: pd.DataFrame, days: int) -> pd.Series:
    if days == 0:
        return base["truflation"]

    shifted = base[["date", "truflation"]].dropna().copy()
    shifted["date"] = shifted["date"] + pd.to_timedelta(days, unit="D")
    ghost = base[["date"]].merge(
        shifted.rename(columns={"truflation": "ghost"}), on="date", how="left"
    )
    return ghost["ghost"]


def corr_overlap(ghost: pd.Series, bls: pd.Series) -> float | None:
    sub = pd.concat([ghost, bls], axis=1, keys=["g", "b"]).dropna()
    if len(sub) < 5:
        return None
    return float(sub["g"].corr(sub["b"]))


# -------- Correlation curves --------
def compute_corr_curve(base: pd.DataFrame, shift_min: int = -80, shift_max: int = 80) -> pd.DataFrame:
    rows = []
    for d in range(shift_min, shift_max + 1):
        ghost = shift_truflation_on_grid(base, d)
        r = corr_overlap(ghost, base["bls"])
        rows.append({"shift": d, "r": r})
    return pd.DataFrame(rows)


def build_corr_figs(corr_df: pd.DataFrame):
    valid = corr_df.dropna(subset=["r"]).copy()
    if valid.empty:
        empty_corr = go.Figure()
        empty_corr.update_layout(
            title="Correlation vs shift",
            xaxis_title="Shift (days)",
            yaxis_title="Pearson r",
            template="plotly_white",
        )
        empty_r2 = go.Figure()
        empty_r2.update_layout(
            title="R² vs shift",
            xaxis_title="Shift (days)",
            yaxis_title="R²",
            template="plotly_white",
        )
        return empty_corr, None, None, empty_r2, None, None

    valid["r2"] = valid["r"] ** 2

    idx_max_r = valid["r"].idxmax()
    max_r_shift = int(valid.loc[idx_max_r, "shift"])
    max_r = float(valid.loc[idx_max_r, "r"])

    idx_max_r2 = valid["r2"].idxmax()
    max_r2_shift = int(valid.loc[idx_max_r2, "shift"])
    max_r2 = float(valid.loc[idx_max_r2, "r2"])

    corr_fig = go.Figure()
    corr_fig.add_trace(
        go.Scatter(
            x=corr_df["shift"],
            y=corr_df["r"],
            mode="lines+markers",
            name="Pearson r",
        )
    )
    corr_fig.add_vline(
        x=max_r_shift,
        line_dash="dash",
        line_width=2,
        line_color="red",
        annotation_text=f"max r={max_r:.3f} at {max_r_shift:+d}d",
        annotation_position="top left",
    )
    corr_fig.update_layout(
        title="Correlation vs time shift (Truflation shifted vs BLS)",
        xaxis_title="Shift (days, Truflation vs BLS)",
        yaxis_title="Pearson correlation r",
        template="plotly_white",
    )

    r2_full = pd.Series(index=corr_df.index, dtype=float)
    r2_full.loc[valid.index] = valid["r2"]

    r2_fig = go.Figure()
    r2_fig.add_trace(
        go.Scatter(
            x=corr_df["shift"],
            y=r2_full,
            mode="lines+markers",
            name="R²",
        )
    )
    r2_fig.add_vline(
        x=max_r2_shift,
        line_dash="dash",
        line_width=2,
        line_color="red",
        annotation_text=f"max R²={max_r2:.3f} at {max_r2_shift:+d}d",
        annotation_position="top left",
    )
    r2_fig.update_layout(
        title="R² vs time shift (Truflation shifted vs BLS)",
        xaxis_title="Shift (days, Truflation vs BLS)",
        yaxis_title="R²",
        template="plotly_white",
    )

    return corr_fig, max_r_shift, max_r, r2_fig, max_r2_shift, max_r2


# -------- Plot --------
UIREV = "keep-zoom"


def build_fig(base: pd.DataFrame, ghost: pd.Series, shapes: list[dict] | None = None, line_opacity: float = 0.25) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=base["date"],
            y=base["truflation"],
            mode="lines",
            name="Truflation (unshifted)",
            yaxis="y1",
            line=dict(width=2, color="rgba(0, 102, 255, 1)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=base["date"],
            y=ghost,
            mode="lines",
            name="Truflation (shifted)",
            yaxis="y1",
            line=dict(width=2),
            opacity=0.35,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=base["date"],
            y=base["bls"],
            mode="lines",
            name="BLS",
            yaxis="y2",
            line=dict(width=2, color="rgba(219, 123, 73, 1)"),
        )
    )

    fig.update_layout(
        newshape=dict(
            opacity=line_opacity,
            line_color=f"rgba(0,0,0,{line_opacity})",
            line_width=2,
        )
    )

    fig.update_layout(
        uirevision=UIREV,
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            title="Date",
            type="date",
            rangeslider=dict(visible=False),
            fixedrange=False,
            dtick="M6",
            tick0="2025-10-01",
            tickformat="%b %Y",
            tickangle=-35,
        ),
        yaxis=dict(title="Truflation CPI YoY", side="left", fixedrange=False),
        yaxis2=dict(title="BLS CPI YoY", side="right", overlaying="y", anchor="x", fixedrange=False),
        dragmode="pan",
        template="plotly_white",
        shapes=shapes or [],
    )
    return fig


def enforce_shape_opacity(shapes: list[dict], line_opacity: float) -> list[dict]:
    fixed = []
    for s in shapes or []:
        s = dict(s)
        if "opacity" not in s:
            s["opacity"] = line_opacity

        line = s.get("line", {})
        if isinstance(line, dict):
            color = line.get("color")
            if color and color.startswith("rgb("):
                line["color"] = f"rgba{color[3:-1]},{line_opacity})"
            elif color and color.startswith("rgba("):
                try:
                    parts = color[5:-1].split(",")
                    if len(parts) == 4:
                        parts[-1] = str(line_opacity)
                    line["color"] = "rgba(" + ",".join(parts) + ")"
                except Exception:
                    pass
            else:
                line.setdefault("color", f"rgba(0,0,0,{line_opacity})")

            line.setdefault("width", 2)
            s["line"] = line

        fixed.append(s)
    return fixed


# -------- Dash app --------
def make_app(base: pd.DataFrame, args) -> Dash:
    app = Dash(__name__)

    graph_config = {
        "scrollZoom": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
        "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
        "edits": {"shapePosition": True},
    }

    corr_df = compute_corr_curve(base, shift_min=-80, shift_max=80)
    corr_fig, max_r_shift, max_r, r2_fig, max_r2_shift, max_r2 = build_corr_figs(corr_df)

    init_shift = int(args.shift_default)
    ghost0 = shift_truflation_on_grid(base, init_shift)
    r0 = corr_overlap(ghost0, base["bls"])
    fig0 = build_fig(base, ghost0, shapes=[], line_opacity=args.line_opacity)
    rtxt0 = (
        f"Shift: {init_shift:+d} days | Pearson r (ghost vs BLS): "
        + (f"{r0:.3f}" if r0 is not None else "n/a")
    )

    app.layout = html.Div(
        [
            html.H3("Truflation vs BLS"),
            dcc.Tabs(
                id="main-tabs",
                value="tab-series",
                children=[
                    dcc.Tab(
                        label="Time series & manual shift",
                        value="tab-series",
                        children=[
                            html.Label(
                                "Shift (days): adjusts Truflation in time; if alignment "
                                "requires pushing it later, that means Truflation is leading the BLS."
                            ),
                            dcc.Slider(
                                id="shift-slider",
                                min=int(args.shift_min),
                                max=int(args.shift_max),
                                step=int(args.shift_step),
                                value=init_shift,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Delete last line",
                                        id="btn-delete-last",
                                        n_clicks=0,
                                        style={"marginRight": "8px"},
                                    ),
                                    html.Button("Delete all lines", id="btn-delete-all", n_clicks=0),
                                ],
                                style={"margin": "8px 0"},
                            ),
                            html.Div(
                                id="shift-readout",
                                style={"margin": "8px 0", "fontFamily": "monospace"},
                                children=rtxt0,
                            ),
                            dcc.Store(id="shapes-store", data=[]),
                            dcc.Graph(
                                id="chart",
                                figure=fig0,
                                config=graph_config,
                                style={"height": "78vh"},
                            ),
                            html.Div(
                                "Tip: Use the pencil in the toolbar to draw. "
                                "Lines are semi-transparent by default. Use the buttons to delete.",
                                style={"color": "#555", "fontSize": "0.9rem", "marginTop": "6px"},
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Correlation vs shift (r)",
                        value="tab-corr",
                        children=[
                            dcc.Graph(id="corr-graph", figure=corr_fig, style={"height": "78vh"}),
                            html.Div(
                                (
                                    "Max Pearson r = {:.3f} at shift = {:+d} days "
                                    "(Truflation shifted vs BLS).".format(max_r, max_r_shift)
                                )
                                if max_r is not None
                                else "Correlation curve could not be computed (not enough overlapping data).",
                                style={"margin": "8px 0", "fontFamily": "monospace"},
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="R² vs shift",
                        value="tab-r2",
                        children=[
                            dcc.Graph(id="r2-graph", figure=r2_fig, style={"height": "78vh"}),
                            html.Div(
                                (
                                    "Max R² = {:.3f} at shift = {:+d} days "
                                    "(Truflation shifted vs BLS).".format(max_r2, max_r2_shift)
                                )
                                if max_r2 is not None
                                else "R² curve could not be computed (not enough overlapping data).",
                                style={"margin": "8px 0", "fontFamily": "monospace"},
                            ),
                        ],
                    ),
                ],
            ),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "12px"},
    )

    def merge_relayout_into_shapes(relayout: dict, current: list[dict]) -> list[dict]:
        if not relayout:
            return current or []

        if "shapes" in relayout and isinstance(relayout["shapes"], list):
            return relayout["shapes"]

        updated = list(current or [])
        for k, v in relayout.items():
            if not k.startswith("shapes["):
                continue
            try:
                idx = int(k.split("[", 1)[1].split("]", 1)[0])
            except Exception:
                continue

            while len(updated) <= idx:
                updated.append({})

            if "]." in k:
                prop = k.split("].", 1)[1]
                d = updated[idx]
                parts = prop.split(".")
                tgt = d
                for p in parts[:-1]:
                    if p not in tgt or not isinstance(tgt[p], dict):
                        tgt[p] = {}
                    tgt = tgt[p]
                tgt[parts[-1]] = v

        return updated

    @app.callback(
        Output("chart", "figure"),
        Output("shift-readout", "children"),
        Output("shapes-store", "data"),
        Input("shift-slider", "value"),
        Input("chart", "relayoutData"),
        Input("btn-delete-last", "n_clicks"),
        Input("btn-delete-all", "n_clicks"),
        State("shapes-store", "data"),
        prevent_initial_call=False,
    )
    def update_fig(shift_days: int, relayout_data: dict, n_last: int, n_all: int, shapes_data: list[dict]):
        trigger = (
            dash.callback_context.triggered[0]["prop_id"].split(".")[0]
            if dash.callback_context.triggered
            else None
        )

        shapes = merge_relayout_into_shapes(relayout_data or {}, shapes_data or [])

        if trigger == "btn-delete-last" and shapes:
            shapes = shapes[:-1]
        elif trigger == "btn-delete-all":
            shapes = []

        shapes = enforce_shape_opacity(shapes, args.line_opacity)

        ghost = shift_truflation_on_grid(base, int(shift_days))
        r = corr_overlap(ghost, base["bls"])

        fig = build_fig(base, ghost, shapes=shapes, line_opacity=args.line_opacity)
        txt = (
            f"Shift: {int(shift_days):+d} days | Pearson correlation (Truflation vs BLS): "
            + (f"{r:.3f}" if r is not None else "n/a")
        )

        return fig, txt, shapes

    return app


# -------- main --------
def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]

    tru_path = (project_root / args.truflation_path).resolve()
    bls_path = (project_root / args.bls_path).resolve()

    if not tru_path.exists():
        raise SystemExit(f"Truflation CSV not found: {tru_path}")
    if not bls_path.exists():
        raise SystemExit(f"BLS CSV not found: {bls_path}")

    tru = load_truflation(tru_path)
    bls = load_bls_auto(bls_path)

    base = build_full_grid(tru, bls, args.shift_min, args.shift_max)

    app = make_app(base, args)

    print(f"Serving at http://127.0.0.1:{args.port} (Ctrl+C to quit)")
    app.run(debug=False, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
