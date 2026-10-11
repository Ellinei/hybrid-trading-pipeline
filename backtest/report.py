"""Self-contained Plotly HTML report for a completed cli experiment run:
equity curve + drawdown, monthly PF bars, trade PnL scatter, exit-reason
mix. Written to results/<exp_id>/report.html by backtest.experiment.

Palette and mark choices follow the project's dataviz skill: categorical
hues in a fixed order (never cycled), status colors (good/critical) for
pass/fail encoding rather than a hue borrowed from the categorical set,
equity and drawdown as stacked subplots sharing an x-axis rather than a
dual-axis chart (different units, never one chart/two y-scales), and win/
loss on the trade scatter carried by marker SHAPE as well as color so it
isn't color-alone. Light theme only: this is a one-off artifact generated
fresh per experiment and opened once during analysis, not a persisted
dashboard revisited across sessions, so a light/dark toggle was scoped out
as low value for the added complexity.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots

from backtest.metrics import profit_factor, win_rate

# dataviz skill's validated default palette (light mode)
_CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
_GOOD = "#0ca30c"
_CRITICAL = "#d03b3b"
_SURFACE = "#fcfcfb"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _symbol_color_map(symbols: list[str]) -> dict[str, str]:
    """Fixed categorical order — same symbol set always gets the same
    colors regardless of the order it's encountered in."""
    return {sym: _CATEGORICAL[i % len(_CATEGORICAL)] for i, sym in enumerate(sorted(set(symbols)))}


def _monthly_windows(trades: list[dict]) -> list[dict]:
    """Bucket trades by entry-month and score each independently — a
    diagnostic view of performance stability over time. Walk-forward
    retrain boundaries aren't available here (cli experiment always
    replays via ReplayAggregator, which doesn't record retrain_events),
    so calendar months stand in as the window unit."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        buckets[t["entry_time"].strftime("%Y-%m")].append(t)
    return [
        {
            "month": month,
            "trade_count": len(ts),
            "profit_factor": profit_factor(ts),
            "win_rate": win_rate(ts),
            "pnl": sum(t["pnl"] for t in ts),
        }
        for month, ts in sorted(buckets.items())
    ]


def _drawdown_series(equity_curve: list[tuple[datetime, float]]) -> list[float]:
    """Running drawdown as a negative percentage (0 at a new peak)."""
    if not equity_curve:
        return []
    peak = equity_curve[0][1]
    out = []
    for _, equity in equity_curve:
        peak = max(peak, equity)
        out.append(-(peak - equity) / peak * 100 if peak > 0 else 0.0)
    return out


def _base_layout(height: int, showlegend: bool = False) -> dict:
    return dict(
        height=height,
        paper_bgcolor=_SURFACE,
        plot_bgcolor=_SURFACE,
        font=dict(family=_FONT, color=_TEXT_PRIMARY, size=13),
        margin=dict(l=60, r=30, t=40, b=40),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor=_SURFACE, font_color=_TEXT_PRIMARY, bordercolor=_BASELINE),
    )


def _equity_drawdown_figure(equity_curve: list[tuple[datetime, float]]) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.65, 0.35], subplot_titles=("Equity", "Drawdown"),
    )
    if equity_curve:
        timestamps = [t for t, _ in equity_curve]
        equity = [e for _, e in equity_curve]
        drawdown = _drawdown_series(equity_curve)
        fig.add_trace(
            go.Scatter(x=timestamps, y=equity, mode="lines", name="Equity",
                       line=dict(color=_CATEGORICAL[0], width=2),
                       hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=timestamps, y=drawdown, mode="lines", name="Drawdown",
                       line=dict(color=_CRITICAL, width=2),
                       fill="tozeroy", fillcolor="rgba(208,59,59,0.12)",
                       hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>"),
            row=2, col=1,
        )
    fig.update_layout(**_base_layout(height=480))
    fig.update_yaxes(title_text="Equity ($)", gridcolor=_GRIDLINE, row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", gridcolor=_GRIDLINE, row=2, col=1)
    fig.update_xaxes(gridcolor=_GRIDLINE, row=2, col=1)
    return fig


def _monthly_pf_figure(windows: list[dict]) -> go.Figure:
    fig = go.Figure()
    if windows:
        capped_pf = [min(w["profit_factor"], 5.0) if w["profit_factor"] != float("inf") else 5.0
                     for w in windows]
        colors = [_GOOD if w["profit_factor"] >= 1.0 else _CRITICAL for w in windows]
        fig.add_trace(go.Bar(
            x=[w["month"] for w in windows], y=capped_pf, marker_color=colors,
            customdata=[[w["trade_count"], w["profit_factor"]] for w in windows],
            hovertemplate=(
                "%{x}<br>PF %{customdata[1]:.2f} (%{customdata[0]} trades)<extra></extra>"
            ),
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color=_BASELINE)
    fig.update_layout(**_base_layout(height=320))
    fig.update_yaxes(title_text="Profit factor (capped at 5)", gridcolor=_GRIDLINE)
    fig.update_xaxes(title_text="Month", gridcolor=_GRIDLINE)
    return fig


def _trade_pnl_scatter(trades: list[dict]) -> go.Figure:
    fig = go.Figure()
    if trades:
        symbols = sorted({t["symbol"] for t in trades})
        color_map = _symbol_color_map(symbols)
        for sym in symbols:
            sym_trades = [t for t in trades if t["symbol"] == sym]
            fig.add_trace(go.Scatter(
                x=[t["exit_time"] for t in sym_trades],
                y=[t["pnl"] for t in sym_trades],
                mode="markers", name=sym,
                marker=dict(
                    color=color_map[sym], size=9,
                    symbol=["circle" if t["pnl"] > 0 else "x" for t in sym_trades],
                    line=dict(width=1, color=_SURFACE),
                ),
                hovertemplate=f"{sym}<br>" + "%{x}<br>$%{y:,.2f}<extra></extra>",
            ))
        fig.add_hline(y=0, line_color=_BASELINE)
    fig.update_layout(**_base_layout(height=380, showlegend=bool(trades)))
    fig.update_yaxes(title_text="Trade PnL ($) — ● win / ✕ loss", gridcolor=_GRIDLINE)
    fig.update_xaxes(title_text="Exit time", gridcolor=_GRIDLINE)
    return fig


def _exit_reason_figure(exit_reasons: dict[str, int]) -> go.Figure:
    fig = go.Figure()
    if exit_reasons:
        order = sorted(exit_reasons, key=lambda k: -exit_reasons[k])
        colors = [_CATEGORICAL[i % len(_CATEGORICAL)] for i in range(len(order))]
        fig.add_trace(go.Bar(
            x=[exit_reasons[k] for k in order], y=order, orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x}<extra></extra>",
        ))
    fig.update_layout(**_base_layout(height=220))
    fig.update_xaxes(title_text="Trade count", gridcolor=_GRIDLINE)
    return fig


def _stat_cards(exp_id: str, metrics: dict) -> str:
    def card(label, value):
        return (
            f'<div style="flex:1;min-width:120px;padding:14px 16px;'
            f'background:{_SURFACE};border:1px solid {_GRIDLINE};border-radius:8px;">'
            f'<div style="font-size:12px;color:{_TEXT_SECONDARY};'
            f'text-transform:uppercase;letter-spacing:.04em;">{label}</div>'
            f'<div style="font-size:22px;color:{_TEXT_PRIMARY};font-weight:600;'
            f'margin-top:4px;">{value}</div></div>'
        )

    pf = metrics["profit_factor"]
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
    cards = [
        card("Trades", metrics["trade_count"]),
        card("Win rate", f"{metrics['win_rate']:.1%}"),
        card("Profit factor", pf_str),
        card("Expectancy", f"${metrics['expectancy']:.2f}"),
        card("Max drawdown", f"{metrics['max_drawdown']:.2%}"),
        card("Sharpe", f"{metrics['sharpe_ratio']:.2f}"),
    ]
    return (
        f'<h1 style="font-family:{_FONT};color:{_TEXT_PRIMARY};margin-bottom:4px;">{exp_id}</h1>'
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 28px;">'
        + "".join(cards) + "</div>"
    )


def build_report_html(exp_id: str, metrics: dict, trades: list[dict],
                       equity_curve: list[tuple[datetime, float]]) -> str:
    """Self-contained HTML string — Plotly bundled inline so the file
    works fully offline (no CDN dependency)."""
    windows = _monthly_windows(trades)

    figs = [
        _equity_drawdown_figure(equity_curve),
        _monthly_pf_figure(windows),
        _trade_pnl_scatter(trades),
        _exit_reason_figure(metrics.get("exit_reason_breakdown", {})),
    ]
    chart_html = [
        to_html(fig, include_plotlyjs=(i == 0), full_html=False, config={"displaylogo": False})
        for i, fig in enumerate(figs)
    ]

    body = (
        f'<div style="max-width:960px;margin:0 auto;padding:32px 24px;'
        f'font-family:{_FONT};background:#f9f9f7;">'
        + _stat_cards(exp_id, metrics)
        + "".join(f'<div style="margin-bottom:24px;">{c}</div>' for c in chart_html)
        + "</div>"
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{exp_id} — experiment report</title></head>"
        f"<body>{body}</body></html>"
    )
