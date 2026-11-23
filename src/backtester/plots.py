"""Static matplotlib figures (PNG) with a provenance footer on every chart.

Every figure carries: strategy, data mode, date range, execution convention,
cost scenario and sample status (in-sample / OOS / hold-out / exploratory).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats  # noqa: E402

from .performance import drawdown_series, monthly_table, period_returns  # noqa: E402

# reference categorical palette (fixed order) and text inks
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C7, C8 = "#6250d6", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#c43c3b", "#f2f1ee", "#1f63b5"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#2a78d6", "#0d3c75"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.6,
})


@dataclass(frozen=True)
class Stamp:
    strategy: str
    data_mode: str
    date_range: str
    execution: str = "signal at close t, fill at next tradable open"
    costs: str = ""
    sample: str = ""

    def text(self) -> str:
        mode = self.data_mode.upper()
        if self.data_mode == "synthetic":
            mode = "SYNTHETIC DATA (not market data)"
        return (f"{self.strategy} | {mode} | {self.date_range} | {self.execution} | "
                f"costs: {self.costs} | sample: {self.sample}")


def _finish(fig: plt.Figure, path: str | Path, stamp: Stamp | None,
            data: pd.DataFrame | pd.Series | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if stamp is not None:
        fig.text(0.01, 0.005, stamp.text(), fontsize=6.5, color=INK2, ha="left", va="bottom",
                 wrap=True)
    fig.tight_layout(rect=(0, 0.035 if stamp else 0, 1, 1))
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    if data is not None:
        dpath = p.parent.parent / "tables" / "figure_data" / (p.stem + ".csv")
        dpath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(dpath)
    return p


def _pct(ax: plt.Axes, axis: str = "y") -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------- panels
def panel_equity(ax: plt.Axes, r: pd.Series, bench: pd.Series | None = None,
                 label: str = "Strategy", bench_label: str = "Benchmark", log: bool = True,
                 shade: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> None:
    w = (1 + r).cumprod()
    ax.plot(w.index, w.to_numpy(), color=C1, label=label)
    if bench is not None:
        b = bench.reindex(r.index).fillna(0.0)
        ax.plot(b.index, (1 + b).cumprod().to_numpy(), color=C2, label=bench_label)
    if shade is not None:
        ax.axvspan(shade[0], shade[1], color="#f2f1ee", zorder=0,  # type: ignore[arg-type]
                   label="out-of-sample")
    if log:
        ax.set_yscale("log")
    ax.set_title("Growth of 1 (log scale)" if log else "Growth of 1")
    ax.legend(loc="upper left")


def panel_drawdown(ax: plt.Axes, r: pd.Series, bench: pd.Series | None = None) -> None:
    dd = drawdown_series(r)
    ax.fill_between(dd.index, dd.to_numpy(), 0, color=C1, alpha=0.35, linewidth=0,
                    label="Strategy")
    ax.plot(dd.index, dd.to_numpy(), color=C1, linewidth=1.0)
    if bench is not None:
        bd = drawdown_series(bench.reindex(r.index).fillna(0.0))
        ax.plot(bd.index, bd.to_numpy(), color=C2, linewidth=1.0, label="Benchmark")
        ax.legend(loc="lower left")
    ax.set_title("Underwater (drawdown from running peak)")
    _pct(ax)


def panel_monthly(ax: plt.Axes, r: pd.Series) -> None:
    t = monthly_table(r)
    v = np.nanmax(np.abs(t.to_numpy())) if t.size else 0.1
    im = ax.imshow(t.to_numpy(), cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(t.shape[1]), [str(c) for c in t.columns])
    ax.set_yticks(range(t.shape[0]), [str(i) for i in t.index])
    ax.grid(False)
    if t.shape[0] <= 12:
        for (i, j), val in np.ndenumerate(t.to_numpy()):
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=6, color=INK)
    ax.set_title("Monthly returns (compounded)")
    plt.colorbar(im, ax=ax, format=matplotlib.ticker.PercentFormatter(1.0, 0), shrink=0.8)


def panel_rolling_sharpe(ax: plt.Axes, r: pd.Series, window: int = 126) -> None:
    m = r.rolling(window).mean()
    s = r.rolling(window).std(ddof=1)
    rs = (m / s * np.sqrt(252)).replace([np.inf, -np.inf], np.nan)
    ax.plot(rs.index, rs.to_numpy(), color=C1)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_title(f"Rolling {window}-session Sharpe (rf = 0, annualised)")


def panel_distribution(ax: plt.Axes, r: pd.Series) -> None:
    x = r.to_numpy()
    ax.hist(x, bins=80, density=True, color=C1, alpha=0.6, edgecolor="white", linewidth=0.3,
            label="daily returns")
    if len(x) > 2 and x.std() > 0:
        g = np.linspace(x.min(), x.max(), 300)
        ax.plot(g, stats.norm.pdf(g, x.mean(), x.std(ddof=1)), color=C2, label="normal fit")
    ax.set_title("Daily return distribution")
    _pct(ax, "x")
    ax.legend()


def panel_annual(ax: plt.Axes, r: pd.Series, bench: pd.Series | None = None) -> None:
    a = period_returns(r, "YE")
    xs = np.arange(len(a))
    wdt = 0.4 if bench is not None else 0.7
    ax.bar(xs - (wdt / 2 if bench is not None else 0), a.to_numpy(), width=wdt, color=C1,
           label="Strategy")
    if bench is not None:
        b = period_returns(bench.reindex(r.index).fillna(0.0), "YE")
        ax.bar(xs + wdt / 2, b.to_numpy(), width=wdt, color=C2, label="Benchmark")
        ax.legend()
    ax.set_xticks(xs, [str(d.year) for d in a.index], rotation=45)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_title("Annual returns (compounded)")
    _pct(ax)


# ---------------------------------------------------------------------- figures
def tearsheet(r: pd.Series, path: str | Path, stamp: Stamp, bench: pd.Series | None = None,
              title: str = "", shade: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> Path:
    fig, axs = plt.subplots(3, 2, figsize=(13, 11))
    panel_equity(axs[0, 0], r, bench, shade=shade)
    panel_drawdown(axs[0, 1], r, bench)
    panel_monthly(axs[1, 0], r)
    panel_rolling_sharpe(axs[1, 1], r)
    panel_distribution(axs[2, 0], r)
    panel_annual(axs[2, 1], r, bench)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", color=INK)
    data = pd.DataFrame({"strategy_return": r})
    if bench is not None:
        data["benchmark_return"] = bench.reindex(r.index)
    return _finish(fig, path, stamp, data)


def single(kind: str, r: pd.Series, path: str | Path, stamp: Stamp,
           bench: pd.Series | None = None, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.6 if kind != "monthly" else 6))
    {"equity": lambda: panel_equity(ax, r, bench),
     "drawdown": lambda: panel_drawdown(ax, r, bench),
     "monthly": lambda: panel_monthly(ax, r),
     "rolling_sharpe": lambda: panel_rolling_sharpe(ax, r),
     "distribution": lambda: panel_distribution(ax, r),
     "annual": lambda: panel_annual(ax, r, bench)}[kind]()
    if title:
        ax.set_title(title)
    data = pd.DataFrame({"strategy_return": r})
    if bench is not None:
        data["benchmark_return"] = bench.reindex(r.index)
    return _finish(fig, path, stamp, data)


def exposure(equity_frame: pd.DataFrame, path: str | Path, stamp: Stamp, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    ef = equity_frame
    ax.plot(ef.index, ef["gross_exposure"], color=C1, label="gross exposure")
    ax.plot(ef.index, ef["net_exposure"], color=C2, label="net exposure", linewidth=1.1)
    ax.set_title(title or "Gross and net exposure (fraction of equity, at each close)")
    _pct(ax)
    ax.legend(loc="lower right")
    return _finish(fig, path, stamp, ef[["gross_exposure", "net_exposure"]])


def turnover_and_costs(equity_frame: pd.DataFrame, path: str | Path, stamp: Stamp,
                       title: str = "") -> Path:
    from .performance import turnover_series

    ef = equity_frame
    to = turnover_series(ef)
    monthly_to = to.resample("ME").sum()
    comps = ["commissions", "spread_costs", "slippage_costs", "impact_costs", "borrow_fees"]
    last = ef.iloc[-1]
    vals = [float(last[c]) for c in comps]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    axs[0].bar(monthly_to.index, monthly_to.to_numpy(), width=20, color=C1)
    axs[0].set_title("One-way turnover per month (traded notional / 2 / prior equity)")
    _pct(axs[0])
    ys = np.arange(len(comps))
    axs[1].barh(ys, vals, color=[C1, C2, C3, C4, C7])
    axs[1].set_yticks(ys, ["commission", "half-spread", "slippage", "impact", "borrow fee"])
    for y, v in zip(ys, vals, strict=True):
        axs[1].text(v, y, f" {v:,.0f}", va="center", fontsize=7, color=INK)
    axs[1].set_title("Cumulative cost components (currency)")
    if title:
        fig.suptitle(title, fontweight="bold")
    data = pd.DataFrame({"monthly_turnover": monthly_to})
    return _finish(fig, path, stamp, data)


def heatmap(table: pd.DataFrame, path: str | Path, stamp: Stamp, title: str,
            xlabel: str, ylabel: str, fmt: str = "{:.2f}", diverging: bool = True) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    v = table.to_numpy(dtype=float)
    if diverging:
        lim = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1.0
        im = ax.imshow(v, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    else:
        im = ax.imshow(v, cmap=SEQUENTIAL, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns])
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    for (i, j), val in np.ndenumerate(v):
        ax.text(j, i, "invalid" if not np.isfinite(val) else fmt.format(val), ha="center",
                va="center", fontsize=8, color=INK)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)
    return _finish(fig, path, stamp, table)


def lines(series: dict[str, pd.Series], path: str | Path, stamp: Stamp, title: str,
          ylabel: str = "", log: bool = False,
          spans: list[tuple[pd.Timestamp, pd.Timestamp, str]] | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = [C1, C2, C3, C4, C7, C8]
    for (name, s), c in zip(series.items(), colors, strict=False):
        ax.plot(s.index, s.to_numpy(), color=c, label=name)
    for k, (a, b, lab) in enumerate(spans or []):
        ax.axvspan(a, b, color="#f2f1ee" if k % 2 == 0 else "#e9e8e4",  # type: ignore[arg-type]
                   zorder=0)
        ax.text(a, 1.0, lab, transform=ax.get_xaxis_transform(),  # type: ignore[arg-type]
                fontsize=6, color=INK2, va="top")
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(loc="upper left")
    return _finish(fig, path, stamp, pd.DataFrame(series))


def bars(table: pd.DataFrame, path: str | Path, stamp: Stamp, title: str,
         columns: list[str], pct: list[bool] | None = None) -> Path:
    fig, axs = plt.subplots(1, len(columns), figsize=(3.6 * len(columns), 3.8))
    axs = np.atleast_1d(axs)
    for k, (ax, col) in enumerate(zip(axs, columns, strict=True)):
        vals = table[col].astype(float).to_numpy()
        ax.bar([str(i) for i in table.index], vals, color=C1)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(col)
        if pct and pct[k]:
            _pct(ax)
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle(title, fontweight="bold")
    return _finish(fig, path, stamp, table[columns])


def architecture_diagram(path: str | Path) -> Path:
    """Event-flow diagram of the engine (drawn with matplotlib boxes)."""
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", facecolor=color,
            edgecolor="white", linewidth=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=INK)

    def arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        ax.annotate("", (x2, y2), (x1, y1), arrowprops={"arrowstyle": "->", "color": INK2,
                                                       "linewidth": 1.2})
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, fontsize=7, color=INK2,
                    ha="center")

    ax.text(0.2, 5.85, "Session t - OPEN phase", fontsize=10, fontweight="bold", color=INK)
    box(0.2, 4.6, 2.3, 0.9, "MarketData\n(open of t only)", "#dbe8f8")
    box(3.2, 4.6, 2.6, 0.9, "SimulatedExecution\npending orders -> fills", "#fde2d6")
    box(6.6, 4.6, 2.4, 0.9, "FillEvent\n(itemised costs)", "#fde2d6")
    box(9.6, 4.6, 2.2, 0.9, "Portfolio.apply_fill\ncash, lots, P&L", "#d4f0e5")
    arrow(2.5, 5.05, 3.2, 5.05, "MarketEvent(OPEN)")
    arrow(5.8, 5.05, 6.6, 5.05)
    arrow(9.0, 5.05, 9.6, 5.05, "applied now")

    ax.text(0.2, 3.75, "Session t - CLOSE phase", fontsize=10, fontweight="bold", color=INK)
    box(0.2, 2.5, 2.3, 0.9, "PointInTimeView\n(bars 0..t)", "#dbe8f8")
    box(3.2, 2.5, 2.6, 0.9, "Portfolio: financing,\nmark at close, reconcile", "#d4f0e5")
    box(6.6, 2.5, 2.4, 0.9, "Strategy.on_bar\n-> SignalEvents", "#efe9fb")
    box(9.6, 2.5, 2.2, 0.9, "Sizer + Constraints\n-> target weights", "#efe9fb")
    arrow(2.5, 2.95, 3.2, 2.95, "MarketEvent(CLOSE)")
    arrow(5.8, 2.95, 6.6, 2.95, "post-fill holdings")
    arrow(9.0, 2.95, 9.6, 2.95)
    box(6.6, 0.6, 5.2, 0.9, "OrderEvents (eligible_from = open of t+1) -> pending book", "#fde2d6")
    arrow(10.7, 2.5, 10.7, 1.5)
    ax.annotate("", (4.5, 4.6), (6.6, 1.05), arrowprops={
        "arrowstyle": "->", "color": C1, "linewidth": 1.4, "connectionstyle": "arc3,rad=-0.3"})
    ax.text(3.0, 1.0, "next session's OPEN phase\n(never the same close)", fontsize=8,
            color=C1)
    return _finish(fig, path, None)
