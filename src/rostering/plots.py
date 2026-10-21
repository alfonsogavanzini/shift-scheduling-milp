"""Gráficos estáticos (matplotlib). Paleta categórica validada (azul, naranja, aguamarina) sobre
fondo claro; cada celda del cuadrante lleva además su letra, de modo que el color nunca es el único canal."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from .instance import Instance
from .model import Solution

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = {"M": "#2a78d6", "A": "#eb6834", "N": "#1baf7a"}  # slots 1-3, validados all-pairs
OFF = "#f0efec"
VACATION = "#c3c2b7"
BAR = "#2a78d6"
DAYS = ["L", "M", "X", "J", "V", "S", "D"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK_2)
    ax.xaxis.label.set_color(INK_2)


def plot_roster(inst: Instance, sol: Solution, path: str | Path, title: str = "Cuadrante óptimo (4 semanas)") -> None:
    n_e, n_d = inst.n_employees, inst.n_days
    fig, ax = plt.subplots(figsize=(13, 0.42 * n_e + 1.9), facecolor=SURFACE)
    _style(ax)
    for e in range(n_e):
        for d in range(n_d):
            s = sol.roster.get((e, d))
            vac = (e, d) in inst.unavailable
            color = SERIES[s] if s else (VACATION if vac else OFF)
            ax.add_patch(Rectangle((d, n_e - 1 - e), 1, 1, facecolor=color, edgecolor=SURFACE, linewidth=2))
            if s:
                ax.text(d + 0.5, n_e - 1 - e + 0.5, s, ha="center", va="center", fontsize=9,
                        color="white" if s == "M" else INK, fontweight="bold")
            elif vac:
                ax.text(d + 0.5, n_e - 1 - e + 0.5, "V", ha="center", va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, n_d)
    ax.set_ylim(0, n_e)
    ax.set_yticks([n_e - 1 - e + 0.5 for e in range(n_e)])
    ax.set_yticklabels([f"{emp.name} ({emp.contract})" for emp in inst.employees])
    ax.set_xticks([d + 0.5 for d in range(n_d)])
    ax.set_xticklabels([DAYS[d % 7] for d in range(n_d)])
    for w in range(inst.n_weeks):
        ax.text(7 * w + 3.5, n_e + 0.35, f"Semana {w + 1}", ha="center", va="bottom", fontsize=9, color=INK_2)
        if w:
            ax.axvline(7 * w, color=INK_2, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=34)
    handles = [Patch(facecolor=SERIES["M"], label="Mañana (M)"), Patch(facecolor=SERIES["A"], label="Tarde (A)"),
               Patch(facecolor=SERIES["N"], label="Noche (N)"), Patch(facecolor=OFF, label="Libre"),
               Patch(facecolor=VACATION, label="Vacaciones / baja (V)")]
    leg = ax.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _bars(ax, x, y, title: str, ylabel: str) -> None:
    _style(ax)
    ax.bar(range(len(x)), y, color=BAR, width=0.62)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([str(v) for v in x])
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    top = max(y) if max(y) > 0 else 1
    ax.set_ylim(0, top * 1.15)
    for i, v in enumerate(y):
        ax.text(i, v + top * 0.02, f"{v:g}", ha="center", va="bottom", fontsize=9, color=INK_2)


def plot_staff(df: pd.DataFrame, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    _bars(axes[0], df["employees"], df["understaffed"], "Turnos sin cubrir según tamaño de plantilla",
          "personas-turno sin cubrir")
    _bars(axes[1], df["employees"], df["surplus"].fillna(0), "Exceso de personas-turno sobre la demanda",
          "personas-turno en exceso")
    for ax in axes:
        ax.set_xlabel("número de empleados")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_demand(df: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    _bars(ax, [f"+{int(p)}%" for p in df["demand_increase_pct"]], df["understaffed"],
          "Turnos sin cubrir si sube la demanda (13 empleados)", "personas-turno sin cubrir")
    ax.set_xlabel("aumento de la demanda total")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
