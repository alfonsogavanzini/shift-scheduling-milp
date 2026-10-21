"""Análisis de sensibilidad: cada función resuelve varios escenarios y devuelve un DataFrame."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .instance import Instance, generate_instance
from .model import CONSTRAINT_FAMILIES, Config, Solution, solve
from .validate import validate

FAMILY_LABELS = {
    "rest": "Descanso entre turnos (A→M, N→M, N→A)",
    "consec_days": "Máx. 5 días seguidos",
    "consec_nights": "Máx. 3 noches seguidas",
    "weekly_cap": "Tope semanal de turnos",
    "min_contract": "Mínimo de turnos del contrato",
    "unavailability": "Vacaciones / bajas",
}


def _surplus(inst: Instance, sol: Solution) -> int | None:
    """Personas-turno asignadas por encima de la demanda (el modelo no penaliza el exceso)."""
    if not sol.roster:
        return None
    assigned: dict[tuple[int, str], int] = {}
    for (_, d), s in sol.roster.items():
        assigned[d, s] = assigned.get((d, s), 0) + 1
    return sum(max(0, n - inst.demand[k]) for k, n in assigned.items())


def _row(inst: Instance, sol: Solution, **extra) -> dict:
    return {
        **extra,
        "status": sol.status,
        "objective": sol.objective,
        "pref_cost": sol.pref_cost,
        "fairness_spread": sol.fairness_spread,
        "understaffed": sol.understaffed,
        "surplus": _surplus(inst, sol),
        "seconds": round(sol.seconds, 2),
        "gap": sol.gap,
    }


def demand_sensitivity(inst: Instance, pcts=(0, 5, 10, 15, 20), cfg: Config = Config()) -> pd.DataFrame:
    """¿Qué pasa si sube la demanda total un X %? (+1 persona en celdas día/turno elegidas al azar)"""
    rows = []
    for pct in pcts:
        scen = inst.with_demand_increase(pct)
        rows.append(_row(scen, solve(scen, cfg), demand_increase_pct=pct, total_demand=scen.total_demand))
    return pd.DataFrame(rows)


def staff_sensitivity(pool: Instance, sizes=range(9, 18), cfg: Config = Config()) -> pd.DataFrame:
    """¿Qué pasa si se añade o se quita gente? La plantilla de tamaño n son los n primeros del pool."""
    rows = []
    for n in sizes:
        scen = pool.with_n_employees(n)
        sol = solve(scen, cfg)
        n_pt = sum(e.contract == "PT" for e in scen.employees)
        rows.append(_row(scen, sol, employees=n, part_time=n_pt))
    return pd.DataFrame(rows)


def constraint_cost(inst: Instance, cfg: Config = Config()) -> pd.DataFrame:
    """Coste de cada familia de restricciones: se resuelve el modelo ignorándola y se mide cuánto mejora el objetivo.

    No es un precio sombra (el modelo es entero), sino una medida directa y fácil de interpretar:
    "si no existiera esta regla, ahorraríamos X puntos de coste". Un 0 significa que la regla no
    es la que limita la solución óptima (en esa instancia), no que sea inútil.
    """
    base = solve(inst, cfg)
    rows = [_row(inst, base, relaxed="(ninguna: modelo completo)", delta_vs_base=0.0)]
    for fam in CONSTRAINT_FAMILIES:
        sol = solve(inst, replace(cfg, relax=frozenset({fam})))
        delta = round(sol.objective - base.objective, 6)
        rows.append(_row(inst, sol, relaxed=FAMILY_LABELS[fam], delta_vs_base=delta))

    # La equidad no es una restricción sino un término del objetivo: se resuelve con peso 0 y la
    # solución se revalúa con los pesos reales (validador) para que sea comparable.
    sol = solve(inst, replace(cfg, w_fair=0.0))
    rep = validate(inst, sol.roster, cfg)
    row = _row(inst, sol, relaxed="Equidad de fines de semana (peso = 0)", delta_vs_base=round(rep.objective - base.objective, 6))
    row.update(objective=rep.objective, pref_cost=rep.pref_cost, fairness_spread=rep.fairness_spread, understaffed=rep.understaffed)
    rows.append(row)
    return pd.DataFrame(rows)


def formulation_comparison(seeds=(7, 11, 23), pcts=(0, 5, 10), cfg: Config = Config()) -> pd.DataFrame:
    """Formulación "pairwise" frente a "clique" para las restricciones de descanso."""
    rows = []
    for seed in seeds:
        base = generate_instance(seed=seed)
        for pct in pcts:
            inst = base.with_demand_increase(pct)
            for form in ("pairwise", "clique"):
                c = replace(cfg, rest_formulation=form)
                lp = solve(inst, replace(c, integer=False))
                mip = solve(inst, c)
                rows.append(
                    {
                        "seed": seed,
                        "demand_increase_pct": pct,
                        "formulation": form,
                        "constraints": mip.n_constraints,
                        "lp_bound": lp.objective,
                        "milp_objective": mip.objective,
                        "milp_seconds": round(mip.seconds, 2),
                        "nodes": mip.nodes,
                    }
                )
    return pd.DataFrame(rows)


def roster_summary(inst: Instance, sol: Solution, cfg: Config = Config()) -> pd.DataFrame:
    """Resumen por empleado: turnos totales y por tipo, fines de semana, penalización de preferencias."""
    rows = []
    for e, emp in enumerate(inst.employees):
        mine = {d: s for (ee, d), s in sol.roster.items() if ee == e}
        rows.append(
            {
                "employee": emp.name,
                "contract": emp.contract,
                "shifts": len(mine),
                "min_contract": emp.min_shifts,
                "M": sum(s == "M" for s in mine.values()),
                "A": sum(s == "A" for s in mine.values()),
                "N": sum(s == "N" for s in mine.values()),
                "weekend_shifts": sum(inst.is_weekend(d) for d in mine),
                "pref_penalty": sum(inst.penalty(e, d, s) for d, s in mine.items()),
            }
        )
    return pd.DataFrame(rows)


def roster_check(inst: Instance, sol: Solution, cfg: Config = Config()) -> dict:
    """Pasa la solución por el validador independiente y compara el objetivo."""
    rep = validate(inst, sol.roster, cfg)
    return {
        "feasible": rep.feasible,
        "violations": rep.violations,
        "objective_solver": sol.objective,
        "objective_validator": rep.objective,
        "match": abs(sol.objective - rep.objective) < 1e-6,
    }
