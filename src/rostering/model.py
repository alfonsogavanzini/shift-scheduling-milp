"""Modelo MILP de planificación de turnos (PuLP + HiGHS).

Variables
    x[e,d,s] ∈ {0,1}   el empleado e trabaja el turno s el día d
    u[d,s]   ≥ 0 ent.  personas que faltan para cubrir la demanda (holgura penalizada)
    wmax, wmin         máximo y mínimo de fines de semana trabajados (equidad)

Objetivo
    min  Σ pref[e,d,s]·x[e,d,s]  +  w_fair·(wmax − wmin)  +  w_under·Σ u[d,s]
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pulp

from .instance import SHIFTS, Instance

# Sucesiones de turnos prohibidas (turno del día d, turno del día d+1): descanso < 11 h.
#   Tarde (14-22) -> Mañana (06): 8 h     Noche (22-06) -> Mañana (06): 0 h     Noche -> Tarde (14): 8 h
FORBIDDEN = (("A", "M"), ("N", "M"), ("N", "A"))

CONSTRAINT_FAMILIES = (
    "rest",  # descanso mínimo entre turnos consecutivos
    "consec_days",  # máximo de días seguidos trabajando
    "consec_nights",  # máximo de noches seguidas
    "weekly_cap",  # máximo de turnos por semana natural
    "min_contract",  # mínimo de turnos del contrato
    "unavailability",  # vacaciones / bajas (días bloqueados)
)


@dataclass(frozen=True)
class Config:
    w_under: float = 100.0  # penalización por persona-turno sin cubrir
    w_fair: float = 2.0  # peso de la diferencia máx-mín de fines de semana trabajados
    max_consec_days: int = 5
    max_consec_nights: int = 3
    rest_formulation: str = "clique"  # "clique" (más fuerte) o "pairwise"
    relax: frozenset[str] = frozenset()  # familias de restricciones a ignorar
    integer: bool = True  # False -> relajación lineal
    time_limit: float = 120.0
    mip_gap: float = 1e-4
    fair_group: str = "FT"  # la equidad de fines de semana se mide entre este tipo de contrato


@dataclass
class Solution:
    status: str
    objective: float | None
    roster: dict[tuple[int, int], str] = field(default_factory=dict)  # (empleado, día) -> turno
    pref_cost: float | None = None
    fairness_spread: float | None = None
    understaffed: float | None = None
    best_bound: float | None = None
    gap: float | None = None
    seconds: float = 0.0
    nodes: int | None = None
    n_vars: int = 0
    n_constraints: int = 0

    @property
    def optimal(self) -> bool:
        return self.status == "Optimal"


def build_model(inst: Instance, cfg: Config = Config()):
    """Construye el problema PuLP. Devuelve (prob, x, u, wmax, wmin)."""
    for fam in cfg.relax:
        if fam not in CONSTRAINT_FAMILIES:
            raise ValueError(f"familia desconocida: {fam}")
    if cfg.rest_formulation not in ("clique", "pairwise"):
        raise ValueError("rest_formulation debe ser 'clique' o 'pairwise'")

    E, D = range(inst.n_employees), inst.days
    cat = pulp.LpBinary if cfg.integer else pulp.LpContinuous
    prob = pulp.LpProblem("rostering", pulp.LpMinimize)

    x = {
        (e, d, s): pulp.LpVariable(f"x_{e}_{d}_{s}", 0, 1, cat)
        for e in E
        for d in D
        for s in SHIFTS
    }
    ucat = pulp.LpInteger if cfg.integer else pulp.LpContinuous
    u = {(d, s): pulp.LpVariable(f"u_{d}_{s}", 0, None, ucat) for d in D for s in SHIFTS}

    fair = [e for e in E if inst.employees[e].contract == cfg.fair_group]
    weekend_days = [d for d in D if inst.is_weekend(d)]
    wmax = pulp.LpVariable("wmax", 0)
    wmin = pulp.LpVariable("wmin", 0)

    # ------------------------------------------------------------- objetivo
    pref = pulp.lpSum(inst.penalty(e, d, s) * x[e, d, s] for e in E for d in D for s in SHIFTS)
    prob += (
        pref
        + cfg.w_fair * (wmax - wmin)
        + cfg.w_under * pulp.lpSum(u.values())
    ), "objective"

    # --------------------------------------------------- restricciones base
    for e in E:
        for d in D:  # un turno como máximo por día
            prob += pulp.lpSum(x[e, d, s] for s in SHIFTS) <= 1, f"one_shift_{e}_{d}"

    for d in D:
        for s in SHIFTS:  # cobertura (con holgura penalizada)
            prob += (
                pulp.lpSum(x[e, d, s] for e in E) + u[d, s] >= inst.demand[d, s],
                f"cover_{d}_{s}",
            )

    # Equidad: wmax/wmin acotan los fines de semana trabajados por el grupo de referencia
    if fair and weekend_days:
        for e in fair:
            wk = pulp.lpSum(x[e, d, s] for d in weekend_days for s in SHIFTS)
            prob += wk <= wmax, f"fair_max_{e}"
            prob += wk >= wmin, f"fair_min_{e}"
    else:
        prob += wmax == 0
        prob += wmin == 0

    # ------------------------------------------- familias "relajables"
    if "rest" not in cfg.relax:
        for e in E:
            for d in range(inst.n_days - 1):
                if cfg.rest_formulation == "pairwise":
                    for s1, s2 in FORBIDDEN:
                        prob += x[e, d, s1] + x[e, d + 1, s2] <= 1, f"rest_{e}_{d}_{s1}{s2}"
                else:
                    # Cliques del grafo de conflictos (con "un turno por día" son válidas):
                    #   {A_d, N_d, M_d+1}  y  {N_d, M_d+1, A_d+1}
                    prob += (
                        x[e, d, "A"] + x[e, d, "N"] + x[e, d + 1, "M"] <= 1,
                        f"rest_c1_{e}_{d}",
                    )
                    prob += (
                        x[e, d, "N"] + x[e, d + 1, "M"] + x[e, d + 1, "A"] <= 1,
                        f"rest_c2_{e}_{d}",
                    )

    if "consec_days" not in cfg.relax:
        k = cfg.max_consec_days
        for e in E:
            for d0 in range(inst.n_days - k):  # ventanas de k+1 días: como mucho k trabajados
                prob += (
                    pulp.lpSum(x[e, d, s] for d in range(d0, d0 + k + 1) for s in SHIFTS) <= k,
                    f"consec_days_{e}_{d0}",
                )

    if "consec_nights" not in cfg.relax:
        k = cfg.max_consec_nights
        for e in E:
            for d0 in range(inst.n_days - k):
                prob += (
                    pulp.lpSum(x[e, d, "N"] for d in range(d0, d0 + k + 1)) <= k,
                    f"consec_nights_{e}_{d0}",
                )

    if "weekly_cap" not in cfg.relax:
        for e in E:
            cap = inst.employees[e].max_per_week
            for w in range(inst.n_weeks):
                prob += (
                    pulp.lpSum(x[e, d, s] for d in range(7 * w, 7 * w + 7) for s in SHIFTS) <= cap,
                    f"weekly_cap_{e}_{w}",
                )

    if "min_contract" not in cfg.relax:
        for e in E:
            prob += (
                pulp.lpSum(x[e, d, s] for d in D for s in SHIFTS) >= inst.employees[e].min_shifts,
                f"min_contract_{e}",
            )

    if "unavailability" not in cfg.relax:
        for e, d in inst.unavailable:
            prob += pulp.lpSum(x[e, d, s] for s in SHIFTS) == 0, f"unavail_{e}_{d}"

    return prob, x, u, wmax, wmin


def solve(inst: Instance, cfg: Config = Config()) -> Solution:
    """Resuelve la instancia con HiGHS y devuelve la solución con sus métricas."""
    prob, x, u, wmax, wmin = build_model(inst, cfg)
    solver = pulp.HiGHS(
        msg=False,
        timeLimit=cfg.time_limit,
        gapRel=cfg.mip_gap,
        threads=1,
        random_seed=0,
    )
    t0 = time.perf_counter()
    prob.solve(solver)
    seconds = time.perf_counter() - t0

    status = pulp.LpStatus[prob.status]
    sol = Solution(
        status=status,
        objective=None,
        seconds=seconds,
        n_vars=len(prob.variables()),
        n_constraints=len(prob.constraints),
    )

    # Información del solver (cota, gap, nodos)
    h = getattr(prob, "solverModel", None)
    if h is not None:
        try:
            info = h.getInfo()
            if cfg.integer:
                sol.best_bound = float(info.mip_dual_bound)
                sol.gap = float(info.mip_gap)
                sol.nodes = int(info.mip_node_count)
        except Exception:  # pragma: no cover - depende de la versión de highspy
            pass

    has_solution = status == "Optimal" or (
        status == "Not Solved" and pulp.value(prob.objective) is not None and cfg.integer
    )
    if not has_solution or pulp.value(prob.objective) is None:
        return sol

    sol.objective = round(float(pulp.value(prob.objective)), 6) + 0.0
    if not cfg.integer:
        sol.best_bound = sol.objective
        return sol

    for (e, d, s), var in x.items():
        if var.value() is not None and var.value() > 0.5:
            sol.roster[(e, d)] = s
    sol.pref_cost = sum(inst.penalty(e, d, s) for (e, d), s in sol.roster.items())
    sol.fairness_spread = round(wmax.value() - wmin.value(), 6) + 0.0
    sol.understaffed = float(round(sum(v.value() for v in u.values())))
    return sol
