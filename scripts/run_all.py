"""Reproduce todos los resultados del README:  python scripts/run_all.py

Escribe en results/ el cuadrante base, los análisis de sensibilidad (CSV) y las figuras (PNG),
y guarda la instancia base en data/instance_base.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from rostering import analysis, plots  # noqa: E402
from rostering.instance import generate_instance  # noqa: E402
from rostering.model import Config, solve  # noqa: E402

RESULTS = ROOT / "results"
DATA = ROOT / "data"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    cfg = Config()

    pool = generate_instance(n_employees=17)  # plantillas de 9..17 = prefijos de este pool
    base = pool.with_n_employees(13)
    base.to_json(DATA / "instance_base.json")

    print("1/5 Modelo base (relajación LP y MILP)")
    lp = solve(base, Config(integer=False))
    sol = solve(base, cfg)
    check = analysis.roster_check(base, sol, cfg)
    assert check["feasible"] and check["match"], check
    summ = analysis.roster_summary(base, sol, cfg)
    summ.to_csv(RESULTS / "roster_summary.csv", index=False)
    pd.DataFrame(
        [{"employee": base.employees[e].name, "day": d + 1, "shift": s} for (e, d), s in sorted(sol.roster.items())]
    ).to_csv(RESULTS / "roster_base.csv", index=False)
    plots.plot_roster(base, sol, RESULTS / "roster_base.png")
    summary = {
        "employees": base.n_employees,
        "days": base.n_days,
        "total_demand": base.total_demand,
        "binary_variables": base.n_employees * base.n_days * 3,
        "constraints": sol.n_constraints,
        "lp_relaxation_bound": lp.objective,
        "milp_objective": sol.objective,
        "milp_status": sol.status,
        "milp_gap": sol.gap,
        "milp_seconds": round(sol.seconds, 2),
        "pref_cost": sol.pref_cost,
        "fairness_spread": sol.fairness_spread,
        "understaffed": sol.understaffed,
        "validator": check,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("2/5 Sensibilidad a la demanda")
    dem = analysis.demand_sensitivity(base, cfg=cfg)
    dem.to_csv(RESULTS / "sensitivity_demand.csv", index=False)
    plots.plot_demand(dem, RESULTS / "sensitivity_demand.png")

    print("3/5 Sensibilidad al tamaño de plantilla")
    staff = analysis.staff_sensitivity(pool, cfg=cfg)
    staff.to_csv(RESULTS / "sensitivity_staff.csv", index=False)
    plots.plot_staff(staff, RESULTS / "sensitivity_staff.png")

    print("4/5 Coste de cada familia de restricciones (demanda base y +5 %)")
    for pct in (0, 5):
        cc = analysis.constraint_cost(base.with_demand_increase(pct), cfg)
        cc.insert(0, "demand_increase_pct", pct)
        cc.to_csv(RESULTS / f"constraint_cost_demand_plus{pct}.csv", index=False)

    print("5/5 Formulación de descanso: pairwise vs clique")
    analysis.formulation_comparison(cfg=cfg).to_csv(RESULTS / "formulation_comparison.csv", index=False)
    print("Listo. Resultados en", RESULTS)


if __name__ == "__main__":
    main()
