"""Tests del modelo, del validador y del generador de instancias.

Los tests de modelo usan una instancia pequeña (2 semanas) para que la suite tarde pocos segundos.
"""
from dataclasses import replace

import pytest

from rostering.instance import SHIFTS, Employee, Instance, generate_instance
from rostering.model import CONSTRAINT_FAMILIES, Config, solve
from rostering.validate import validate

# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def small() -> Instance:
    return generate_instance(n_employees=10, n_weeks=2, seed=7, weekday_demand=(2, 2, 1), weekend_demand=(1, 1, 1))


@pytest.fixture(scope="module")
def small_solution(small):
    return solve(small, Config(time_limit=60))


def tiny_instance(n_days=7, demand=None, n_employees=1) -> Instance:
    """Instancia mínima hecha a mano, sin vacaciones ni peticiones, para probar reglas concretas."""
    emp = Employee("T", "FT", 0, 7, {"M": 0, "A": 0, "N": 0})
    employees = [replace(emp, name=f"T{i}") for i in range(n_employees)]
    dem = {(d, s): 0 for d in range(n_days) for s in SHIFTS}
    dem.update(demand or {})
    return Instance(employees, n_days, dem)


# ---------------------------------------------------------------- instancias


def test_generator_is_deterministic():
    a, b = generate_instance(seed=3), generate_instance(seed=3)
    assert a.demand == b.demand and a.unavailable == b.unavailable and a.day_off_requests == b.day_off_requests
    assert [e.shift_penalty for e in a.employees] == [e.shift_penalty for e in b.employees]


def test_staff_sizes_are_prefixes_of_the_pool():
    pool = generate_instance(n_employees=17)
    direct = generate_instance(n_employees=13)
    cut = pool.with_n_employees(13)
    assert cut.employees == direct.employees
    assert cut.unavailable == direct.unavailable
    assert cut.day_off_requests == direct.day_off_requests


def test_json_roundtrip(tmp_path):
    inst = generate_instance()
    inst.to_json(tmp_path / "i.json")
    back = Instance.from_json(tmp_path / "i.json")
    assert back.employees == inst.employees
    assert back.demand == inst.demand
    assert back.unavailable == inst.unavailable
    assert back.day_off_requests == inst.day_off_requests


def test_demand_increase_is_nested_and_exact():
    inst = generate_instance()
    d5, d10 = inst.with_demand_increase(5), inst.with_demand_increase(10)
    assert d5.total_demand == inst.total_demand + round(inst.total_demand * 0.05)
    assert all(d10.demand[k] >= d5.demand[k] >= inst.demand[k] for k in inst.demand)


def test_contract_minimum_is_reachable_for_everyone():
    """Sin el prorrateo por vacaciones esta instancia era infactible: se comprueba que no se rompa."""
    inst = generate_instance()
    for e, emp in enumerate(inst.employees):
        free_weeks = [sum((e, d) not in inst.unavailable for d in range(7 * w, 7 * w + 7)) for w in range(inst.n_weeks)]
        capacity = sum(min(emp.max_per_week, n) for n in free_weeks)
        assert emp.min_shifts <= capacity, emp.name


# -------------------------------------------------------------------- modelo


def test_base_instance_is_solved_to_optimality_with_full_coverage():
    inst = generate_instance()
    sol = solve(inst, Config(time_limit=60))
    assert sol.optimal and sol.understaffed == 0
    assert validate(inst, sol.roster).feasible


def test_solution_is_feasible_and_validator_objective_matches(small, small_solution):
    rep = validate(small, small_solution.roster)
    assert small_solution.optimal
    assert rep.feasible, rep.violations
    assert rep.objective == pytest.approx(small_solution.objective)
    assert rep.pref_cost == pytest.approx(small_solution.pref_cost)
    assert rep.understaffed == small_solution.understaffed


def test_lp_relaxation_is_a_lower_bound(small, small_solution):
    lp = solve(small, Config(integer=False))
    assert lp.objective <= small_solution.objective + 1e-6


def test_both_rest_formulations_give_the_same_optimum(small, small_solution):
    pairwise = solve(small, Config(rest_formulation="pairwise", time_limit=60))
    assert pairwise.objective == pytest.approx(small_solution.objective)


def test_clique_relaxation_is_at_least_as_tight_as_pairwise(small):
    clique = solve(small, Config(integer=False, rest_formulation="clique"))
    pairwise = solve(small, Config(integer=False, rest_formulation="pairwise"))
    assert clique.objective >= pairwise.objective - 1e-6


@pytest.mark.parametrize("family", CONSTRAINT_FAMILIES)
def test_relaxing_a_constraint_never_makes_the_optimum_worse(small, small_solution, family):
    relaxed = solve(small, Config(relax=frozenset({family}), time_limit=60))
    assert relaxed.objective <= small_solution.objective + 1e-6


def test_unknown_family_is_rejected(small):
    with pytest.raises(ValueError):
        solve(small, Config(relax=frozenset({"no_existe"})))


def test_rest_rule_binds_on_a_handcrafted_case():
    """Un único empleado, noche el día 0 y mañana el día 1: con la regla de descanso no se puede cubrir todo."""
    inst = tiny_instance(n_days=7, demand={(0, "N"): 1, (1, "M"): 1})
    strict = solve(inst, Config())
    relaxed = solve(inst, Config(relax=frozenset({"rest"})))
    assert strict.understaffed == 1
    assert relaxed.understaffed == 0


def test_max_consecutive_days_binds_on_a_handcrafted_case():
    """Con 6 días seguidos de demanda y un solo empleado, el máximo de 5 días seguidos obliga a dejar uno sin cubrir."""
    inst = tiny_instance(n_days=7, demand={(d, "M"): 1 for d in range(6)})
    assert solve(inst, Config()).understaffed == 1
    assert solve(inst, Config(relax=frozenset({"consec_days"}))).understaffed == 0


def test_excess_demand_creates_understaffing(small):
    sol = solve(small.with_demand_increase(60), Config(time_limit=60))
    assert sol.understaffed > 0


# ----------------------------------------------------------------- validador


def test_validator_accepts_an_empty_roster_when_no_contract_minimum():
    inst = tiny_instance()
    assert validate(inst, {}).feasible


@pytest.mark.parametrize(
    "roster, expected",
    [
        ({(0, 0): "N", (0, 1): "M"}, "descanso"),
        ({(0, 0): "A", (0, 1): "M"}, "descanso"),
        ({(0, 0): "N", (0, 1): "A"}, "descanso"),
        ({(0, d): "M" for d in range(6)}, "consec_days"),
        ({(0, d): "N" for d in range(4)}, "consec_nights"),
    ],
)
def test_validator_detects_violations(roster, expected):
    inst = tiny_instance()
    inst.employees[0] = replace(inst.employees[0], max_per_week=7)
    rep = validate(inst, roster)
    assert not rep.feasible
    assert any(expected in v for v in rep.violations), rep.violations


def test_validator_detects_weekly_cap_and_vacation():
    inst = tiny_instance()
    inst.employees[0] = replace(inst.employees[0], max_per_week=2, min_shifts=3)
    inst.unavailable.add((0, 6))
    rep = validate(inst, {(0, 0): "M", (0, 2): "M", (0, 4): "M", (0, 6): "M"})
    text = " ".join(rep.violations)
    assert "semana" in text and "mínimo del contrato" not in text  # 4 turnos >= mínimo 3
    assert "vacaciones" in text


def test_validator_counts_understaffing_and_penalties():
    inst = tiny_instance(demand={(0, "M"): 2, (1, "N"): 1})
    inst.employees[0] = replace(inst.employees[0], shift_penalty={"M": 1, "A": 1, "N": 3})
    rep = validate(inst, {(0, 0): "M"})
    assert rep.understaffed == 2  # falta 1 en (0,M) y 1 en (1,N)
    assert rep.pref_cost == 1
    assert rep.objective == pytest.approx(1 + Config().w_under * 2)


def test_validator_rejects_invalid_shift():
    inst = tiny_instance()
    rep = validate(inst, {(0, 0): "X"})
    assert not rep.feasible
