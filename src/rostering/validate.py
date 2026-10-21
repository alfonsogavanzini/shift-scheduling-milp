"""Validador independiente de planificaciones.

No usa PuLP ni comparte código con el modelo: recorre el cuadrante día a día y comprueba cada
regla "a mano". Sirve para (1) verificar que lo que devuelve el solver es realmente factible y
(2) recalcular el valor objetivo por otra vía y comprobar que coincide.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .instance import SHIFTS, Instance
from .model import FORBIDDEN, Config

Roster = dict[tuple[int, int], str]  # (empleado, día) -> turno; los días libres no aparecen


@dataclass
class Report:
    violations: list[str] = field(default_factory=list)
    pref_cost: float = 0.0
    fairness_spread: float = 0.0
    understaffed: int = 0
    objective: float = 0.0

    @property
    def feasible(self) -> bool:
        return not self.violations


def validate(inst: Instance, roster: Roster, cfg: Config = Config()) -> Report:
    rep = Report()
    relax = cfg.relax
    n_days = inst.n_days

    def shift_of(e: int, d: int) -> str | None:
        return roster.get((e, d))

    for (e, d), s in roster.items():
        if s not in SHIFTS:
            rep.violations.append(f"turno inválido {s!r} (empleado {e}, día {d})")
        if not (0 <= e < inst.n_employees and 0 <= d < n_days):
            rep.violations.append(f"asignación fuera de rango (empleado {e}, día {d})")
    if rep.violations:
        return rep

    for e, emp in enumerate(inst.employees):
        # --- indisponibilidad
        if "unavailability" not in relax:
            for d in range(n_days):
                if shift_of(e, d) and (e, d) in inst.unavailable:
                    rep.violations.append(f"{emp.name} trabaja el día {d} estando de vacaciones/baja")

        # --- descanso entre turnos consecutivos
        if "rest" not in relax:
            for d in range(n_days - 1):
                pair = (shift_of(e, d), shift_of(e, d + 1))
                if pair in FORBIDDEN:
                    rep.violations.append(f"{emp.name}: descanso insuficiente {pair[0]}->{pair[1]} (días {d}-{d + 1})")

        # --- rachas de días trabajados / de noches (se recorren las rachas, no ventanas)
        for family, limit, pred in (
            ("consec_days", cfg.max_consec_days, lambda d: shift_of(e, d) is not None),
            ("consec_nights", cfg.max_consec_nights, lambda d: shift_of(e, d) == "N"),
        ):
            if family in relax:
                continue
            run = 0
            for d in range(n_days):
                run = run + 1 if pred(d) else 0
                if run > limit:
                    rep.violations.append(f"{emp.name}: racha de {run} ({family}) que termina el día {d}, máx {limit}")
                    break

        # --- tope semanal
        if "weekly_cap" not in relax:
            for w in range(inst.n_weeks):
                worked = sum(1 for d in range(7 * w, 7 * w + 7) if shift_of(e, d))
                if worked > emp.max_per_week:
                    rep.violations.append(f"{emp.name}: {worked} turnos en la semana {w + 1}, máx {emp.max_per_week}")

        # --- mínimo del contrato
        if "min_contract" not in relax:
            total = sum(1 for d in range(n_days) if shift_of(e, d))
            if total < emp.min_shifts:
                rep.violations.append(f"{emp.name}: {total} turnos, mínimo del contrato {emp.min_shifts}")

    # --- cobertura (déficit = parte del objetivo, no una violación)
    for d in range(n_days):
        for s in SHIFTS:
            assigned = sum(1 for e in range(inst.n_employees) if shift_of(e, d) == s)
            rep.understaffed += max(0, inst.demand[d, s] - assigned)

    # --- objetivo recalculado desde cero
    rep.pref_cost = sum(inst.penalty(e, d, s) for (e, d), s in roster.items())
    weekend_load = [
        sum(1 for d in range(n_days) if inst.is_weekend(d) and shift_of(e, d))
        for e, emp in enumerate(inst.employees)
        if emp.contract == cfg.fair_group
    ]
    rep.fairness_spread = float(max(weekend_load) - min(weekend_load)) if weekend_load else 0.0
    rep.objective = rep.pref_cost + cfg.w_fair * rep.fairness_spread + cfg.w_under * rep.understaffed
    return rep
