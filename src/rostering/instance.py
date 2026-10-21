"""Datos del problema: empleados, demanda por turno, indisponibilidades y preferencias.

La instancia es sintética pero determinista (semilla fija), de modo que cualquiera puede
reproducir exactamente los resultados del README.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SHIFTS = ("M", "A", "N")  # Mañana 06-14, Tarde 14-22, Noche 22-06
SHIFT_NAMES = {"M": "Mañana", "A": "Tarde", "N": "Noche"}

# Penalización (coste de preferencia) por trabajar un turno poco deseado, y por trabajar
# un día en el que el empleado había pedido libre.
BASE_PENALTY = {"M": 1, "A": 1, "N": 3}
DISLIKED_NIGHT_PENALTY = 6
DAY_OFF_REQUEST_PENALTY = 4


@dataclass(frozen=True)
class Employee:
    name: str
    contract: str  # "FT" (jornada completa) o "PT" (parcial)
    min_shifts: int  # mínimo de turnos en todo el horizonte
    max_per_week: int  # máximo de turnos por semana natural (lunes-domingo)
    shift_penalty: dict[str, int]  # penalización por tipo de turno


@dataclass
class Instance:
    employees: list[Employee]
    n_days: int
    demand: dict[tuple[int, str], int]  # (día, turno) -> personas necesarias
    unavailable: set[tuple[int, int]] = field(default_factory=set)  # (empleado, día): no puede trabajar
    day_off_requests: set[tuple[int, int]] = field(default_factory=set)  # (empleado, día): prefiere librar

    # ------------------------------------------------------------------ helpers
    @property
    def n_employees(self) -> int:
        return len(self.employees)

    @property
    def days(self) -> range:
        return range(self.n_days)

    @property
    def n_weeks(self) -> int:
        return self.n_days // 7

    def is_weekend(self, d: int) -> bool:
        return d % 7 in (5, 6)  # el día 0 es lunes

    def penalty(self, e: int, d: int, s: str) -> int:
        """Coste de preferencia de asignar al empleado e el turno s el día d."""
        p = self.employees[e].shift_penalty[s]
        if (e, d) in self.day_off_requests:
            p += DAY_OFF_REQUEST_PENALTY
        return p

    @property
    def total_demand(self) -> int:
        return sum(self.demand.values())

    # --------------------------------------------------------------- variantes
    def with_n_employees(self, n: int) -> "Instance":
        """Primeros n empleados del conjunto (la plantilla se construye por prefijos)."""
        return Instance(
            employees=self.employees[:n],
            n_days=self.n_days,
            demand=dict(self.demand),
            unavailable={(e, d) for (e, d) in self.unavailable if e < n},
            day_off_requests={(e, d) for (e, d) in self.day_off_requests if e < n},
        )

    def with_demand_increase(self, pct: float, seed: int = 0) -> "Instance":
        """Sube la demanda total aproximadamente un pct % sumando +1 persona a celdas (día, turno).

        Las celdas se eligen con una permutación fija: el escenario del 10 % contiene al del 5 %.
        """
        cells = sorted(self.demand)
        order = np.random.default_rng(seed).permutation(len(cells))
        n_bump = round(self.total_demand * pct / 100)
        demand = dict(self.demand)
        for k in range(n_bump):
            demand[cells[order[k % len(cells)]]] += 1
        return Instance(
            employees=list(self.employees),
            n_days=self.n_days,
            demand=demand,
            unavailable=set(self.unavailable),
            day_off_requests=set(self.day_off_requests),
        )

    # ------------------------------------------------------------------- JSON
    def to_json(self, path: str | Path) -> None:
        payload = {
            "n_days": self.n_days,
            "shifts": list(SHIFTS),
            "employees": [
                {
                    "name": e.name,
                    "contract": e.contract,
                    "min_shifts": e.min_shifts,
                    "max_per_week": e.max_per_week,
                    "shift_penalty": e.shift_penalty,
                }
                for e in self.employees
            ],
            "demand": [[self.demand[(d, s)] for s in SHIFTS] for d in self.days],
            "unavailable": sorted(map(list, self.unavailable)),
            "day_off_requests": sorted(map(list, self.day_off_requests)),
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Instance":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            employees=[Employee(**e) for e in raw["employees"]],
            n_days=raw["n_days"],
            demand={(d, s): raw["demand"][d][i] for d in range(raw["n_days"]) for i, s in enumerate(SHIFTS)},
            unavailable={tuple(x) for x in raw["unavailable"]},
            day_off_requests={tuple(x) for x in raw["day_off_requests"]},
        )


def _vacation_block(i: int, n_days: int, seed: int) -> list[int]:
    """Bloque de vacaciones/baja (días bloqueados) del empleado i; lista vacía si no tiene."""
    rng = np.random.default_rng([seed, i, 1])
    if rng.random() < 0.4:
        length = int(rng.integers(3, 6))
        start = int(rng.integers(0, n_days - length))
        return list(range(start, start + length))
    return []


def _make_employee(i: int, n_weeks: int, seed: int) -> Employee:
    """Cada empleado se genera con su propio generador: el empleado i es el mismo
    independientemente del tamaño de plantilla, lo que permite comparar escenarios.

    El mínimo de turnos del contrato se prorratea por los días de vacaciones; sin ese
    ajuste, un empleado con una semana casi entera de baja no podría alcanzar su mínimo
    respetando el tope semanal y la instancia sería infactible.
    """
    rng = np.random.default_rng([seed, i])
    contract = "PT" if i % 4 == 3 else "FT"
    shifts_per_week = 4.5 if contract == "FT" else 2.5
    max_per_week = 5 if contract == "FT" else 3
    vacation_days = len(_vacation_block(i, 7 * n_weeks, seed))
    min_shifts = round(shifts_per_week * (n_weeks - vacation_days / 7))

    penalty = dict(BASE_PENALTY)
    preferred = rng.choice(SHIFTS, p=[0.4, 0.4, 0.2])
    penalty[str(preferred)] = 0
    if preferred != "N" and rng.random() < 0.3:
        penalty["N"] = DISLIKED_NIGHT_PENALTY
    return Employee(f"E{i + 1:02d}", contract, min_shifts, max_per_week, penalty)


def generate_instance(
    n_employees: int = 13,
    n_weeks: int = 4,
    seed: int = 7,
    weekday_demand: tuple[int, int, int] = (3, 3, 2),
    weekend_demand: tuple[int, int, int] = (2, 2, 2),
) -> Instance:
    """Genera una instancia sintética reproducible.

    Demanda por defecto: L-V 3 mañana / 3 tarde / 2 noche; S-D 2 / 2 / 2.
    Cada empleado tiene un turno preferido, y puede tener un bloque de vacaciones (hard)
    y algunos días de libranza solicitados (soft).
    """
    n_days = 7 * n_weeks
    employees = [_make_employee(i, n_weeks, seed) for i in range(n_employees)]

    demand: dict[tuple[int, str], int] = {}
    for d in range(n_days):
        row = weekend_demand if d % 7 in (5, 6) else weekday_demand
        for s, q in zip(SHIFTS, row):
            demand[(d, s)] = q

    unavailable: set[tuple[int, int]] = set()
    requests: set[tuple[int, int]] = set()
    for i in range(n_employees):
        vac = _vacation_block(i, n_days, seed)
        unavailable.update((i, d) for d in vac)
        rng = np.random.default_rng([seed, i, 2])
        for d in rng.choice(n_days, size=int(rng.integers(2, 4)), replace=False):
            if (i, int(d)) not in unavailable:
                requests.add((i, int(d)))

    return Instance(employees, n_days, demand, unavailable, requests)
