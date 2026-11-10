"""
Benchmark and comparison experiments for the exam timetabling solver.

    python benchmark.py            # everything (about 10-20 minutes)
    python benchmark.py --quick    # small version for a smoke test

Writes CSVs and PNG charts to results/. Each time is the median of --repeats
runs (build + solve), measured with perf_counter.

Optional comparison: if OR-Tools is installed, the same C1-C5 problem is also
solved with Google's CP-SAT solver using integer and Boolean formulations.
These are reference models only (skipped if ortools is not installed); the
main solver is Z3-based.
"""
import argparse
import csv
import glob
import os
import statistics
import time
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cw1_solver import (
    MODELS,
    NO_EXTRAS,
    Extras,
    check_timetable,
    count_timetables,
    explain_unsat,
    get_student_data,
    optimise,
    read_file,
    solve,
)
from gen_instances import random_instance

try:
    from ortools.sat.python import cp_model
except ImportError:  # the comparison is skipped without OR-Tools
    cp_model = None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
COLOURS = {"int": "#2ca02c", "bool": "#1f77b4", "cnf": "#ff7f0e",
           "cpsat_int": "#9467bd", "cpsat_bool": "#8c564b"}


# ---- OR-Tools CP-SAT reference models (C1-C5 only; comparison only) ----

def cpsat_int(instance):
    """CP-SAT integer model: one Int slot and room per exam."""
    model = cp_model.CpModel()
    E, T, R = instance.number_of_exams, instance.number_of_slots, instance.number_of_rooms
    sizes, _, pairs = get_student_data(instance)
    slot = [model.NewIntVar(0, T - 1, f"slot_{e}") for e in range(E)]
    room = [model.NewIntVar(0, R - 1, f"room_{e}") for e in range(E)]
    for a, b in combinations(range(E), 2):  # C2 via "same slot" / "same room" helpers
        same_slot, same_room = model.NewBoolVar(""), model.NewBoolVar("")
        model.Add(slot[a] == slot[b]).OnlyEnforceIf(same_slot)
        model.Add(slot[a] != slot[b]).OnlyEnforceIf(same_slot.Not())
        model.Add(room[a] == room[b]).OnlyEnforceIf(same_room)
        model.Add(room[a] != room[b]).OnlyEnforceIf(same_room.Not())
        model.AddBoolOr([same_slot.Not(), same_room.Not()])
    for e in range(E):  # C3
        for r in range(R):
            if sizes[e] > instance.room_capacities[r]:
                model.Add(room[e] != r)
    for a, b in pairs:  # C4 and C5: |slot[a] - slot[b]| >= 2
        gap = model.NewIntVar(0, max(T - 1, 0), "")
        model.AddAbsEquality(gap, slot[a] - slot[b])
        model.Add(gap >= 2)
    return model


def cpsat_bool(instance):
    """CP-SAT Boolean model: slot_assign / room_assign tables."""
    model = cp_model.CpModel()
    E, T, R = instance.number_of_exams, instance.number_of_slots, instance.number_of_rooms
    sizes, _, pairs = get_student_data(instance)
    s = [[model.NewBoolVar("") for _ in range(T)] for _ in range(E)]
    r = [[model.NewBoolVar("") for _ in range(R)] for _ in range(E)]
    for e in range(E):  # C1
        model.AddExactlyOne(s[e])
        model.AddExactlyOne(r[e])
    for a, b in combinations(range(E), 2):  # C2
        for t in range(T):
            for k in range(R):
                model.AddBoolOr([s[a][t].Not(), r[a][k].Not(), s[b][t].Not(), r[b][k].Not()])
    for e in range(E):  # C3
        for k in range(R):
            if sizes[e] > instance.room_capacities[k]:
                model.Add(r[e][k] == 0)
    for a, b in pairs:  # C4 and C5
        for t in range(T):
            model.AddBoolOr([s[a][t].Not(), s[b][t].Not()])
            if t + 1 < T:
                model.AddBoolOr([s[a][t].Not(), s[b][t + 1].Not()])
                model.AddBoolOr([s[b][t].Not(), s[a][t + 1].Not()])
    return model


def cpsat_solve(instance, build, timeout_ms):
    start = time.perf_counter()
    if instance.number_of_exams and not (instance.number_of_slots and instance.number_of_rooms):
        return "unsat", (time.perf_counter() - start) * 1000  # CP-SAT rejects empty domains
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1  # single thread, fixed seed
    solver.parameters.random_seed = 0
    if timeout_ms:
        solver.parameters.max_time_in_seconds = timeout_ms / 1000
    status = solver.Solve(build(instance))
    name = {cp_model.OPTIMAL: "sat", cp_model.FEASIBLE: "sat", cp_model.INFEASIBLE: "unsat"}
    return name.get(status, "unknown"), (time.perf_counter() - start) * 1000


SOLVERS = list(MODELS) + (["cpsat_int", "cpsat_bool"] if cp_model else [])


def timed(instance, model, repeats, timeout_ms=None, extras=NO_EXTRAS):
    """Median build/check/total ms over repeats runs, plus status and validity."""
    if model.startswith("cpsat"):
        build = cpsat_int if model == "cpsat_int" else cpsat_bool
        runs = [cpsat_solve(instance, build, timeout_ms) for _ in range(repeats)]
        return {"status": runs[0][0], "valid": True, "build_ms": "", "check_ms": "",
                "total_ms": statistics.median(ms for _, ms in runs)}
    runs = [solve(instance, model, extras, timeout_ms) for _ in range(repeats)]
    first = runs[0]
    valid = first.timetable is None or not check_timetable(instance, first.timetable, extras)
    return {"status": first.status, "valid": valid,
            "build_ms": statistics.median(r.build_ms for r in runs),
            "check_ms": statistics.median(r.check_ms for r in runs),
            "total_ms": statistics.median(r.total_ms for r in runs)}


def instance_paths():
    def key(path):
        name = os.path.basename(path)
        return name.startswith("unsat"), int("".join(filter(str.isdigit, name)))
    return sorted(glob.glob(os.path.join(HERE, "inst", "*.txt")), key=key)


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote results/{name} ({len(rows)} rows)", flush=True)


# ---- 1. supplied instances --------------------------------------------------------

def supplied(repeats):
    rows = []
    for path in instance_paths():
        name = os.path.basename(path)[:-4]
        inst = read_file(path)
        for model in SOLVERS:
            t = timed(inst, model, repeats)
            exp = "unsat" if name.startswith("unsat") else "sat"
            n = sum(map(len, MODELS[model](inst, NO_EXTRAS)[0].values())) if model in MODELS else ""
            rows.append({"instance": name, "expected": exp, "model": model,
                         "exams": inst.number_of_exams, "slots": inst.number_of_slots,
                         "rooms": inst.number_of_rooms, "constraints": n, **t})
            assert t["status"] == exp and t["valid"], rows[-1]
    write_csv("supplied_timings.csv", rows)
    names = list(dict.fromkeys(r["instance"] for r in rows))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    width = 0.8 / len(SOLVERS)
    for i, model in enumerate(SOLVERS):
        vals = [next(r["total_ms"] for r in rows if r["instance"] == n and r["model"] == model)
                for n in names]
        ax.bar([j + i * width for j in range(len(names))], vals, width, label=model, color=COLOURS[model])
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(names))], names, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel(f"median time, ms (log, n={repeats})")
    ax.set_title("Build + solve time on the 20 supplied instances")
    ax.legend(ncol=len(SOLVERS))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "supplied_times.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    build = [sum(r["build_ms"] for r in rows if r["model"] == m) for m in MODELS]
    check = [sum(r["check_ms"] for r in rows if r["model"] == m) for m in MODELS]
    ax.bar(list(MODELS), build, label="build (Python API)", color="#bbbbbb")
    ax.bar(list(MODELS), check, bottom=build, label="check (Z3)", color="#444444")
    ax.set_ylabel("total over 20 instances, ms")
    ax.set_title("Where the time goes: building vs solving")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "build_vs_check.png"), dpi=160)
    plt.close(fig)


# ---- 2. scaling study --------------------------------------------------------------

def scaling(sizes, seeds, timeout_ms, budget_ms):
    """Random instances with |S| = 3|E|, |T| = |E|, |R| = max(2, |E|/5).
    A solver is dropped for larger sizes once its median passes budget_ms."""
    rows, active = [], set(SOLVERS)
    for n in sizes:
        for seed in range(seeds):
            inst = random_instance(3 * n, n, n, max(2, n // 5), (1, 3), (5, 25), seed=1000 * n + seed)
            for model in [m for m in SOLVERS if m in active]:
                t = timed(inst, model, 1, timeout_ms)
                rows.append({"exams": n, "seed": seed, "model": model, **t})
                print(f"  scaling E={n} seed={seed} {model}: {t['status']} {t['total_ms']:.0f} ms",
                      flush=True)
        for model in list(active):
            times = [r["total_ms"] for r in rows if r["model"] == model and r["exams"] == n]
            if statistics.median(times) > budget_ms:
                active.discard(model)
    write_csv("scaling.csv", rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in SOLVERS:
        xs = sorted({r["exams"] for r in rows if r["model"] == model})
        ax.plot(xs, [statistics.median(r["total_ms"] for r in rows if r["model"] == model and r["exams"] == x)
                     for x in xs], "o-", label=model, color=COLOURS[model])
    ax.set_yscale("log")
    ax.set_xlabel("|E| (with |S| = 3|E|, |T| = |E|, |R| = |E|/5)")
    ax.set_ylabel(f"median ms over {seeds} seeds (log)")
    ax.axhline(timeout_ms, ls=":", color="grey")
    ax.set_title("Scaling: build + solve time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scaling.png"), dpi=160)
    plt.close(fig)


# ---- 3. phase transition -------------------------------------------------------------

def phase_transition(n_exams, slot_values, seeds, timeout_ms):
    """Fix |E|, vary |T|: the share of sat instances goes 0 -> 1 and hardness peaks in between."""
    rows = []
    for n_slots in slot_values:
        for seed in range(seeds):
            inst = random_instance(3 * n_exams, n_exams, n_slots, 4, (1, 3), (5, 25), seed=seed)
            for model in SOLVERS:
                rows.append({"slots": n_slots, "seed": seed, "model": model,
                             **timed(inst, model, 1, timeout_ms)})
        print(f"  phase T={n_slots} done", flush=True)
    write_csv("phase_transition.csv", rows)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()
    ref = [r for r in rows if r["model"] == "int"]
    ax2.plot(slot_values, [sum(r["status"] == "sat" for r in ref if r["slots"] == t) / seeds
                           for t in slot_values], "k--", label="P(sat)")
    for model in SOLVERS:
        ax1.plot(slot_values, [statistics.median(r["total_ms"] for r in rows
                                                 if r["model"] == model and r["slots"] == t)
                               for t in slot_values], "o-", label=model, color=COLOURS[model])
    ax1.set_yscale("log")
    ax1.set_xlabel(f"|T| (|E| = {n_exams}, |S| = {3 * n_exams}, |R| = 4)")
    ax1.set_ylabel("median build + solve time, ms (log)")
    ax2.set_ylabel("fraction satisfiable")
    ax2.set_ylim(-0.05, 1.05)
    ax1.set_title("Easy-hard-easy: solver effort against number of slots")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "phase_transition.png"), dpi=160)
    plt.close(fig)


# ---- 4. analysis features and extras on the supplied instances ---------------------------

def analyses(model, cap):
    counts, cores, optima = [], [], []
    for path in instance_paths():
        name = os.path.basename(path)[:-4]
        inst = read_file(path)
        if name.startswith("unsat"):
            cores.append({"instance": name, **{f"core_{m}": " ".join(explain_unsat(inst, m) or [])
                                               for m in MODELS}})
            continue
        start = time.perf_counter()
        found, exact = count_timetables(inst, model, cap=cap)
        counts.append({"instance": name, "solutions": len(found), "exact": exact,
                       "ms": (time.perf_counter() - start) * 1000})
        for objective in ("gap", "makespan"):
            start = time.perf_counter()
            _, tt, value = optimise(inst, objective, model)
            optima.append({"instance": name, "objective": objective, "optimum": value,
                           "valid": not check_timetable(inst, tt),
                           "ms": (time.perf_counter() - start) * 1000})
    write_csv("solution_counts.csv", counts)
    write_csv("unsat_cores.csv", cores)
    write_csv("optimisation.csv", optima)


EXTRA_SETTINGS = {
    "X1 max 1 exam per 3-slot day": Extras(max_per_day=(1, 3)),
    "X2 room 0 closed in slots 0-1": Extras(unavailable=frozenset({(0, 0), (0, 1)})),
    "X3 3 invigilators, 1 per 10": Extras(invigilators=(3, 10)),
    "X4 min gap 3": Extras(min_gap=3),
    "X5 >=10 students in first 3 slots": Extras(large_early=(10, 3)),
}


def extras(model):
    """Effect of each extra constraint on the supplied sat instances."""
    rows = []
    for path in instance_paths():
        name = os.path.basename(path)[:-4]
        if name.startswith("unsat"):
            continue
        inst = read_file(path)
        row = {"instance": name}
        for label, x in EXTRA_SETTINGS.items():
            t = timed(inst, model, 3, extras=x)
            assert t["valid"]
            row[label] = f"{t['status']} ({t['total_ms']:.1f} ms)"
        rows.append(row)
    write_csv("extra_constraints.csv", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--model", default="int", help="model for the analysis tables")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for model in MODELS:  # untimed warm-up, as in the solver
        solve(read_file(instance_paths()[0]), model)
    supplied(3 if a.quick else a.repeats)
    analyses(a.model, 200 if a.quick else 2_000)
    extras(a.model)
    if a.quick:
        scaling([5, 10, 20], 2, 5_000, 5_000)
        phase_transition(10, [4, 8, 12], 3, 5_000)
    else:
        scaling([5, 10, 20, 30, 40, 60, 80, 100, 120], 3, 30_000, 10_000)
        phase_transition(20, list(range(6, 31, 2)), 10, 10_000)


if __name__ == "__main__":
    main()
