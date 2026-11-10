#!/usr/bin/env python3
"""Z3-based solver for the exam timetabling problem."""
import argparse
import math
import re
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations

import z3


class Instance:
    """One problem instance, as read from a .txt file."""

    def __init__(self):
        self.number_of_students = 0
        self.number_of_exams = 0
        self.number_of_slots = 0
        self.number_of_rooms = 0
        self.room_capacities = []
        self.exams_to_students = []  # (exam, student) pairs


def read_file(filename):
    """Read an instance file.

    Trailing spaces and blank lines are allowed; out-of-range exam/student
    numbers are rejected.
    """

    def read_attribute(name):
        line = f.readline()
        match = re.match(f"{name}:\\s*(\\d+)\\s*$", line)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not parse line {line!r}; expected the {name} attribute")

    instance = Instance()
    with open(filename) as f:
        instance.number_of_students = read_attribute("Number of students")
        instance.number_of_exams = read_attribute("Number of exams")
        instance.number_of_slots = read_attribute("Number of slots")
        instance.number_of_rooms = read_attribute("Number of rooms")
        for r in range(instance.number_of_rooms):
            instance.room_capacities.append(read_attribute(f"Room {r} capacity"))
        for line in f:
            if not line.strip():
                continue
            m = re.match("^\\s*(\\d+)\\s+(\\d+)\\s*$", line)
            if not m:
                raise ValueError(f"Failed to parse this line: {line!r}")
            exam, student = int(m.group(1)), int(m.group(2))
            if not (exam < instance.number_of_exams and student < instance.number_of_students):
                raise ValueError(f"exam {exam} / student {student} is out of range")
            instance.exams_to_students.append((exam, student))
    return instance


def get_student_data(instance):
    """exam_sizes[e]: how many students take exam e. student_exams[s]: the exams of
    student s. pairs: exam pairs (a, b), a < b, that share a student (for C4/C5)."""
    exam_students = [set() for _ in range(instance.number_of_exams)]
    student_exams = [set() for _ in range(instance.number_of_students)]
    for e, s in instance.exams_to_students:  # sets drop duplicate lines
        exam_students[e].add(s)
        student_exams[s].add(e)
    exam_sizes = [len(students) for students in exam_students]
    student_exams = [sorted(exams) for exams in student_exams]
    return exam_sizes, student_exams, student_exam_pairs(student_exams)


def student_exam_pairs(student_exams):
    """Unique exam pairs that share at least one student."""
    return sorted({pair for exams in student_exams for pair in combinations(exams, 2)})


# ============================================================
# Extra constraints X1-X5 (shared by all three models)
# ============================================================
@dataclass(frozen=True)
class Extras:
    """Which extras are switched on. None / empty means OFF (the default)."""

    max_per_day: tuple = None             # X1 (K, D)
    unavailable: frozenset = frozenset()  # X2 {(room, slot)}
    invigilators: tuple = None            # X3 (N, Q)
    min_gap: int = None                   # X4 G
    large_early: tuple = None             # X5 (N, K)


NO_EXTRAS = Extras()

# Constraint groups. Every model files each constraint under one of these names,
# so --explain can say which rules clash.
GROUPS = {
    "C1": "each exam in exactly one slot and one room",
    "C2": "at most one exam per room per slot",
    "C3": "room capacity",
    "C4": "no consecutive exams for a student",
    "C5": "no two exams in the same slot for a student",
    "X1": "max exams per student per day",
    "X2": "room unavailability",
    "X3": "invigilator limit per slot",
    "X4": "minimum gap between a student's exams",
    "X5": "large exams in early slots",
}


def get_extra_constraint_data(instance, exam_sizes, student_exams, extras):
    """Precompute the data each switched-on extra constraint needs."""
    T = instance.number_of_slots
    data = {}
    if extras.max_per_day:   # X1: K, and (exams of a student, slots of one day)
        k, d = extras.max_per_day
        data["X1"] = (k, [(exams, range(start, min(start + d, T)))
                          for exams in student_exams if len(exams) > k
                          for start in range(0, T, d)])
    if extras.unavailable:   # X2: closed (room, slot) cells that exist here
        data["X2"] = [(r, t) for r, t in sorted(extras.unavailable)
                      if r < instance.number_of_rooms and t < T]
    if extras.invigilators:  # X3: N, and need[e] = ceil(size / Q), at least 1
        n, q = extras.invigilators
        data["X3"] = (n, [max(1, math.ceil(size / q)) for size in exam_sizes])
    if extras.min_gap:       # X4: G, and slot pairs (t, u) closer than G
        g = extras.min_gap
        data["X4"] = (g, [(t, u) for t in range(T) for u in range(T) if 0 < abs(t - u) < g])
    if extras.large_early:   # X5: the large exams, and the early slots they may use
        n, k = extras.large_early
        data["X5"] = ([e for e, size in enumerate(exam_sizes) if size >= n], range(min(k, T)))
    return data


# ============================================================
# Model 1: int - one Int slot and one Int room per exam
# ============================================================
def build_int(instance, extras):
    """Returns (constraint groups, slot[e], room[e]) as Z3 integer terms."""
    E, T, R = instance.number_of_exams, instance.number_of_slots, instance.number_of_rooms
    exam_sizes, student_exams, pairs = get_student_data(instance)
    x = get_extra_constraint_data(instance, exam_sizes, student_exams, extras)
    c = {name: [] for name in GROUPS}

    slot = [z3.Int(f"slot_{e}") for e in range(E)]
    room = [z3.Int(f"room_{e}") for e in range(E)]

    # C1: one Int holds exactly one value, so we only keep it in range.
    for e in range(E):
        c["C1"].append(z3.And(slot[e] >= 0, slot[e] < T, room[e] >= 0, room[e] < R))
    # C2: two exams never have the same slot AND the same room.
    for a, b in combinations(range(E), 2):
        c["C2"].append(z3.Or(slot[a] != slot[b], room[a] != room[b]))
    # C3: an exam never uses a room that is too small.
    for e in range(E):
        for r in range(R):
            if exam_sizes[e] > instance.room_capacities[r]:
                c["C3"].append(room[e] != r)
    # C4: exams sharing a student are not one slot apart.
    for a, b in pairs:
        c["C4"].append(z3.And(slot[a] - slot[b] != 1, slot[b] - slot[a] != 1))
    # C5: exams sharing a student are in different slots.
    for a, b in pairs:
        c["C5"].append(slot[a] != slot[b])

    # X1: count a student's exams that fall on the day; at most K.
    if "X1" in x:
        k, days = x["X1"]
        for exams, day in days:
            on_day = [z3.If(z3.And(slot[e] >= day.start, slot[e] < day.stop), 1, 0) for e in exams]
            c["X1"].append(z3.Sum(on_day) <= k)
    # X2: no exam in room r during slot t.
    for r, t in x.get("X2", []):
        for e in range(E):
            c["X2"].append(z3.Or(slot[e] != t, room[e] != r))
    # X3: invigilators needed by the exams in slot t are at most N.
    if "X3" in x:
        n, need = x["X3"]
        for t in range(T):
            c["X3"].append(z3.Sum([z3.If(slot[e] == t, need[e], 0) for e in range(E)]) <= n)
    # X4: exams sharing a student are at least G slots apart.
    if "X4" in x:
        g, _ = x["X4"]
        for a, b in pairs:
            c["X4"].append(z3.Or(slot[a] - slot[b] >= g, slot[b] - slot[a] >= g))
    # X5: large exams go in the first K slots.
    if "X5" in x:
        large, early = x["X5"]
        for e in large:
            c["X5"].append(slot[e] < early.stop)
    return c, slot, room


# ============================================================
# Model 2: bool - True/False tables
# ============================================================
def exactly_one(row):
    """Exactly one Bool in row is True: the pseudo-Boolean sum of the row == 1."""
    return z3.PbEq([(v, 1) for v in row], 1) if row else z3.BoolVal(False)


def position(row):
    """Index of the True entry in a one-hot row, as a Z3 integer (row [F,T,F] -> 1).
    Lets --count, --optimize and the decoder treat every model the same way."""
    return z3.Sum([z3.If(v, i, 0) for i, v in enumerate(row)])


def build_bool(instance, extras):
    """Returns (constraint groups, slot[e], room[e]) as Z3 integer terms."""
    E, T, R = instance.number_of_exams, instance.number_of_slots, instance.number_of_rooms
    exam_sizes, student_exams, pairs = get_student_data(instance)
    x = get_extra_constraint_data(instance, exam_sizes, student_exams, extras)
    c = {name: [] for name in GROUPS}

    # slot_assign[e][t] is True when exam e is in slot t; room_assign[e][r] likewise.
    slot_assign = [[z3.Bool(f"slot_{e}_{t}") for t in range(T)] for e in range(E)]
    room_assign = [[z3.Bool(f"room_{e}_{r}") for r in range(R)] for e in range(E)]

    # C1: exactly one slot and exactly one room per exam.
    for e in range(E):
        c["C1"] += [exactly_one(slot_assign[e]), exactly_one(room_assign[e])]
    # C2: if two exams share a slot, they must not share a room.
    for a, b in combinations(range(E), 2):
        same_slot = z3.Or([z3.And(slot_assign[a][t], slot_assign[b][t]) for t in range(T)])
        same_room = z3.Or([z3.And(room_assign[a][r], room_assign[b][r]) for r in range(R)])
        c["C2"].append(z3.Implies(same_slot, z3.Not(same_room)))
    # C3: an exam never uses a room that is too small.
    for e in range(E):
        for r in range(R):
            if exam_sizes[e] > instance.room_capacities[r]:
                c["C3"].append(z3.Not(room_assign[e][r]))
    # C4: if a is in slot t, b is not in slot t+1 (and the other way round).
    for a, b in pairs:
        for t in range(T - 1):
            c["C4"].append(z3.Implies(slot_assign[a][t], z3.Not(slot_assign[b][t + 1])))
            c["C4"].append(z3.Implies(slot_assign[b][t], z3.Not(slot_assign[a][t + 1])))
    # C5: if a is in slot t, b is not in slot t.
    for a, b in pairs:
        for t in range(T):
            c["C5"].append(z3.Implies(slot_assign[a][t], z3.Not(slot_assign[b][t])))

    # X1: at most K of a student's exams are in the day's slots.
    if "X1" in x:
        k, days = x["X1"]
        for exams, day in days:
            c["X1"].append(z3.AtMost(*[slot_assign[e][t] for e in exams for t in day], k))
    # X2: no exam in room r during slot t.
    for r, t in x.get("X2", []):
        for e in range(E):
            c["X2"].append(z3.Not(z3.And(slot_assign[e][t], room_assign[e][r])))
    # X3: invigilators needed in slot t (weighted sum) are at most N.
    if "X3" in x and E:
        n, need = x["X3"]
        for t in range(T):
            c["X3"].append(z3.PbLe([(slot_assign[e][t], need[e]) for e in range(E)], n))
    # X4: if a is in slot t, b is not in any slot u that is too close.
    if "X4" in x:
        _, close = x["X4"]
        for a, b in pairs:
            for t, u in close:
                c["X4"].append(z3.Implies(slot_assign[a][t], z3.Not(slot_assign[b][u])))
    # X5: a large exam is in one of the early slots.
    if "X5" in x:
        large, early = x["X5"]
        for e in large:
            c["X5"].append(z3.Or([slot_assign[e][t] for t in early]))
    return c, [position(row) for row in slot_assign], [position(row) for row in room_assign]


# ============================================================
# Model 3: cnf - the bool tables, clauses only
# ============================================================
def at_most(lits, k, weights=None):
    """Clauses for "the (weighted) number of True lits is at most k": for every
    smallest group of lits that would go over k, add "not all of them".
    With k = 1 this is the usual pairwise rule Or(Not(a), Not(b))."""
    weights = weights or [1] * len(lits)
    clauses = []

    def grow(group, total, start):
        for i in range(start, len(lits)):
            if total + weights[i] > k:
                clauses.append(z3.Or([z3.Not(lits[j]) for j in group + [i]]))
            else:
                grow(group + [i], total + weights[i], i + 1)

    grow([], 0, 0)
    return clauses


def build_cnf(instance, extras):
    """Returns (constraint groups, slot[e], room[e]) as Z3 integer terms.
    Every constraint below is an Or of variables or Not(variable)."""
    E, T, R = instance.number_of_exams, instance.number_of_slots, instance.number_of_rooms
    exam_sizes, student_exams, pairs = get_student_data(instance)
    x = get_extra_constraint_data(instance, exam_sizes, student_exams, extras)
    c = {name: [] for name in GROUPS}
    Not, Or = z3.Not, z3.Or

    slot_assign = [[z3.Bool(f"slot_{e}_{t}") for t in range(T)] for e in range(E)]
    room_assign = [[z3.Bool(f"room_{e}_{r}") for r in range(R)] for e in range(E)]

    # C1: "at least one" is one clause; "at most one" is "not both" for every pair.
    for e in range(E):
        for row in (slot_assign[e], room_assign[e]):
            c["C1"].append(Or(row))
            c["C1"] += at_most(row, 1)
    # C2: helper same_slot is forced True when a and b share a slot; then they may
    #     not share a room. This is the bool model's C2 converted to clauses.
    for a, b in combinations(range(E), 2):
        same_slot = z3.Bool(f"same_slot_{a}_{b}")
        for t in range(T):
            c["C2"].append(Or(Not(slot_assign[a][t]), Not(slot_assign[b][t]), same_slot))
        for r in range(R):
            c["C2"].append(Or(Not(same_slot), Not(room_assign[a][r]), Not(room_assign[b][r])))
    # C3: a unit clause bans each room that is too small.
    for e in range(E):
        for r in range(R):
            if exam_sizes[e] > instance.room_capacities[r]:
                c["C3"].append(Not(room_assign[e][r]))
    # C4: not (a in t and b in t+1), and the other way round.
    for a, b in pairs:
        for t in range(T - 1):
            c["C4"].append(Or(Not(slot_assign[a][t]), Not(slot_assign[b][t + 1])))
            c["C4"].append(Or(Not(slot_assign[b][t]), Not(slot_assign[a][t + 1])))
    # C5: not (a in t and b in t).
    for a, b in pairs:
        for t in range(T):
            c["C5"].append(Or(Not(slot_assign[a][t]), Not(slot_assign[b][t])))

    # X1: no K+1 of a student's exams can all be on the same day.
    if "X1" in x:
        k, days = x["X1"]
        for exams, day in days:
            c["X1"] += at_most([slot_assign[e][t] for e in exams for t in day], k)
    # X2: not (e in slot t and e in room r).
    for r, t in x.get("X2", []):
        for e in range(E):
            c["X2"].append(Or(Not(slot_assign[e][t]), Not(room_assign[e][r])))
    # X3: no group of exams needing more than N invigilators shares slot t.
    if "X3" in x:
        n, need = x["X3"]
        for t in range(T):
            c["X3"] += at_most([slot_assign[e][t] for e in range(E)], n, need)
    # X4: not (a in t and b in u) when t and u are too close.
    if "X4" in x:
        _, close = x["X4"]
        for a, b in pairs:
            for t, u in close:
                c["X4"].append(Or(Not(slot_assign[a][t]), Not(slot_assign[b][u])))
    # X5: a large exam is in one of the early slots (one clause).
    if "X5" in x:
        large, early = x["X5"]
        for e in large:
            c["X5"].append(Or([slot_assign[e][t] for t in early]))
    return c, [position(row) for row in slot_assign], [position(row) for row in room_assign]


MODELS = {"int": build_int, "bool": build_bool, "cnf": build_cnf}
DEFAULT_MODEL = "int"


# ============================================================
# Solving and analysis
# ============================================================
@dataclass
class Result:
    status: str                # "sat", "unsat" or "unknown"
    timetable: dict            # exam -> (slot, room), or None
    build_ms: float
    check_ms: float
    total_ms: float = field(init=False)

    def __post_init__(self):
        self.total_ms = self.build_ms + self.check_ms


def add_all(solver, groups):
    """Add every constraint of every group to a Z3 Solver or Optimize."""
    for constraints in groups.values():
        solver.add(constraints)
    return solver


def read_timetable(model, slot, room):
    """exam -> (slot, room) from a Z3 model."""
    return {e: (model.eval(slot[e], model_completion=True).as_long(),
                model.eval(room[e], model_completion=True).as_long()) for e in range(len(slot))}


def solve(instance, model=DEFAULT_MODEL, extras=NO_EXTRAS, timeout_ms=None):
    """Build the chosen model, ask Z3 whether it is sat, and read the timetable."""
    start = time.perf_counter()
    groups, slot, room = MODELS[model](instance, extras)
    # bool/cnf only use True/False variables, so Z3's finite-domain (SAT) engine suits them best.
    solver = add_all(z3.Solver() if model == "int" else z3.SolverFor("QF_FD"), groups)
    if timeout_ms:
        solver.set("timeout", timeout_ms)
    built = time.perf_counter()
    result = solver.check()
    done = time.perf_counter()
    timetable = read_timetable(solver.model(), slot, room) if result == z3.sat else None
    return Result(str(result), timetable, (built - start) * 1000, (done - built) * 1000)


def count_timetables(instance, model=DEFAULT_MODEL, extras=NO_EXTRAS, cap=1000):
    """Find up to cap timetables. After each one, add a clause that blocks it (at
    least one exam must change slot or room) and solve again."""
    groups, slot, room = MODELS[model](instance, extras)
    solver = add_all(z3.Solver(), groups)
    found = []
    while len(found) < cap and solver.check() == z3.sat:
        timetable = read_timetable(solver.model(), slot, room)
        found.append(timetable)
        solver.add(z3.Or([z3.Or(slot[e] != t, room[e] != r) for e, (t, r) in timetable.items()]))
    return found, len(found) < cap or solver.check() == z3.unsat


def explain_unsat(instance, model=DEFAULT_MODEL, extras=NO_EXTRAS):
    """Smallest set of constraint groups that cannot all hold (None if sat).
    1. Each group gets an on/off switch; Z3's unsat core lists the switches it used.
    2. Drop each group in the core in turn; if the rest is still unsat, it was not
       needed. A fresh QF_FD (finite-domain, SAT-based) solver proves these checks
       fastest; if it gives up ("unknown") we keep the group, so the answer is
       always a real clash."""
    groups, _, _ = MODELS[model](instance, extras)
    solver = z3.Solver()
    switch = {name: z3.Bool(f"use_{name}") for name in groups if groups[name]}
    for name, sw in switch.items():
        solver.add(z3.Implies(sw, z3.And(groups[name])))
    if solver.check(*switch.values()) != z3.unsat:
        return None
    core = [name for name in switch if any(z3.eq(switch[name], s) for s in solver.unsat_core())]
    for name in list(core):
        rest = z3.SolverFor("QF_FD")
        for other in core:
            if other != name:
                rest.add(groups[other])
        if rest.check() == z3.unsat:
            core.remove(name)
    return core


def optimise(instance, objective, model=DEFAULT_MODEL, extras=NO_EXTRAS):
    """Best timetable with z3.Optimize. Returns (status, timetable, best value).
    gap: largest g with |slot[a] - slot[b]| >= g for every pair sharing a student.
    makespan: smallest m with slot[e] <= m for every exam."""
    groups, slot, room = MODELS[model](instance, extras)
    opt = add_all(z3.Optimize(), groups)
    _, _, pairs = get_student_data(instance)
    goal = None
    if objective == "gap" and pairs:
        goal = z3.Int("gap")
        opt.add(goal <= instance.number_of_slots)
        for a, b in pairs:
            opt.add(z3.Or(slot[a] - slot[b] >= goal, slot[b] - slot[a] >= goal))
        opt.maximize(goal)
    elif objective == "makespan":
        goal = z3.Int("makespan")
        opt.add(goal >= 0)
        opt.add([s <= goal for s in slot])
        opt.minimize(goal)
    status = opt.check()
    if status != z3.sat:
        return str(status), None, None
    m = opt.model()
    value = m.eval(goal).as_long() if goal is not None and slot else None
    return "sat", read_timetable(m, slot, room), value


def check_timetable(instance, timetable, extras=NO_EXTRAS):
    """Re-check a timetable in plain Python (no Z3). Returns the broken rules."""
    exam_sizes, student_exams, pairs = get_student_data(instance)
    T, R, cap = instance.number_of_slots, instance.number_of_rooms, instance.room_capacities
    bad = []
    if set(timetable) != set(range(instance.number_of_exams)):
        bad.append("C1: not every exam has exactly one place")
    used = {}
    for e, (t, r) in timetable.items():
        if not (0 <= t < T and 0 <= r < R):
            bad.append(f"C1: exam {e} has slot {t} / room {r} out of range")
            continue
        if (t, r) in used:
            bad.append(f"C2: exams {used[t, r]} and {e} share slot {t}, room {r}")
        used[t, r] = e
        if exam_sizes[e] > cap[r]:
            bad.append(f"C3: exam {e} has {exam_sizes[e]} students > capacity {cap[r]} of room {r}")
        if (r, t) in extras.unavailable:
            bad.append(f"X2: exam {e} uses closed room {r} in slot {t}")
    for a, b in pairs:
        if a in timetable and b in timetable:
            gap = abs(timetable[a][0] - timetable[b][0])
            if gap == 0:
                bad.append(f"C5: exams {a} and {b} share a student and slot {timetable[a][0]}")
            if gap == 1:
                bad.append(f"C4: exams {a} and {b} share a student in consecutive slots")
            if extras.min_gap and gap < extras.min_gap:
                bad.append(f"X4: exams {a} and {b} are {gap} < {extras.min_gap} slots apart")
    if extras.max_per_day:
        k, d = extras.max_per_day
        for s, exams in enumerate(student_exams):
            days = [timetable[e][0] // d for e in exams if e in timetable]
            bad += [f"X1: student {s} has {days.count(day)} exams on day {day}"
                    for day in set(days) if days.count(day) > k]
    if extras.invigilators:
        n, q = extras.invigilators
        load = {}
        for e, (t, _) in timetable.items():
            load[t] = load.get(t, 0) + max(1, math.ceil(exam_sizes[e] / q))
        bad += [f"X3: slot {t} needs {need} > {n} invigilators" for t, need in load.items() if need > n]
    if extras.large_early:
        n, k = extras.large_early
        bad += [f"X5: exam {e} ({exam_sizes[e]} students) is in slot {t} >= {k}"
                for e, (t, _) in timetable.items() if exam_sizes[e] >= n and t >= k]
    return bad


# ============================================================
# Command line
# ============================================================
def print_timetable(instance, timetable):
    exam_sizes, _, _ = get_student_data(instance)
    print(" Exam | Slot | Room | Students | Capacity")
    print("------+------+------+----------+---------")
    for e, (t, r) in sorted(timetable.items(), key=lambda item: (item[1], item[0])):
        print(f" {e:>4} | {t:>4} | {r:>4} | {exam_sizes[e]:>8} | {instance.room_capacities[r]:>8}")


def room_slot(text):
    """'R:T' -> (R, T) for --unavailable."""
    m = re.fullmatch(r"(\d+):(\d+)", text)
    if not m:
        raise argparse.ArgumentTypeError(f"expected ROOM:SLOT, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Z3 exam timetabling solver")
    ap.add_argument("files", nargs="+", help="instance file(s)")
    ap.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL, help="formulation (default int)")
    ap.add_argument("--check", action="store_true", help="re-check the timetable without Z3")
    ap.add_argument("--count", type=int, metavar="CAP", help="count up to CAP timetables")
    ap.add_argument("--explain", action="store_true", help="for unsat: which constraint groups clash")
    ap.add_argument("--optimize", choices=["gap", "makespan"], help="find a best timetable")
    ap.add_argument("--timeout", type=int, metavar="MS", help="Z3 timeout in ms")
    x = ap.add_argument_group("extra constraints (all OFF by default)")
    x.add_argument("--max-per-day", type=int, nargs=2, metavar=("K", "D"),
                   help="X1: at most K exams per student per day of D slots")
    x.add_argument("--unavailable", type=room_slot, action="append", default=[], metavar="R:T",
                   help="X2: room R closed in slot T (repeatable)")
    x.add_argument("--invigilators", type=int, nargs=2, metavar=("N", "Q"),
                   help="X3: N invigilators per slot, one per Q students")
    x.add_argument("--min-gap", type=int, metavar="G", help="X4: a student's exams at least G slots apart")
    x.add_argument("--large-early", type=int, nargs=2, metavar=("N", "K"),
                   help="X5: exams with >= N students in the first K slots")
    a = ap.parse_args(argv)
    if a.max_per_day and min(a.max_per_day) < 1:
        ap.error("--max-per-day needs K >= 1 and D >= 1")
    if a.invigilators and (a.invigilators[0] < 0 or a.invigilators[1] < 1):
        ap.error("--invigilators needs N >= 0 and Q >= 1")
    if a.min_gap is not None and a.min_gap < 1:
        ap.error("--min-gap needs G >= 1")
    if a.large_early and a.large_early[1] < 0:
        ap.error("--large-early needs K >= 0")
    extras = Extras(tuple(a.max_per_day) if a.max_per_day else None, frozenset(a.unavailable),
                    tuple(a.invigilators) if a.invigilators else None, a.min_gap,
                    tuple(a.large_early) if a.large_early else None)
    return a, extras


def run_file(path, a, extras):
    """Solve one file and print the results. Returns False if --check finds a problem."""
    instance = read_file(path)
    res = solve(instance, a.model, extras, a.timeout)
    print(res.status)
    if res.timetable is not None:
        print_timetable(instance, res.timetable)
    print(f"Time: {res.total_ms:.2f} ms (build {res.build_ms:.2f} ms + solve "
          f"{res.check_ms:.2f} ms, model {a.model})")
    ok = True
    if a.check and res.timetable is not None:
        problems = check_timetable(instance, res.timetable, extras)
        ok = not problems
        print("Checker: OK, all active constraints hold" if ok
              else "Checker: FAILED\n  " + "\n  ".join(problems))
    if a.count:
        start = time.perf_counter()
        found, exact = count_timetables(instance, a.model, extras, a.count)
        ms = (time.perf_counter() - start) * 1000
        print(f"Solutions: {'exactly' if exact else 'at least'} {len(found)} timetable(s) ({ms:.1f} ms)"
              + ("; the timetable is unique" if exact and len(found) == 1 else ""))
    if a.explain and res.status == "unsat":
        print("Explanation: smallest set of clashing constraints:")
        for name in explain_unsat(instance, a.model, extras) or []:
            print(f"  {name}: {GROUPS[name]}")
    if a.optimize:
        status, timetable, value = optimise(instance, a.optimize, a.model, extras)
        if timetable is None:
            print(f"Optimum: {status}")
        else:
            label = {"gap": "largest smallest gap between a student's exams",
                     "makespan": "smallest last slot used"}[a.optimize]
            print(f"Optimum ({label}): {value if value is not None else 'n/a (no shared students)'}")
            print_timetable(instance, timetable)
    return ok


def warm_up(model):
    """Solve a one-exam instance once, untimed, so reported times leave out
    Z3's one-off start-up cost."""
    tiny = Instance()
    tiny.number_of_students = tiny.number_of_exams = tiny.number_of_slots = tiny.number_of_rooms = 1
    tiny.room_capacities, tiny.exams_to_students = [1], [(0, 0)]
    solve(tiny, model)


def main(argv=None):
    a, extras = parse_args(argv)
    warm_up(a.model)
    ok = True
    for i, path in enumerate(a.files):
        if len(a.files) > 1:
            print(("\n" if i else "") + f"== {path}")
        ok &= run_file(path, a, extras)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
