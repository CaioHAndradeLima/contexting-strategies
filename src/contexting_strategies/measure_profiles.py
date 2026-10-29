import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StrategyGate:
    min_evidence_coverage: float | None = None
    max_unauthorized_leak: float | None = None
    min_judge_groundedness: float | None = None
    min_judge_correctness: float | None = None


@dataclass(frozen=True)
class MeasureProfile:
    name: str
    description: str
    strategies: tuple[str, ...]
    candidate_k: int
    budget_tokens: int
    apply_heuristic_rerank: bool
    live_llm: bool
    max_llm_calls: int
    gates: dict[str, StrategyGate]


def load_measure_profiles(path: Path) -> dict[str, MeasureProfile]:
    payload = json.loads(path.read_text())
    profiles = {}
    for name, row in payload.get("profiles", {}).items():
        gates = {}
        for strategy, gate_row in row.get("gates", {}).items():
            gates[strategy] = StrategyGate(
                min_evidence_coverage=gate_row.get("min_evidence_coverage"),
                max_unauthorized_leak=gate_row.get("max_unauthorized_leak"),
                min_judge_groundedness=gate_row.get("min_judge_groundedness"),
                min_judge_correctness=gate_row.get("min_judge_correctness"),
            )
        profiles[name] = MeasureProfile(
            name=name,
            description=row.get("description", ""),
            strategies=tuple(row.get("strategies", [])),
            candidate_k=int(row.get("candidate_k", 10)),
            budget_tokens=int(row.get("budget_tokens", 200)),
            apply_heuristic_rerank=bool(row.get("apply_heuristic_rerank", True)),
            live_llm=bool(row.get("live_llm", False)),
            max_llm_calls=int(row.get("max_llm_calls", 100)),
            gates=gates,
        )
    return profiles


def evaluate_gates(report: dict, profile: MeasureProfile) -> dict:
    summary = report.get("summary", {})
    failures: list[str] = []
    checks: list[dict] = []
    for strategy, gate in profile.gates.items():
        values = summary.get(strategy)
        if values is None:
            failures.append(f"missing strategy summary '{strategy}'")
            continue
        if gate.min_evidence_coverage is not None:
            actual = values.get("evidence_coverage", 0)
            ok = actual >= gate.min_evidence_coverage
            checks.append(
                {
                    "strategy": strategy,
                    "metric": "evidence_coverage",
                    "expected_min": gate.min_evidence_coverage,
                    "actual": actual,
                    "passed": ok,
                }
            )
            if not ok:
                failures.append(
                    f"{strategy} evidence_coverage {actual:.3f} < {gate.min_evidence_coverage}"
                )
        if gate.max_unauthorized_leak is not None:
            actual = values.get("unauthorized_leak", 0)
            ok = actual <= gate.max_unauthorized_leak
            checks.append(
                {
                    "strategy": strategy,
                    "metric": "unauthorized_leak",
                    "expected_max": gate.max_unauthorized_leak,
                    "actual": actual,
                    "passed": ok,
                }
            )
            if not ok:
                failures.append(
                    f"{strategy} unauthorized_leak {actual:.3f} > {gate.max_unauthorized_leak}"
                )
        if gate.min_judge_groundedness is not None:
            actual = values.get("judge_groundedness")
            if actual is None:
                failures.append(f"{strategy} missing judge_groundedness (was live_llm enabled?)")
            else:
                ok = actual >= gate.min_judge_groundedness
                checks.append(
                    {
                        "strategy": strategy,
                        "metric": "judge_groundedness",
                        "expected_min": gate.min_judge_groundedness,
                        "actual": actual,
                        "passed": ok,
                    }
                )
                if not ok:
                    failures.append(
                        f"{strategy} judge_groundedness {actual:.3f} < {gate.min_judge_groundedness}"
                    )
        if gate.min_judge_correctness is not None:
            actual = values.get("judge_correctness")
            if actual is None:
                failures.append(f"{strategy} missing judge_correctness (was live_llm enabled?)")
            else:
                ok = actual >= gate.min_judge_correctness
                checks.append(
                    {
                        "strategy": strategy,
                        "metric": "judge_correctness",
                        "expected_min": gate.min_judge_correctness,
                        "actual": actual,
                        "passed": ok,
                    }
                )
                if not ok:
                    failures.append(
                        f"{strategy} judge_correctness {actual:.3f} < {gate.min_judge_correctness}"
                    )
    return {
        "profile": profile.name,
        "passed": not failures,
        "failures": failures,
        "checks": checks,
    }
