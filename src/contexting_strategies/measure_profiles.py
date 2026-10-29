import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StrategyGate:
    min_evidence_coverage: float | None = None
    max_unauthorized_leak: float | None = None
    max_redundancy_rate: float | None = None
    min_citation_traceability: float | None = None
    max_assembly_ms_p95: float | None = None
    min_abstain_correct: float | None = None
    min_judge_groundedness: float | None = None
    min_judge_correctness: float | None = None
    min_judge_evidence_sufficiency: float | None = None


@dataclass(frozen=True)
class ScenarioGates:
    min_abstain_correct_rate: float | None = None
    min_multihop_hop_coverage: float | None = None
    min_multihop_lift_over_baseline: float | None = None


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
    scenario_gates: ScenarioGates | None = None


def load_measure_profiles(path: Path) -> dict[str, MeasureProfile]:
    payload = json.loads(path.read_text())
    profiles = {}
    for name, row in payload.get("profiles", {}).items():
        gates = {}
        for strategy, gate_row in row.get("gates", {}).items():
            gates[strategy] = StrategyGate(
                min_evidence_coverage=gate_row.get("min_evidence_coverage"),
                max_unauthorized_leak=gate_row.get("max_unauthorized_leak"),
                max_redundancy_rate=gate_row.get("max_redundancy_rate"),
                min_citation_traceability=gate_row.get("min_citation_traceability"),
                max_assembly_ms_p95=gate_row.get("max_assembly_ms_p95"),
                min_abstain_correct=gate_row.get("min_abstain_correct"),
                min_judge_groundedness=gate_row.get("min_judge_groundedness"),
                min_judge_correctness=gate_row.get("min_judge_correctness"),
                min_judge_evidence_sufficiency=gate_row.get("min_judge_evidence_sufficiency"),
            )
        scenario_row = row.get("scenario_gates") or {}
        scenario_gates = ScenarioGates(
            min_abstain_correct_rate=scenario_row.get("min_abstain_correct_rate"),
            min_multihop_hop_coverage=scenario_row.get("min_multihop_hop_coverage"),
            min_multihop_lift_over_baseline=scenario_row.get("min_multihop_lift_over_baseline"),
        )
        if not any(
            value is not None
            for value in (
                scenario_gates.min_abstain_correct_rate,
                scenario_gates.min_multihop_hop_coverage,
                scenario_gates.min_multihop_lift_over_baseline,
            )
        ):
            scenario_gates = None
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
            scenario_gates=scenario_gates,
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
        if gate.max_redundancy_rate is not None:
            actual = values.get("redundancy_rate", 0)
            ok = actual <= gate.max_redundancy_rate
            checks.append(
                {
                    "strategy": strategy,
                    "metric": "redundancy_rate",
                    "expected_max": gate.max_redundancy_rate,
                    "actual": actual,
                    "passed": ok,
                }
            )
            if not ok:
                failures.append(f"{strategy} redundancy_rate {actual:.3f} > {gate.max_redundancy_rate}")
        if gate.min_citation_traceability is not None:
            actual = values.get("citation_traceability", 0)
            ok = actual >= gate.min_citation_traceability
            checks.append(
                {
                    "strategy": strategy,
                    "metric": "citation_traceability",
                    "expected_min": gate.min_citation_traceability,
                    "actual": actual,
                    "passed": ok,
                }
            )
            if not ok:
                failures.append(
                    f"{strategy} citation_traceability {actual:.3f} < {gate.min_citation_traceability}"
                )
        if gate.max_assembly_ms_p95 is not None:
            actual = values.get("assembly_ms_p95", 0)
            ok = actual <= gate.max_assembly_ms_p95
            checks.append(
                {
                    "strategy": strategy,
                    "metric": "assembly_ms_p95",
                    "expected_max": gate.max_assembly_ms_p95,
                    "actual": actual,
                    "passed": ok,
                }
            )
            if not ok:
                failures.append(f"{strategy} assembly_ms_p95 {actual:.1f} > {gate.max_assembly_ms_p95}")
        if gate.min_abstain_correct is not None:
            actual = values.get("abstain_correct")
            if actual is None:
                failures.append(f"{strategy} missing abstain_correct in measure summary")
            else:
                ok = actual >= gate.min_abstain_correct
                checks.append(
                    {
                        "strategy": strategy,
                        "metric": "abstain_correct",
                        "expected_min": gate.min_abstain_correct,
                        "actual": actual,
                        "passed": ok,
                    }
                )
                if not ok:
                    failures.append(f"{strategy} abstain_correct {actual:.3f} < {gate.min_abstain_correct}")
        if gate.min_judge_evidence_sufficiency is not None:
            actual = values.get("judge_evidence_sufficiency")
            if actual is None:
                failures.append(f"{strategy} missing judge_evidence_sufficiency (was live_llm enabled?)")
            else:
                ok = actual >= gate.min_judge_evidence_sufficiency
                checks.append(
                    {
                        "strategy": strategy,
                        "metric": "judge_evidence_sufficiency",
                        "expected_min": gate.min_judge_evidence_sufficiency,
                        "actual": actual,
                        "passed": ok,
                    }
                )
                if not ok:
                    failures.append(
                        f"{strategy} judge_evidence_sufficiency {actual:.3f} < {gate.min_judge_evidence_sufficiency}"
                    )

    if profile.scenario_gates is not None:
        scenarios = report.get("scenarios", {})
        sg = profile.scenario_gates
        if sg.min_abstain_correct_rate is not None:
            actual = scenarios.get("abstention", {}).get("summary", {}).get("abstain_correct_rate")
            if actual is None:
                failures.append("missing scenarios.abstention (run evaluate-pipeline, not measure alone)")
            else:
                ok = actual >= sg.min_abstain_correct_rate
                checks.append(
                    {
                        "scope": "scenarios.abstention",
                        "metric": "abstain_correct_rate",
                        "expected_min": sg.min_abstain_correct_rate,
                        "actual": actual,
                        "passed": ok,
                    }
                )
                if not ok:
                    failures.append(
                        f"abstain_correct_rate {actual:.3f} < {sg.min_abstain_correct_rate}"
                    )
        if sg.min_multihop_hop_coverage is not None:
            actual = (
                scenarios.get("multihop", {})
                .get("summary", {})
                .get("multihop_fair_budget_pipeline", {})
                .get("hop_coverage")
            )
            if actual is None:
                failures.append("missing scenarios.multihop (run evaluate-pipeline, not measure alone)")
            else:
                ok = actual >= sg.min_multihop_hop_coverage
                checks.append(
                    {
                        "scope": "scenarios.multihop",
                        "metric": "hop_coverage",
                        "expected_min": sg.min_multihop_hop_coverage,
                        "actual": actual,
                        "passed": ok,
                    }
                )
                if not ok:
                    failures.append(f"multihop hop_coverage {actual:.3f} < {sg.min_multihop_hop_coverage}")
        if sg.min_multihop_lift_over_baseline is not None:
            mh_summary = scenarios.get("multihop", {}).get("summary", {})
            baseline = mh_summary.get("single_query_baseline", {}).get("hop_coverage")
            multihop = mh_summary.get("multihop_fair_budget_pipeline", {}).get("hop_coverage")
            if baseline is None or multihop is None:
                failures.append("missing multihop baseline/multihop summary for lift check")
            else:
                lift = multihop - baseline
                ok = lift >= sg.min_multihop_lift_over_baseline
                checks.append(
                    {
                        "scope": "scenarios.multihop",
                        "metric": "lift_over_baseline",
                        "expected_min": sg.min_multihop_lift_over_baseline,
                        "actual": lift,
                        "passed": ok,
                    }
                )
                if not ok:
                    failures.append(
                        f"multihop lift {lift:.3f} < {sg.min_multihop_lift_over_baseline}"
                    )

    return {
        "profile": profile.name,
        "passed": not failures,
        "failures": failures,
        "checks": checks,
    }
