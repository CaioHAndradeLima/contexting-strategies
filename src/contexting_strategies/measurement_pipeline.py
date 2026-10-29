from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from statistics import mean
from time import perf_counter

from .assemblers import ContextAssembler, default_assemblers
from .heuristic_rerank import HeuristicReranker
from .llm_judge import OpenAIAnswerGenerator, OpenAIAnswerJudge, OpenAIContextJudge
from .metrics import score_context
from .models import Hit, QueryCase, ScoreWeights
from .pipeline_config import PipelineConfig
from .pricing import PRICING_AS_OF, PRICING_SOURCE, OPENAI_STANDARD_RATES_PER_MILLION, CallBudget, combine_usage
from .scenario_evaluation import abstain_correct
from .sufficiency import apply_sufficiency_gate


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def select_assemblers(names: Sequence[str] | None) -> list[ContextAssembler]:
    available = {assembler.name: assembler for assembler in default_assemblers()}
    if not names:
        return [available["pipeline_acl_dedupe_score_diverse"], available["weighted_score"]]
    selected = []
    for name in names:
        if name not in available:
            raise ValueError(f"unknown assembler '{name}'; choose from {sorted(available)}")
        selected.append(available[name])
    return selected


def run_measurement_pipeline(
    queries: Iterable[QueryCase],
    retriever: Callable[[str, int], list[Hit]],
    config: PipelineConfig,
    weights: ScoreWeights | None = None,
    assemblers: Sequence[ContextAssembler] | None = None,
    *,
    context_judge: OpenAIContextJudge | None = None,
    answer_generator: OpenAIAnswerGenerator | None = None,
    answer_judge: OpenAIAnswerJudge | None = None,
) -> dict:
    weights = weights or ScoreWeights()
    assembler_list = list(assemblers or select_assemblers(None))
    reranker = HeuristicReranker() if config.apply_heuristic_rerank else None
    rows = []
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    llm_usages: dict[str, list[dict]] = defaultdict(list)

    for case in queries:
        started = perf_counter()
        candidates = retriever(case.query, config.candidate_k)
        retrieval_ms = (perf_counter() - started) * 1000
        if reranker is not None:
            candidate_input = reranker.rerank(case.query, candidates, config.candidate_k)
        else:
            candidate_input = candidates

        for assembler in assembler_list:
            asm_started = perf_counter()
            context = apply_sufficiency_gate(
                assembler.assemble(case, candidate_input, config.budget_tokens, weights),
                case,
                min_coverage=config.min_coverage,
            )
            assembly_ms = (perf_counter() - asm_started) * 1000
            offline = score_context(context, case)
            offline["abstain_correct"] = abstain_correct(context, case)
            row = {
                "query_id": case.id,
                "strategy": assembler.name,
                "answerable": case.answerable,
                "retrieval_ms": retrieval_ms,
                "assembly_ms": assembly_ms,
                "should_abstain": context.should_abstain,
                "chunk_ids": [chunk.document.id for chunk in context.chunks],
                "offline": offline,
                "llm": {},
            }

            if config.live_llm and context_judge and answer_generator and answer_judge:
                ctx_j = context_judge.judge(case, context.rendered)
                row["llm"]["context_judge"] = ctx_j
                llm_usages["context_judge"].append(ctx_j)
                gen = answer_generator.generate(case, context.rendered, should_abstain=context.should_abstain)
                row["llm"]["generation"] = {k: v for k, v in gen.items() if k != "answer"}
                row["answer"] = gen["answer"]
                llm_usages["generation"].append(gen)
                evidence = [chunk.document for chunk in context.chunks]
                ans_j = answer_judge.judge(case, gen["answer"], evidence)
                row["llm"]["answer_judge"] = ans_j
                llm_usages["answer_judge"].append(ans_j)

            rows.append(row)
            by_strategy[assembler.name].append(row)

    summary = {}
    for name, strategy_rows in by_strategy.items():
        offline_metrics = strategy_rows[0]["offline"].keys()
        summary[name] = {
            **{metric: mean(r["offline"][metric] for r in strategy_rows) for metric in offline_metrics},
            "assembly_ms_p95": _percentile([r["assembly_ms"] for r in strategy_rows], 0.95),
        }
        if config.live_llm:
            ctx_scores = [r["llm"]["context_judge"]["evidence_sufficiency"] for r in strategy_rows if r.get("llm")]
            ans_g = [r["llm"]["answer_judge"]["groundedness"] for r in strategy_rows if r.get("llm")]
            ans_c = [r["llm"]["answer_judge"]["correctness"] for r in strategy_rows if r.get("llm")]
            if ctx_scores:
                summary[name]["judge_evidence_sufficiency"] = mean(ctx_scores)
            if ans_g:
                summary[name]["judge_groundedness"] = mean(ans_g)
            if ans_c:
                summary[name]["judge_correctness"] = mean(ans_c)

    experiment = {
        "live_llm": config.live_llm,
        "answer_model": config.answer_model if config.live_llm else None,
        "judge_model": config.judge_model if config.live_llm else None,
        "max_llm_calls": config.max_llm_calls,
        "pricing_as_of": PRICING_AS_OF,
        "pricing_source": PRICING_SOURCE,
        "pricing_rates_usd_per_million_tokens": OPENAI_STANDARD_RATES_PER_MILLION,
    }
    if config.live_llm:
        experiment["llm_usage"] = {
            stage: combine_usage(usages) for stage, usages in llm_usages.items()
        }
        experiment["llm_usage_total"] = combine_usage(
            usage for usages in llm_usages.values() for usage in usages
        )

    recommendation = None
    if summary:

        def _rank_key(item):
            name, values = item
            primary = values.get("judge_groundedness", values.get("evidence_coverage", 0))
            return (primary, -values.get("unauthorized_leak", 0), -values.get("redundancy_rate", 0), name)

        recommendation = max(summary.items(), key=_rank_key)[0]

    return {
        "experiment": experiment,
        "config": {
            "candidate_k": config.candidate_k,
            "budget_tokens": config.budget_tokens,
            "apply_heuristic_rerank": config.apply_heuristic_rerank,
            "assembler_names": [a.name for a in assembler_list],
        },
        "summary": summary,
        "recommendation": recommendation,
        "rows": rows,
    }
