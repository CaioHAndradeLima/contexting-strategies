import re

from .assemblers import PipelineAssembler
from .full_prompt import render_full_prompt
from .models import AnalyticalCase, ApplicationContext, Hit, ScoreWeights, StructuredResult
from .sufficiency import apply_sufficiency_gate
from .tokens import estimate_tokens

METRIC_TOKEN = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def numeric_tokens(text: str) -> set[str]:
    return set(METRIC_TOKEN.findall(text))


def structured_numeric_support(answer: str, structured: StructuredResult) -> float:
    answer_nums = numeric_tokens(answer)
    if not answer_nums:
        return 1.0
    table_text = structured.render_table() + " " + structured.units_note
    table_nums = numeric_tokens(table_text)
    supported = answer_nums & table_nums
    return len(supported) / len(answer_nums)


def required_metrics_present(rendered: str, required: tuple[str, ...]) -> float:
    if not required:
        return 1.0
    found = sum(1 for metric in required if metric.lower() in rendered.lower())
    return found / len(required)


def assemble_analytical(
    case: AnalyticalCase,
    hits: list[Hit],
    structured: StructuredResult,
    weights: ScoreWeights,
    *,
    budget_tokens: int = 200,
    system_policy: str = "Answer using structured query results for numbers; use documents for definitions only.",
) -> dict:
    pipeline = PipelineAssembler()
    pseudo = _as_query_case(case)
    evidence = apply_sufficiency_gate(
        pipeline.assemble(pseudo, hits, budget_tokens // 2, weights),
        pseudo,
        min_coverage=0.34,
    )
    application = ApplicationContext(question=case.query)
    rendered = render_full_prompt(
        system_policy=system_policy,
        evidence=evidence,
        structured=structured,
        history_text="",
        application=application,
    )
    return {
        "rendered": rendered,
        "token_count": estimate_tokens(rendered),
        "structured_block_present": structured.label in rendered,
        "required_metrics_score": required_metrics_present(rendered, case.required_metrics),
        "evidence_coverage": evidence.evidence_coverage,
        "should_abstain": evidence.should_abstain,
        "chunk_ids": [chunk.document.id for chunk in evidence.chunks],
    }


def _as_query_case(case: AnalyticalCase):
    from .models import QueryCase

    return QueryCase(
        id=case.id,
        query=case.query,
        relevance=case.relevance,
        tags=case.tags,
        answerable=case.answerable,
        reference_answer=case.reference_answer,
    )
