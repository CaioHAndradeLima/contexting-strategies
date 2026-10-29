from .models import ApplicationContext, AssembledContext, StructuredResult
from .reliability import sanitize_chunk_text
from .tokens import estimate_tokens


def render_full_prompt(
    *,
    system_policy: str,
    evidence: AssembledContext,
    structured: StructuredResult | None,
    history_text: str,
    application: ApplicationContext,
) -> str:
    """Render the five context inputs from context-info-for-output (assembly order)."""
    evidence_blocks = []
    for chunk in evidence.chunks:
        safe_text, _ = sanitize_chunk_text(chunk.document.text)
        evidence_blocks.append(
            f"[{chunk.citation_id}] doc_id={chunk.document.id}\n"
            f"{chunk.document.title}\n{safe_text}"
        )
    structured_block = ""
    if structured is not None:
        structured_block = (
            f"Structured results ({structured.label}):\n"
            f"columns={','.join(structured.columns)}\n"
            f"{structured.render_table()}\n"
            f"units_note={structured.units_note}"
        )
    parts = [
        f"System policy:\n{system_policy.strip()}",
        "Retrieved evidence:\n" + ("\n\n".join(evidence_blocks) if evidence_blocks else "(none)"),
    ]
    if structured_block:
        parts.append(structured_block)
    if history_text.strip():
        parts.append(f"Conversation history:\n{history_text.strip()}")
    parts.append(
        "Current request:\n"
        f"question={application.question}\n"
        f"locale={application.locale}\n"
        f"output_format={application.output_format}"
    )
    rendered = "\n\n".join(parts)
    return rendered


def token_count(rendered: str) -> int:
    return estimate_tokens(rendered)
