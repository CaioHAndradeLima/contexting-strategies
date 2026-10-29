import re

from .models import StructuredResult

_SELECT_RE = re.compile(
    r"^\s*SELECT\s+(?P<cols>[\w\s,*]+)\s+FROM\s+(?P<table>[a-zA-Z0-9_]+)"
    r"(?:\s+WHERE\s+(?P<where>.+))?\s*$",
    re.I,
)


def run_validated_select(sql: str, structured: StructuredResult) -> dict:
    """
    Parameterized, allowlisted SELECT only (context-analytical-question.md).
    Table name must match structured.label.
    """
    match = _SELECT_RE.match(sql.strip())
    if not match:
        raise ValueError("only simple SELECT ... FROM ... WHERE ... queries are allowed")
    table = match.group("table")
    if table != structured.label:
        raise ValueError(f"table '{table}' does not match structured label '{structured.label}'")
    where = (match.group("where") or "").strip()
    rows = structured.rows
    if where:
        # Support: column = 'value'
        eq = re.match(r"(\w+)\s*=\s*'([^']+)'", where)
        if not eq:
            raise ValueError("WHERE clause must be column = 'value' in the lab")
        col = eq.group(1)
        val = eq.group(2)
        if col not in structured.columns:
            raise ValueError(f"unknown column {col}")
        idx = structured.columns.index(col)
        rows = [row for row in rows if str(row[idx]) == val]
    return {
        "columns": structured.columns,
        "rows": rows,
        "row_count": len(rows),
        "validated": True,
    }
