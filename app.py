import re
from fastapi import HTTPException

def get_context_line_from_raw(raw: str) -> str:
    # prend la 2e ligne de heading si dispo (##### ...)
    lines = [l.strip() for l in raw.replace("\r", "").split("\n") if l.strip()]
    heading = [l for l in lines if l.startswith("#")]
    if len(heading) >= 2:
        return heading[1]
    # fallback: 2e ligne non vide
    return lines[1] if len(lines) > 1 else ""

def strip_md(line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", line).strip()

def split_ctx(line: str):
    # line like: "Cerise Media (...) - Marcq ... | de juin 2018 à oct 2024"
    txt = strip_md(line)
    left, *rest = [p.strip() for p in txt.split("|")]
    date_range_raw = rest[0] if rest else None

    parts = re.split(r"\s-\s", left, maxsplit=1)
    company_part = parts[0].strip() if parts else left.strip()
    location_raw = parts[1].strip() if len(parts) > 1 else None
    return company_part, location_raw, date_range_raw

def parse_group_hints(company_part: str):
    # "(Prisma Media / Vivendi)" -> ["Prisma Media", "Vivendi"]
    m = re.search(r"\((.*?)\)", company_part)
    if not m:
        return []
    inside = m.group(1)
    hints = [x.strip() for x in re.split(r"[/,;]| et ", inside) if x.strip()]
    return hints[:10]

def clean_company_query(company_part: str):
    # remove parentheses content
    q = re.sub(r"\s*\(.*?\)\s*", " ", company_part).strip()
    # remove trailing separators
    q = re.sub(r"\s{2,}", " ", q)
    return q

@app.post("/v1/xp/company")
def xp_company(payload: dict):
    raw = payload.get("raw", "")
    if not raw or not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="raw is required")

    ctx_line = get_context_line_from_raw(raw)
    if not ctx_line:
        raise HTTPException(status_code=422, detail="Could not find context line")

    company_part, location_raw, date_range_raw = split_ctx(ctx_line)

    company_display = strip_md(company_part)
    company_group_hints = parse_group_hints(company_display)

    company_query = clean_company_query(company_display)

    # spaCy refinement: keep longest ORG if present (optional but useful)
    doc = nlp(company_query)
    orgs = [e.text.strip() for e in doc.ents if e.label_ == "ORG"]
    if orgs:
        company_query = max(orgs, key=len)

    return {
        "ctx_line": strip_md(ctx_line),
        "company_display": company_display,
        "company_query": company_query,
        "company_group_hints": company_group_hints,
        "location_raw": location_raw,
        "date_range_raw": date_range_raw,
    }
