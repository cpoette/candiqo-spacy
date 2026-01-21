import os
import re
import time
from fastapi import FastAPI, HTTPException
import spacy

# --- Config
MODEL = os.getenv("SPACY_MODEL_FR", "fr_core_news_md")
MAX_CHARS = int(os.getenv("SPACY_MAX_CHARS", "120000"))

# --- App must be defined BEFORE decorators
app = FastAPI(title="candiqo-spacy", version="0.2.0")

# --- Load model at startup (module import)
LOAD_TS = time.time()
nlp = spacy.load(MODEL)

# ------------------------
# Helpers
# ------------------------
def strip_md(line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", (line or "")).strip()

def get_context_line_from_raw(raw: str) -> str:
    lines = [l.strip() for l in (raw or "").replace("\r", "").split("\n") if l.strip()]
    heading = [l for l in lines if l.startswith("#")]
    # pattern: 1st heading = title, 2nd heading = context line
    if len(heading) >= 2:
        return heading[1]
    # fallback: second non-empty line
    return lines[1] if len(lines) > 1 else ""

def parse_group_hints(company_part: str):
    m = re.search(r"\((.*?)\)", company_part or "")
    if not m:
        return []
    inside = m.group(1)
    hints = [x.strip() for x in re.split(r"[/,;]| et ", inside) if x.strip()]
    return hints[:10]

def clean_company_query(company_part: str):
    q = re.sub(r"\s*\(.*?\)\s*", " ", company_part or "").strip()
    q = re.sub(r"\s{2,}", " ", q)
    return q

# ------------------------
# Routes
# ------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "loaded_at": LOAD_TS,
    }

@app.post("/debug/ents")
def debug_ents(payload: dict):
    text = payload.get("text", "")
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="text must be a string")
    if len(text) > MAX_CHARS:
        raise HTTPException(status_code=413, detail=f"text too large (>{MAX_CHARS} chars)")
    doc = nlp(text)
    return [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char} for e in doc.ents]

@app.post("/v1/xp/company")
def xp_company(payload: dict):
    raw = payload.get("raw", "")
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(status_code=400, detail="raw is required (string)")
    if len(raw) > MAX_CHARS:
        raise HTTPException(status_code=413, detail=f"raw too large (>{MAX_CHARS} chars)")

    ctx_line = get_context_line_from_raw(raw)
    if not ctx_line:
        raise HTTPException(status_code=422, detail="Could not find context line")

    txt = strip_md(ctx_line)

    # Split dates (right of "|")
    left, *rest = [p.strip() for p in txt.split("|")]
    date_range_raw = rest[0] if rest else None

    # Split company vs location (on " - ")
    parts = re.split(r"\s-\s", left, maxsplit=1)
    company_part = parts[0].strip() if parts else left.strip()
    location_raw = parts[1].strip() if len(parts) > 1 else None

    company_group_hints = parse_group_hints(company_part)
    company_query = clean_company_query(company_part)

    # spaCy refinement: choose longest ORG if any
    doc = nlp(company_query)
    orgs = [e.text.strip() for e in doc.ents if e.label_ == "ORG"]
    if orgs:
        company_query = max(orgs, key=len)

    return {
        "ctx_line": txt,
        "company_display": company_part,
        "company_query": company_query,
        "company_group_hints": company_group_hints,
        "location_raw": location_raw,
        "date_range_raw": date_range_raw,
    }
