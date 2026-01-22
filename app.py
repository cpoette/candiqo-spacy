import os
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
import spacy

# --- Config
MODEL = os.getenv("SPACY_MODEL_FR", "fr_core_news_md")
MAX_CHARS = int(os.getenv("SPACY_MAX_CHARS", "120000"))
DEBUG_DEFAULT = os.getenv("SPACY_DEBUG_DEFAULT", "false").lower() == "true"

app = FastAPI(title="candiqo-spacy", version="0.3.0")

LOAD_TS = time.time()
nlp = spacy.load(MODEL)

# ------------------------
# Regex / helpers
# ------------------------

RX_MD_HEADING = re.compile(r"^#{1,6}\s+")
RX_PIPE = re.compile(r"\|")
RX_DASH_SPLIT = re.compile(r"\s[-—–]\s")
RX_PARENS = re.compile(r"\((.*?)\)")
RX_MULTI_SPACE = re.compile(r"\s{2,}")
RX_MAILTO = re.compile(r"\bmailto:\s*", re.I)

# Date-ish tokens (loose; only used to FIND/ISOLATE a raw range, not parse)
MONTHS_FR = r"(janv\.?|janvier|f[eé]v\.?|f[eé]vr\.?|f[eé]vrier|mars|avr\.?|avril|mai|juin|juil\.?|juillet|ao[uû]t|sept\.?|septembre|oct\.?|octobre|nov\.?|novembre|d[eé]c\.?|d[eé]cembre)"
MONTHS_EN = r"(jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|sept\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)"
RX_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
RX_MMYYYY = re.compile(r"\b(0?[1-9]|1[0-2])\s*[\/\.-]\s*(19\d{2}|20\d{2})\b")
RX_PRESENT = re.compile(r"\b(pr[eé]sent|present|actuel|actuellement|aujourd['’]hui|en\s*cours|current|now|today)\b", re.I)

# very tolerant range detector (keeps raw)
RX_DATE_RANGE = re.compile(
    rf"(?P<range>(?:{MONTHS_FR}|{MONTHS_EN})?\s*{RX_YEAR.pattern}"
    rf"\s*(?:[-–—]|to|\bà\b|\bau\b|\bjusqu)\s*"
    rf"(?:(?:{MONTHS_FR}|{MONTHS_EN})?\s*)?(?:{RX_YEAR.pattern}|{RX_PRESENT.pattern}))",
    re.I
)

RX_SINCE = re.compile(
    rf"\b(?:depuis|since)\s+(?:(?:{MONTHS_FR}|{MONTHS_EN})\s*)?(?:{RX_YEAR.pattern}|{RX_MMYYYY.pattern})",
    re.I
)

def strip_md(line: str) -> str:
    return RX_MD_HEADING.sub("", (line or "")).strip()

def clean_common(s: str) -> str:
    t = (s or "")
    t = t.replace("\r", "")
    t = RX_MAILTO.sub("", t)
    t = RX_MULTI_SPACE.sub(" ", t)
    return t.strip()

def lines_nonempty(text: str) -> List[str]:
    return [l.strip() for l in clean_common(text).split("\n") if l.strip()]

def parse_group_hints(company_part: str) -> List[str]:
    m = RX_PARENS.search(company_part or "")
    if not m:
        return []
    inside = m.group(1)
    hints = [x.strip() for x in re.split(r"[/,;]| et ", inside) if x.strip()]
    return hints[:10]

def clean_company_query(company_part: str) -> str:
    q = re.sub(r"\s*\(.*?\)\s*", " ", company_part or "").strip()
    q = RX_MULTI_SPACE.sub(" ", q)
    return q

def spacy_best_org(text: str) -> Optional[str]:
    if not text:
        return None
    doc = nlp(text)
    orgs = [e.text.strip() for e in doc.ents if e.label_ == "ORG"]
    if not orgs:
        return None
    # prefer longest span (often the real org name)
    return max(orgs, key=len)

def detect_date_range_raw(s: str) -> Optional[str]:
    """Find a date-ish range in a string, return raw substring (no parsing)."""
    if not s:
        return None
    m = RX_DATE_RANGE.search(s)
    if m:
        return m.group("range").strip()
    m2 = RX_SINCE.search(s)
    if m2:
        return m2.group(0).strip()
    # allow MM/YYYY ranges like "06/2021 - présent" to be caught by DATE_RANGE sometimes,
    # else just return first MMYYYY occurrence if present-ish elsewhere (weak)
    return None

def split_company_location(left_part: str) -> (str, Optional[str]):
    """Split on ' - ' if present; keep raw."""
    parts = RX_DASH_SPLIT.split(left_part, maxsplit=1)
    company_part = parts[0].strip() if parts else left_part.strip()
    location_raw = parts[1].strip() if len(parts) > 1 else None
    return company_part, location_raw

def split_title_company(ctx: str) -> (Optional[str], Optional[str]):
    """
    Try to split '... : Title | Company - Location' or 'Title | Company ...'
    Returns (title_raw, rest_after_title)
    """
    if not ctx:
        return None, None

    # Prefer ":" as title separator if present
    if ":" in ctx:
        before, after = ctx.split(":", 1)
        # if before looks like date-ish, title is after
        if RX_YEAR.search(before) or RX_MMYYYY.search(before) or RX_PRESENT.search(before):
            return after.strip(), None
        # else keep after as title
        return after.strip(), None

    return ctx.strip(), None

def get_ctx_line(payload_raw: str, meta: Dict[str, Any]) -> (str, str, List[str]):
    """
    Build a robust context line from XP raw + hints.
    Strategy:
    - if headings exist: first heading line is ctx
    - else: first non-empty line
    - then enrich with meta hints if missing bits
    """
    used_hints: List[str] = []
    strat = "unknown"

    ls = lines_nonempty(payload_raw)
    if not ls:
        return "", "empty", used_hints

    headings = [l for l in ls if l.startswith("#")]
    if headings:
        ctx = strip_md(headings[0])
        strat = "heading_first"
    else:
        ctx = ls[0]
        strat = "first_non_empty"

    ctx = clean_common(ctx)

    # If ctx has no pipe but we have company_hint, we can append
    # Keep conservative: only if company_hint exists and doesn't already appear
    if meta:
        if meta.get("date_hint") and not detect_date_range_raw(ctx):
            # prefix with date_hint if it seems like a date-ish token
            ctx = f"{meta['date_hint'].strip()} : {ctx}"
            used_hints.append("date_hint")

        if meta.get("company_hint") and ("|" not in ctx):
            ch = str(meta["company_hint"]).strip()
            if ch and ch.lower() not in ctx.lower():
                ctx = f"{ctx} | {ch}"
                used_hints.append("company_hint")

        if meta.get("location_hint") and (" - " not in ctx):
            lh = str(meta["location_hint"]).strip()
            if lh and lh.lower() not in ctx.lower():
                ctx = f"{ctx} - {lh}"
                used_hints.append("location_hint")

        if used_hints:
            strat = strat + "|hint_merge"

    return ctx.strip(), strat, used_hints

def confidence_score(title_raw: Optional[str], company: Optional[str], date_range_raw: Optional[str]) -> float:
    score = 0.0
    if date_range_raw: score += 0.35
    if title_raw and len(title_raw.split()) >= 2: score += 0.30
    if company and len(company) >= 2: score += 0.30
    # cap
    return max(0.0, min(1.0, score))

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

@app.post("/v1/xp/parse")
def xp_parse(payload: dict):
    """
    Input:
      - id: optional string
      - raw: string (XP raw block)
      - meta: optional hints {title_hint, company_hint, location_hint, date_hint}
      - debug: optional bool
    Output: structured xp parse (no date normalization)
    """
    raw = payload.get("raw", "")
    xp_id = payload.get("id")
    meta = payload.get("meta") or {}
    debug = bool(payload.get("debug", DEBUG_DEFAULT))

    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(status_code=400, detail="raw is required (string)")
    if len(raw) > MAX_CHARS:
        raise HTTPException(status_code=413, detail=f"raw too large (>{MAX_CHARS} chars)")
    if meta and not isinstance(meta, dict):
        raise HTTPException(status_code=400, detail="meta must be an object")

    ctx_line, ctx_strategy, used_hints = get_ctx_line(raw, meta)
    if not ctx_line:
        raise HTTPException(status_code=422, detail="Could not find context line")

    ctx = clean_common(ctx_line)

    # Split pipe => right part tends to be date or company depending on formats.
    # We'll interpret conservatively:
    # - If pipe exists, left is "main", right is "tail"
    # - Try detect date in whole ctx first (more robust)
    date_range_raw = detect_date_range_raw(ctx)

    left, right = None, None
    if "|" in ctx:
        parts = [p.strip() for p in ctx.split("|", 1)]
        left = parts[0]
        right = parts[1] if len(parts) > 1 else ""
    else:
        left = ctx
        right = ""

    # Company + location mostly live in right if "Title | Company - Location"
    # But some CVs do "Company - Location | dates"
    company_part = ""
    location_raw = None

    # Heuristic: if right exists and looks like org-ish, prefer it for company parsing
    candidate_for_company = right or left

    # Remove date range substring from candidate when possible (keep raw date separately)
    if date_range_raw and candidate_for_company:
        candidate_for_company = candidate_for_company.replace(date_range_raw, " ").strip()
        candidate_for_company = RX_MULTI_SPACE.sub(" ", candidate_for_company)

    # If candidate still contains '-', split company/location
    company_part, location_raw = split_company_location(candidate_for_company)

    # Title detection: try after ":" in left if left includes date-ish
    title_raw = None
    if left:
        # remove date fragment from left to isolate title candidates
        left_wo_date = left
        if date_range_raw:
            left_wo_date = left_wo_date.replace(date_range_raw, " ").strip()
            left_wo_date = RX_MULTI_SPACE.sub(" ", left_wo_date)

        if ":" in left_wo_date:
            # take after colon
            title_raw = left_wo_date.split(":", 1)[1].strip()
        else:
            # fallback to meta hint
            th = str(meta.get("title_hint", "")).strip()
            if th:
                title_raw = th
            else:
                # last resort: if left looks like pure title
                title_raw = left_wo_date.strip() if left_wo_date else None

    # Company query + group hints
    company_group_hints = parse_group_hints(company_part)
    company_query = clean_company_query(company_part)

    # spaCy refine ORG
    sp_org = spacy_best_org(company_query)
    if sp_org and len(sp_org) >= 2:
        company_query = sp_org

    warnings: List[str] = []
    if not date_range_raw:
        # not fatal (junior / missing dates)
        warnings.append("NO_DATE_RANGE_DETECTED")
    if not company_query:
        warnings.append("NO_COMPANY_DETECTED")
    if not title_raw or len(title_raw) < 3:
        warnings.append("WEAK_TITLE_DETECTED")

    conf = confidence_score(title_raw, company_query, date_range_raw)

    out: Dict[str, Any] = {
        "id": xp_id,
        "ctx_line": ctx,
        "title_raw": title_raw,
        "company_display": company_part.strip(),
        "company_query": company_query,
        "company_group_hints": company_group_hints,
        "location_raw": location_raw,
        "date_range_raw": date_range_raw,
        "confidence": conf,
        "warnings": warnings,
    }

    if debug:
        out["debug"] = {
            "ctx_line_strategy": ctx_strategy,
            "used_meta_hints": used_hints,
            "left": left,
            "right": right,
            "candidate_for_company": candidate_for_company,
        }

    return out


@app.post("/v1/xp/extract_simple")
def xp_extract_simple(payload: dict):
    """
    Simple entity extraction approach:
    - Use spaCy to tag ORG, DATE, LOC
    - Everything else = likely job title
    
    Input:
      - text: string (single XP line or block)
      - debug: optional bool
    
    Output:
      - orgs: list of ORG entities
      - dates: list of DATE entities  
      - locations: list of LOC entities
      - job_title: everything that's not tagged (cleaned)
    """
    text = payload.get("text", "")
    debug = bool(payload.get("debug", DEBUG_DEFAULT))
    
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="text is required (string)")
    if len(text) > MAX_CHARS:
        raise HTTPException(status_code=413, detail=f"text too large (>{MAX_CHARS} chars)")
    
    # Clean input
    text = clean_common(text)
    
    # SpaCy NER
    doc = nlp(text)
    
    orgs = [e.text.strip() for e in doc.ents if e.label_ == "ORG"]
    dates = [e.text.strip() for e in doc.ents if e.label_ == "DATE"]
    locations = [e.text.strip() for e in doc.ents if e.label_ == "LOC"]
    
    # Build set of tagged token indices
    tagged_indices = set()
    for ent in doc.ents:
        for i in range(ent.start, ent.end):
            tagged_indices.add(i)
    
    # Extract job title = non-tagged, non-stop, non-punct tokens
    job_title_tokens = [
        token.text for token in doc 
        if token.i not in tagged_indices 
        and not token.is_punct 
        and not token.is_stop
        and token.text.strip()
    ]
    job_title = " ".join(job_title_tokens).strip()
    
    result = {
        "orgs": orgs,
        "dates": dates,
        "locations": locations,
        "job_title": job_title,
    }
    
    if debug:
        result["debug"] = {
            "all_entities": [
                {"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char} 
                for e in doc.ents
            ],
            "tagged_token_count": len(tagged_indices),
            "total_token_count": len(doc),
        }
    
    return result