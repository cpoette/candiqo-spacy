import os
import time
from fastapi import FastAPI
import spacy

MODEL = os.getenv("SPACY_MODEL_FR", "fr_core_news_md")

LOAD_TS = time.time()
nlp = spacy.load(MODEL)

app = FastAPI(title="candiqo-spacy", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "loaded_at": LOAD_TS}

@app.post("/debug/ents")
def debug_ents(payload: dict):
    text = payload.get("text", "")
    doc = nlp(text)
    return [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char} for e in doc.ents]
