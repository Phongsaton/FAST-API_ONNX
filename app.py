import os
import sys
import time
import re
import csv
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Sequence

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

# ==== Simple ONNX Encoder (no external project dependency) ====
import onnxruntime as ort
from transformers import AutoTokenizer

APP_VERSION = "1.3.1-feedback-A"

# ===== Paths & config =====
ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

def _load_yaml_config() -> Dict[str, Any]:
    cfg_path = os.getenv("API_CONFIG", str(ROOT_DIR / "api_config.yaml"))
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[config:warn] cannot read {cfg_path}: {e}")
    return {}

_CFG = _load_yaml_config()

def _resolve_path(p: Optional[str], default: Path) -> str:
    if not p:
        return str(default)
    p = str(p)
    return p if os.path.isabs(p) else str((ROOT_DIR / p).resolve())

CSV_PATH = _resolve_path(_CFG.get("data", {}).get("csv_path"), ROOT_DIR / "data" / "QA.csv")
CSV_PATH = os.getenv("CSV_PATH", CSV_PATH)

CSV_ENCODING_ENV = (
    os.getenv("CSV_ENCODING", "").strip()
    or (_CFG.get("data", {}).get("encoding") or "").strip()
    or None
)

CORPUS_PKL = _resolve_path(_CFG.get("index", {}).get("corpus_pkl"), ROOT_DIR / "outputs" / "corpus_embeds.pkl")
CORPUS_PKL = os.getenv("INDEX_PKL", CORPUS_PKL)

ONNX_DIR = _resolve_path(_CFG.get("runtime", {}).get("onnx_dir"), ROOT_DIR / "onnx_path")
ONNX_DIR = os.getenv("ONNX_DIR", ONNX_DIR)

TOKENIZER_DIR = _resolve_path(_CFG.get("runtime", {}).get("tokenizer_dir"), ONNX_DIR)

FEEDBACK_CSV = _resolve_path(_CFG.get("feedback", {}).get("csv_path"), ROOT_DIR / "data" / "user_feedback.csv")
FEEDBACK_CSV = os.getenv("FEEDBACK_CSV", FEEDBACK_CSV)

FEEDBACK_FIELDS = [
    "log_id", "ts_iso", "session_id", "route",
    "q",
    "best_module", "best_question", "best_answer", "best_score",
    "alpha", "hybrid", "min_score", "topk",
    "n_rows", "dim", "onnx_dir",
    "latency_ms",
    "label", "feedback_ts_iso",
    "alternatives_json"
]

# ===== Guardrails =====
USE_GUARD = bool(int(os.getenv("GR_USE", "1")))
GR_MIN_TOP1 = float(os.getenv("GR_MIN_TOP1", "0.80"))
GR_MIN_SCORE = float(os.getenv("GR_MIN_SCORE", "0.78"))
GR_MIN_KW = float(os.getenv("GR_MIN_KW", "0.02"))
GR_MIN_OVERLAP = float(os.getenv("GR_MIN_OVERLAP", "0.15"))
GR_MIN_LEN = int(os.getenv("GR_MIN_LEN", "2"))
GR_MIN_MARGIN = float(os.getenv("GR_MIN_MARGIN", "0.02"))
GR_MIN_LEN_CHARS = int(os.getenv("GR_MIN_LEN_CHARS", "4"))

# ===== Thai-friendly tokenization =====
THAI_TOKEN_RE = re.compile(r"[A-Za-z0-9\u0E00-\u0E7F]+")

def tokenize_unicode(text: str) -> List[str]:
    text = (text or "").lower()
    return THAI_TOKEN_RE.findall(text)

def has_thai(text: str) -> bool:
    return any("\u0E00" <= ch <= "\u0E7F" for ch in text or "")

# ===== Globals =====
DF: Optional[pd.DataFrame] = None
DOCS: List[str] = []
DOC_META: List[Dict[str, Any]] = []
EMB: Optional[np.ndarray] = None  # (N, D) float32 L2-normalized
DIM: int = 0
N_ROWS: int = 0

VECTORIZER: Optional[TfidfVectorizer] = None
TFIDF_MAT = None
TFIDF_VOCAB: int = 0

# ===== Simple ONNX encoder =====
class SimpleOnnxEncoder:
    """
    Load encoder.onnx + tokenizer from ONNX_DIR.
    Supports two cases:
      1) ONNX outputs 'embedding' directly
      2) ONNX outputs 'last_hidden_state' -> mean-pool -> L2 normalize
    """
    def __init__(self, model_dir: str, providers: Optional[Sequence[str]] = None):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        onnx_path = str(Path(model_dir) / "encoder.onnx")
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CPUExecutionProvider"]
        )
        self.out_names = [o.name for o in self.session.get_outputs()]
        self.has_embedding = "embedding" in self.out_names
        # default to last output if naming is different
        self.lhs_name = "last_hidden_state" if "last_hidden_state" in self.out_names else self.out_names[0]

    @staticmethod
    def _mean_pool(lhs: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = attention_mask.astype(np.float32)[:, :, None]  # [B,T,1]
        summed = (lhs * mask).sum(axis=1)                     # [B,H]
        counts = np.clip(mask.sum(axis=1), 1e-9, None)        # [B,1]
        return summed / counts

    @staticmethod
    def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.sqrt((x * x).sum(axis=1, keepdims=True)) + eps
        return x / n

    def encode_queries(self, texts: List[str], max_len: int = 256, batch_size: int = 32) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tok = self.tokenizer(batch, padding="max_length", truncation=True, max_length=max_len)
            inputs = {
                "input_ids": np.asarray(tok["input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(tok["attention_mask"], dtype=np.int64),
            }
            outs = self.session.run(None, inputs)
            if self.has_embedding:
                emb = outs[self.out_names.index("embedding")].astype(np.float32)
            else:
                lhs = outs[self.out_names.index(self.lhs_name)]  # [B,T,H]
                emb = self._mean_pool(lhs, inputs["attention_mask"].astype(np.float32)).astype(np.float32)
            emb = self._l2norm(emb)
            vecs.append(emb)
        return np.vstack(vecs).astype(np.float32)

_HAS_ONNX = True
ENCODER: Optional[SimpleOnnxEncoder] = None

# ===== Utils =====
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def ensure_feedback_csv():
    os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
    if not os.path.exists(FEEDBACK_CSV):
        with open(FEEDBACK_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS).writeheader()

def append_log_row(row: dict) -> str:
    ensure_feedback_csv()
    log_id = str(uuid.uuid4())
    base = {k: "" for k in FEEDBACK_FIELDS}
    base.update(row or {})
    base["log_id"] = log_id
    base["ts_iso"] = _utcnow_iso()
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS).writerow(base)
    return log_id

def update_feedback_label(log_id: str, label: int) -> bool:
    if not os.path.exists(FEEDBACK_CSV):
        return False
    df = pd.read_csv(FEEDBACK_CSV, encoding="utf-8-sig")
    if "log_id" not in df.columns:
        return False
    mask = (df["log_id"] == log_id)
    if not mask.any():
        return False
    df.loc[mask, "label"] = int(label)
    df.loc[mask, "feedback_ts_iso"] = _utcnow_iso()
    df.to_csv(FEEDBACK_CSV, index=False, encoding="utf-8-sig")
    return True

def read_csv_smart(path: str) -> pd.DataFrame:
    encodings_try = []
    if CSV_ENCODING_ENV:
        encodings_try.append(CSV_ENCODING_ENV)
    encodings_try.extend(["utf-8", "utf-8-sig", "cp874", "tis-620"])

    last_err = None
    for enc in encodings_try:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[csv] read QA.csv with encoding='{enc}' -> shape={df.shape}")
            return df
        except Exception as e:
            last_err = e
            print(f"[csv:warn] encoding='{enc}' failed: {e}")
    raise RuntimeError(f"cannot read CSV {path}: {last_err}")

def load_pickle_matrix(path: str) -> np.ndarray:
    obj = pd.read_pickle(path)

    if isinstance(obj, np.ndarray):
        arr = obj
    elif isinstance(obj, (list, tuple)):
        arr = np.asarray(obj)
    elif isinstance(obj, dict):
        for key in ["embeds", "X", "vectors", "embeddings", "array", "arr", "data"]:
            if key in obj:
                arr = np.asarray(obj[key])
                break
        else:
            vals = list(obj.values())
            if len(vals) > 0 and hasattr(vals[0], "__len__"):
                arr = np.asarray(vals)
            else:
                raise ValueError(f"{path} dict ไม่รู้จักรูปแบบ (ไม่มี keys มาตรฐาน)")
    elif isinstance(obj, pd.DataFrame):
        cand = [c for c in obj.columns if str(c).lower() in ("embedding","emb","vector","vec","embeddings","vectors")]
        if not cand:
            for c in obj.columns:
                v0 = obj[c].iloc[0]
                if isinstance(v0, (list, tuple, np.ndarray)):
                    cand = [c]; break
        if cand:
            arr = np.vstack(obj[cand[0]].apply(lambda x: np.asarray(x)).values)
        else:
            raise ValueError(f"{path} DataFrame ไม่พบคอลัมน์เวกเตอร์")
    else:
        raise ValueError(f"{path} ไม่รองรับชนิด {type(obj)}")

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{path} ควรเป็นเมทริกซ์ 2 มิติ (N,D) — shape={arr.shape}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr

def l2_normalize(mat: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    denom = np.sqrt((mat * mat).sum(axis=axis, keepdims=True)) + eps
    return mat / denom

def build_docs_from_df(df: pd.DataFrame) -> List[str]:
    col_q, col_a = None, None
    for c in df.columns:
        if str(c).strip() in ("คำถาม", "คำถาม ", "question", "Question"):
            col_q = c
        if str(c).strip() in ("คำตอบ", "answer", "Answer"):
            col_a = c
    texts = []
    for _, row in df.iterrows():
        q = str(row.get(col_q, "")).strip()
        a = str(row.get(col_a, "")).strip()
        texts.append((q + " " + a).strip())
    return texts

def build_meta_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    mod_col, q_col, a_col = None, None, None
    for c in df.columns:
        s = str(c).strip()
        if s in ("Module", "module"):
            mod_col = c
        if s in ("คำถาม", "คำถาม ", "question", "Question"):
            q_col = c
        if s in ("คำตอบ", "answer", "Answer"):
            a_col = c
    out = []
    for _, row in df.iterrows():
        out.append({
            "module": str(row.get(mod_col, "")).strip(),
            "question": str(row.get(q_col, "")).strip(),
            "answer": str(row.get(a_col, "")).strip(),
        })
    return out

def build_tfidf(texts: List[str]):
    vec = TfidfVectorizer(tokenizer=tokenize_unicode, analyzer="word", lowercase=True, min_df=1)
    mat = vec.fit_transform(texts)
    return vec, mat

# ===== App & routes =====
app = FastAPI(title="Retriever API (ONNX + Hybrid + Guardrails)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

# ===== Payloads =====
class SearchPayload(BaseModel):
    q: str
    topk: int = 10
    hybrid: bool = True
    alpha: float = 0.3
    min_score: float = 0.0
    use_reranker: bool = False
    rerank_k: int = 10
    session_id: Optional[str] = Field(default=None)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    messages: List[ChatMessage]
    topk: int = 10
    hybrid: bool = True
    alpha: float = 0.3
    min_score: float = 0.0
    use_reranker: bool = False
    rerank_k: int = 10
    session_id: Optional[str] = Field(default=None)

class FeedbackPayload(BaseModel):
    log_id: str
    label: int  # 1 = like, -1 = dislike

# ===== Guardrails =====
def apply_guardrails(q: str, _cand_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    info = {"triggered": False, "reasons": [], "message": ""}

    if not USE_GUARD:
        return info

    q_norm = (q or "").strip()
    if not q_norm:
        info.update(triggered=True, reasons=["empty"], message="กรุณาพิมพ์ข้อความค้นหา")
        return info

    tokens = tokenize_unicode(q_norm)
    len_tokens = len(tokens)
    len_chars = len(q_norm)
    is_thai = has_thai(q_norm)

    if is_thai and len_tokens == 0 and len_chars >= 1:
        tokens = [q_norm]
        len_tokens = 1

    too_short = False
    if is_thai:
        if len_chars < max(1, GR_MIN_LEN_CHARS):
            too_short = True
    else:
        if len_tokens < max(1, GR_MIN_LEN):
            too_short = True

    if too_short:
        info.update(triggered=True, reasons=["too_short"], message="คำค้นหาสั้นเกินไป—พิมพ์รายละเอียดเพิ่มอีกนิด")

    print(f"[guardrails] q={repr(q_norm)} is_thai={is_thai} len_chars={len_chars} len_tokens={len_tokens} tokens={tokens[:8]}")
    return info

# ===== Search core =====
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(a, b)

def search_core(q: str, topk: int, alpha: float, hybrid: bool, min_score: float):
    global EMB, VECTORIZER, TFIDF_MAT

    alpha_eff = alpha
    if ENCODER is None:
        alpha_eff = 0.0  # TF-IDF only

    # 1) Encode with ONNX (if available)
    t0 = time.time()
    if ENCODER is not None:
        q_text = f"query: {q}"
        q_vec = ENCODER.encode_queries([q_text], max_len=256, batch_size=32)
        q_vec = np.asarray(q_vec)[0].astype(np.float32)
        q_emb = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    else:
        q_emb = np.zeros((DIM,), dtype=np.float32)
    ms_encode = int((time.time() - t0) * 1000)

    # 2) Cosine vs corpus
    cos = cosine_sim(EMB, q_emb)  # (N,)

    # 3) TF-IDF score
    if hybrid and VECTORIZER is not None and TFIDF_MAT is not None:
        q_kw = VECTORIZER.transform([q])  # (1,V)
        kw = (TFIDF_MAT @ q_kw.T).toarray().ravel()
    else:
        kw = np.zeros_like(cos)

    # 4) Blend
    score = alpha_eff * cos + (1 - alpha_eff) * kw

    # 5) Sort & pick
    idx = np.argsort(-score)
    items = []
    for rank, i in enumerate(idx[:max(1, topk)], start=1):
        sc = float(score[i])
        if sc < min_score:
            continue
        items.append({
            "rank": rank,
            "score": sc,
            "cos": float(cos[i]),
            "kw": float(kw[i]),
            "module": DOC_META[i]["module"],
            "question": DOC_META[i]["question"],
            "answer": DOC_META[i]["answer"],
        })

    return items, ms_encode

def split_best_alternatives(items: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if not items:
        return None, []
    return items[0], items[1:]

def build_search_response(
    q: str, topk: int, alpha: float, hybrid: bool, min_score: float,
    items: List[Dict[str, Any]], ms_encode: int, guardrail: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    best, alternatives = split_best_alternatives(items)
    rows = [
        [
            it["rank"], round(it["score"], 3), round(it["cos"], 3), round(it["kw"], 3),
            it["module"], it["question"], it["answer"],
        ] for it in items
    ]
    return {
        "ok": True,
        "version": APP_VERSION,
        "q": q,
        "topk": topk,
        "n_rows": N_ROWS, "N": N_ROWS, "dim": DIM,
        "alpha": alpha, "hybrid": hybrid, "min_score": min_score,
        "use_reranker": False, "rerank_k": 0,
        "ms_encode": ms_encode, "ms_rank": 0,
        "rows": rows, "items": items,
        "best": best, "alternatives": alternatives,
        "vocab": TFIDF_VOCAB,
        "guardrail": guardrail or {"triggered": False, "reasons": [], "message": ""},
    }

# ===== Routes =====
@app.get("/health")
async def health():
    return {
        "ok": True,
        "version": APP_VERSION,
        "n_rows": N_ROWS, "N": N_ROWS, "dim": DIM,
        "vocab": TFIDF_VOCAB,
        "onnx_dir": ONNX_DIR,
        "csv_path": CSV_PATH,
        "index_pkl": CORPUS_PKL,
        "feedback_csv": FEEDBACK_CSV,
        "encoder_available": True,
        "encoder_loaded": bool(ENCODER is not None),
    }

@app.post("/search")
async def search(payload: SearchPayload, request: Request):
    q = payload.q or ""
    topk = int(payload.topk or 10)
    alpha = float(payload.alpha or 0.9)
    hybrid = bool(payload.hybrid)
    min_score = float(payload.min_score or 0.0)

    guard = apply_guardrails(q, [])
    if guard.get("triggered"):
        resp = build_search_response(q, topk, alpha, hybrid, min_score, [], 0, guard)
        resp["log_id"] = ""
        return JSONResponse(resp)

    items, ms_encode = search_core(q, topk, alpha, hybrid, min_score)
    resp = build_search_response(q, topk, alpha, hybrid, min_score, items, ms_encode)

    best = resp.get("best") or {}
    alts = resp.get("alternatives") or []
    alternatives_json = json.dumps(
        [{"score": round(float(x.get("score", 0.0)), 6),
          "module": x.get("module", ""),
          "question": x.get("question", "")}
         for x in alts],
        ensure_ascii=False
    )
    log_id = append_log_row({
        "session_id": payload.session_id or "",
        "route": "search",
        "q": q,
        "best_module": best.get("module", ""),
        "best_question": best.get("question", ""),
        "best_answer": best.get("answer", ""),
        "best_score": round(float(best.get("score", 0.0)), 6) if best else 0.0,
        "alpha": alpha, "hybrid": hybrid, "min_score": min_score, "topk": topk,
        "n_rows": N_ROWS, "dim": DIM, "onnx_dir": ONNX_DIR,
        "latency_ms": resp.get("ms_encode", 0),
        "label": "", "feedback_ts_iso": "", "alternatives_json": alternatives_json,
    })
    resp["log_id"] = log_id
    return JSONResponse(resp)

@app.post("/chat")
async def chat(payload: ChatPayload, request: Request):
    msgs = payload.messages or []
    topk = int(payload.topk or 10)
    alpha = float(payload.alpha or 0.9)
    hybrid = bool(payload.hybrid)
    min_score = float(payload.min_score or 0.0)

    q = ""
    for m in reversed(msgs):
        if (m.role or "").lower() == "user":
            q = m.content or ""
            break

    guard = apply_guardrails(q, [])
    if guard.get("triggered"):
        resp = build_search_response(q, topk, alpha, hybrid, min_score, [], 0, guard)
        resp["messages_echo"] = [{"role": m.role, "content": m.content} for m in msgs][-10:]
        resp["log_id"] = ""
        return JSONResponse(resp)

    items, ms_encode = search_core(q, topk, alpha, hybrid, min_score)
    resp = build_search_response(q, topk, alpha, hybrid, min_score, items, ms_encode)
    resp["messages_echo"] = [{"role": m.role, "content": m.content} for m in msgs][-10:]

    best = resp.get("best") or {}
    alts = resp.get("alternatives") or []
    alternatives_json = json.dumps(
        [{"score": round(float(x.get("score", 0.0)), 6),
          "module": x.get("module", ""),
          "question": x.get("question", "")}
         for x in alts],
        ensure_ascii=False
    )
    log_id = append_log_row({
        "session_id": payload.session_id or "",
        "route": "chat",
        "q": q,
        "best_module": best.get("module", ""),
        "best_question": best.get("question", ""),
        "best_answer": best.get("answer", ""),
        "best_score": round(float(best.get("score", 0.0)), 6) if best else 0.0,
        "alpha": alpha, "hybrid": hybrid, "min_score": min_score, "topk": topk,
        "n_rows": N_ROWS, "dim": DIM, "onnx_dir": ONNX_DIR,
        "latency_ms": resp.get("ms_encode", 0),
        "label": "", "feedback_ts_iso": "", "alternatives_json": alternatives_json,
    })
    resp["log_id"] = log_id
    return JSONResponse(resp)

@app.post("/feedback")
async def submit_feedback(payload: FeedbackPayload):
    ok = update_feedback_label(payload.log_id, payload.label)
    return {"ok": bool(ok), "log_id": payload.log_id}

# ===== Startup =====
def load_everything():
    global DF, DOCS, DOC_META, EMB, DIM, N_ROWS, VECTORIZER, TFIDF_MAT, TFIDF_VOCAB, ENCODER

    DF = read_csv_smart(CSV_PATH)
    DOCS[:] = build_docs_from_df(DF)
    DOC_META[:] = build_meta_from_df(DF)

    arr = load_pickle_matrix(CORPUS_PKL)
    N_ROWS, DIM = arr.shape
    EMB = l2_normalize(arr.astype(np.float32), axis=1)
    print(f"[index] loaded corpus_embeds.pkl shape={arr.shape}, dtype={arr.dtype}")

    VECTORIZER, TFIDF_MAT = build_tfidf(DOCS)
    TFIDF_VOCAB = len(VECTORIZER.vocabulary_)
    print(f"[tfidf] built vocab={TFIDF_VOCAB} shape={TFIDF_MAT.shape}")

    # init ONNX encoder
    try:
        ENCODER = SimpleOnnxEncoder(ONNX_DIR)
        print(f"[onnx] encoder initialized from: {ONNX_DIR}")
    except Exception as e:
        ENCODER = None
        print(f"[onnx:warn] cannot init encoder from {ONNX_DIR}: {e}")

    print(f"[startup] N={N_ROWS} dim={DIM} onnx='{ONNX_DIR}' index='{CORPUS_PKL}' csv='{CSV_PATH}'")

@app.on_event("startup")
def _on_startup():
    ensure_feedback_csv()
    load_everything()
    print("== ENV/CONFIG ==")
    print(f"APP_VERSION={APP_VERSION}")
    print(f"GR_USE={int(USE_GUARD)}")
    print(f"GR_MIN_TOP1={GR_MIN_TOP1}")
    print(f"GR_MIN_SCORE={GR_MIN_SCORE}")
    print(f"GR_MIN_KW={GR_MIN_KW}")
    print(f"GR_MIN_OVERLAP={GR_MIN_OVERLAP}")
    print(f"GR_MIN_LEN={GR_MIN_LEN}")
    print(f"GR_MIN_LEN_CHARS={GR_MIN_LEN_CHARS}")
    print(f"GR_MIN_MARGIN={GR_MIN_MARGIN}")
    print(f"CSV_ENCODING={CSV_ENCODING_ENV or ''}")
    print(f"CSV_PATH={CSV_PATH}")
    print(f"INDEX_PKL={CORPUS_PKL}")
    print(f"ONNX_DIR={ONNX_DIR}")
    print(f"FEEDBACK_CSV={FEEDBACK_CSV}")
    print("==========")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
