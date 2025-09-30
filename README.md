## 0) โครงสร้างของโปรเจค

......\
├─ FAST-API_ONNX\         # โปรเจกต์ API + Web UI
└─ Demo_Onnx\            # โปรเจกต์ฝั่งโมเดล/ONNX และไฟล์ดัชนี

**สิ่งที่ต้องมีใน Onnx-export_script**

Onnx-export_script\
├─ onnx_path\
│   ├─ encoder.onnx
│   ├─ tokenizer.json
│   ├─ tokenizer_config.json
│   └─ special_tokens_map.json
└─ outputs\
    └─ corpus_embeds.pkl   # แนะนำให้เป็น numpy.ndarray 2D (N,D) dtype=float32

> หาก `corpus_embeds.pkl` ยังไม่ใช่เมทริกซ์ 2D ให้แปลงก่อน (ดูหัวข้อ “แก้ปัญหา” → **แปลง index ให้เป็น ndarray 2D**)

---

## 1) ความต้องการระบบ (Requirements)

**Python 3.10 – 3.12** และแพ็กเกจหลัก

fastapi>=0.111
uvicorn[standard]>=0.30
numpy>=1.26
pandas>=2.0
scikit-learn>=1.3
pyyaml>=6.0
onnxruntime>=1.17           # ถ้าใช้ GPU: ติดตั้ง onnxruntime-gpu ให้ตรง CUDA
transformers>=4.40

---

## 2) ไฟล์สำคัญใน FAST-API_ONNX

- `app.py` — โค้ด FastAPI + Simple ONNX Encoder + Hybrid Retrieval + Guardrails
- `frontend/index.html`, `frontend/static/css/style.css`, `frontend/static/js/app.js` — หน้าเว็บ/สคริปต์
- *(ไม่บังคับ)* `data/user_feedback.csv` — จะสร้างอัตโนมัติเมื่อมีการส่งฟีดแบ็ก
- *(ไม่บังคับ)* `outputs/` — เว้นไว้ได้ หากชี้ index ไปที่ Demo_Onnx

**ไม่จำเป็น** ก็อปปี้ `encoder.onnx` หรือ tokenizer มาไว้ใน Demo_FastAPI — ชี้พาธไปที่ Demo_Onnx ก็พอ

---

## 3) ตั้งค่า `api_config.yaml` (ไม่ต้องตั้งค่า ENV/PowerShell)

วางไฟล์ `api_config.yaml` ไว้รากของ Demo_FastAPI แล้วปรับพาธตามจริง:

yaml:
# api_config.yaml (ตัวอย่าง)
data:
  csv_path: ../Demo_Onnx/data/QA.csv
  encoding: cp874           # ถ้าไฟล์เป็น UTF-8 ให้ใส่ utf-8 หรือเว้นว่างให้ auto

index:
  corpus_pkl: ../Demo_Onnx/outputs/corpus_embeds.pkl   # ควรเป็น ndarray 2D (N,D)

runtime:
  onnx_dir: ../Demo_Onnx/onnx_path
  tokenizer_dir: ../Demo_Onnx/onnx_path

feedback:
  csv_path: data/user_feedback.csv

> ใน `app.py` เวอร์ชันล่าสุด: ระบบอ่านค่า **จาก YAML ก่อน** แล้วจึงเปิดโอกาสให้ ENV override (ถ้าจำเป็นในอนาคต)

---

## 4) ขั้นตอนติดตั้งและรัน

### 4.1 สร้าง virtual environment และติดตั้งแพ็กเกจ

**Windows (PowerShell/CMD):**
cd ......\Demo_FastAPI
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install fastapi uvicorn[standard] numpy pandas scikit-learn pyyaml onnxruntime transformers

> หรือวาง `requirements.txt` ด้วยแพ็กเกจชุดเดียวกัน แล้ว `pip install -r requirements.txt`

### 4.2 ตั้งค่า `api_config.yaml`
- ให้ `runtime.onnx_dir` ชี้ไปโฟลเดอร์ที่มี `encoder.onnx` + tokenizer ครบ
- ให้ `index.corpus_pkl` ชี้ไปไฟล์เมทริกซ์เวกเตอร์ 2D (N,D)

### 4.3 รันเซิร์ฟเวอร์

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
PowerShell:
เปิดเบราว์เซอร์: `http://localhost:8000/`  
เรียก `GET /health` เพื่อตรวจสถานะ

---

## 5) กระบวนการทำงาน (Pipeline)

1. โหลด CSV → สร้าง `DOCS` จาก `(คำถาม + คำตอบ)`
2. โหลด `corpus_embeds.pkl` เป็นเมทริกซ์ `EMB (N,D)` และ L2‑normalize
3. สร้าง TF‑IDF จาก `DOCS`
4. โหลด ONNX encoder:
   - ถ้า output มี `embedding` → ใช้เวกเตอร์นั้น
   - ถ้าเป็น `last_hidden_state` → ทำ **mean pooling** + **L2 normalize**
5. เมื่อค้นหา:
   - แปลง query เป็นเวกเตอร์ (ONNX) → cosine กับ `EMB`
   - แปลง query เป็น TF‑IDF vector → จับคู่กับคอร์ปัส
   - รวมคะแนน: `score = alpha * cosine + (1 - alpha) * tfidf`
6. ตอบกลับ: `best`, `alternatives`, `rows`, และบันทึก `log_id` สำหรับฟีดแบ็ก

---

## 6) เอนด์พอยต์ API

### `GET /health`
คืนสถานะระบบ (ขนาดคอร์ปัส, มิติ, พาธไฟล์, สถานะ encoder)
```json
{
  "ok": true,
  "version": "1.3.1-feedback-A",
  "N": 1234,
  "dim": 384,
  "vocab": 9876,
  "onnx_dir": "../Demo_Onnx/onnx_path",
  "csv_path": "../Demo_Onnx/data/QA.csv",
  "index_pkl": "../Demo_Onnx/outputs/corpus_embeds.pkl",
  "feedback_csv": "data/user_feedback.csv",
  "encoder_available": true,
  "encoder_loaded": true
}
```

### `POST /search`
ค้นหาด้วยข้อความ
```json
{
  "q": "วิธีจ่ายเงิน",
  "topk": 10,
  "hybrid": true,
  "alpha": 0.3,
  "min_score": 0.0,
  "session_id": "optional"
}
```

### `POST /chat`
รูปแบบแชท
```json
{
  "messages": [
    {"role": "user", "content": "วิธีจ่ายเงิน"},
    {"role": "assistant", "content": "..."}
  ],
  "topk": 10,
  "hybrid": true,
  "alpha": 0.3
}
```

### `POST /feedback`
บันทึกฟีดแบ็กจาก `log_id`
```json
{ "log_id": "uuid-....", "label": 1 }   // 1 = like, -1 = dislike
```

---

## 7) Config

- **alpha (0..1)**: ยิ่งสูง ยิ่งพึ่ง **cosine** จาก ONNX มากขึ้น
- **min_score**: กรองผลลัพธ์ที่คะแนนต่ำ
- **CSV encoding**: ตั้ง `data.encoding` เป็น `cp874` หรือ `utf-8-sig` หากจำเป็น
- **Guardrails**: ป้องกันข้อความสั้น/ว่าง/ไม่สมเหตุสมผล (ปรับเกณฑ์ผ่าน ENV ได้)

---

## 8) การแก้ปัญหาที่พบบ่อย (Troubleshooting)

### 8.1 cosine = 0.00, หรือ `/health` → `encoder_loaded: false`
- ตรวจ `runtime.onnx_dir` ให้ชี้โฟลเดอร์ที่มี `encoder.onnx` และ tokenizer ครบ
- อ่าน log ตอนสตาร์ท: ควรมี `[onnx] encoder initialized from: ...`
- ตรวจเวอร์ชันแพ็กเกจ `onnxruntime`, `transformers` ว่าติดตั้งแล้ว

### 8.2 โหลด `corpus_embeds.pkl` แล้ว error
- ต้องเป็น **numpy.ndarray 2D (N,D)** `float32`
- ถ้ายังเป็น dict/df ให้แปลงก่อน (หัวข้อถัดไป)

### 8.3 แปลง index ให้เป็น ndarray 2D (ถ้าจำเป็น)
ตัวอย่างสคริปต์ (ปรับพาธตามจริง):

python - << "PY"
import pandas as pd, numpy as np
p = r"../Demo_Onnx/outputs/corpus_embeds.pkl"
obj = pd.read_pickle(p)
arr = None
from collections.abc import Mapping
if isinstance(obj, np.ndarray):
    arr = obj
elif isinstance(obj, (list, tuple)):
    arr = np.asarray(obj)
elif isinstance(obj, Mapping):
    for k in ["embeds","X","vectors","embeddings","array","arr","data"]:
        if k in obj: arr = np.asarray(obj[k]); break
    if arr is None: arr = np.asarray(list(obj.values()))
elif isinstance(obj, pd.DataFrame):
    for c in obj.columns:
        v0 = obj[c].iloc[0]
        if isinstance(v0,(list,tuple,np.ndarray)):
            arr = np.vstack(obj[c].apply(np.asarray).values); break
if arr is None: raise SystemExit("cannot parse index object")
arr = np.asarray(arr, dtype=np.float32)
pd.to_pickle(arr, p)
print("saved:", arr.shape, arr.dtype, "->", p)
PY

### 8.4 หน้าเว็บไม่มี CSS/JS
- ตรวจว่าไฟล์อยู่ที่ `frontend/static/css/style.css` และ `frontend/static/js/app.js`
- ใน `index.html` ต้องอ้างพาธ `/static/...`

### 8.5 คะแนนรวม `score` ต่ำผิดปกติ
- เพิ่มค่า `alpha` (เช่น 0.7–0.9) เพื่อใช้ cosine มากขึ้น
- ตรวจว่า index และ CSV มาจากชุดข้อมูลเดียวกัน
- ตรวจคำถามที่ใส่ (ภาษา/รูปแบบ) ให้สัมพันธ์กับข้อมูล

---
