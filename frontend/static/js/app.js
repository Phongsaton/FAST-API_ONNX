// ===== Helpers =====
const $ = (s) => document.querySelector(s);
const api = (path) => location.origin + path;

function escapeHtml(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function nl2br(s){ return String(s).replace(/\n/g,"<br>"); }
function setStatus(t){ const el = $("#status"); if (el) el.textContent = t || ""; }
function scrollToBottom(){ const box = $("#chatLog"); if (box) box.scrollTop = box.scrollHeight; }

// ===== Feedback =====
async function postFeedback(logId, label){
  const res = await fetch(api("/feedback"), {
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify({ log_id: logId, label })
  });
  return res.json();
}
function makeFeedbackBar(logId){
  const bar = document.createElement("div");
  bar.className = "feedback-bar";
  const like = document.createElement("button");
  like.className = "feedback-btn"; like.textContent = "👍";
  const dislike = document.createElement("button");
  dislike.className = "feedback-btn"; dislike.textContent = "👎";
  const st = document.createElement("span");
  st.className = "feedback-status";

  like.addEventListener("click", async () => {
    like.disabled = true; dislike.disabled = true;
    try{ const r = await postFeedback(logId, 1); st.textContent = r.ok ? "บันทึก 👍 แล้ว" : "บันทึกไม่สำเร็จ"; }
    catch{ st.textContent = "ส่งฟีดแบ็กไม่สำเร็จ"; }
  });
  dislike.addEventListener("click", async () => {
    like.disabled = true; dislike.disabled = true;
    try{ const r = await postFeedback(logId, -1); st.textContent = r.ok ? "บันทึก 👎 แล้ว" : "บันทึกไม่สำเร็จ"; }
    catch{ st.textContent = "ส่งฟีดแบ็กไม่สำเร็จ"; }
  });

  bar.append(like, dislike, st);
  return bar;
}

// ===== Chat rendering =====
function appendUser(text){
  const row = document.createElement("div");
  row.className = "msg user";
  row.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  $("#chatLog").appendChild(row);
  scrollToBottom();
}
function appendBot(answer, meta, logId){
  const row = document.createElement("div");
  row.className = "msg bot";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = nl2br(escapeHtml(answer || "(ไม่พบคำตอบ)"));

  // แสดงปุ่มเฉพาะในบับเบิลของบอท เมื่อมี logId จาก backend
  if (logId) bubble.appendChild(makeFeedbackBar(logId));

  row.appendChild(bubble);
  if (meta){
    const metaDiv = document.createElement("div");
    metaDiv.className = "meta";
    metaDiv.textContent = `score: ${meta.score} · cos: ${meta.cos} · kw: ${meta.kw}${meta.module ? " · " + meta.module : ""}`;
    row.appendChild(metaDiv);
  }
  $("#chatLog").appendChild(row);
  scrollToBottom();
}

// ===== Alternatives (2..K) =====
function renderAlternatives(items = []){
  const ul = $("#altList");
  ul.innerHTML = "";
  if (!items.length){
    ul.innerHTML = `<li class="alt-item">ไม่มีตัวเลือกอื่น</li>`;
    return;
  }
  for (const r of items){
    const li = document.createElement("li");
    li.className = "alt-item";
    const score = (r.score ?? 0).toFixed(3);
    const cos   = (r.cos ?? 0).toFixed(3);
    const kw    = (r.kw  ?? 0).toFixed(3);
    li.innerHTML = `
      <div class="alt-title">
        <strong>${escapeHtml(r.title || r.question || "(ไม่มีหัวข้อ)")}</strong>
        <span class="alt-score">score: ${score}</span>
      </div>
      <div class="alt-body">${nl2br(escapeHtml(r.answer || r.text || ""))}</div>
      ${r.module ? `<div class="alt-meta">${escapeHtml(r.module)}</div>` : ""}
    `;
    // คลิกเพื่อส่งเข้าแชท (ไม่เรียก API ใหม่)
    li.addEventListener("click", () => {
      appendBot(r.answer || r.text || "", { score, cos, kw, module: r.module || "" }, "");
    });
    ul.appendChild(li);
  }
}

// ===== API calls =====
async function fetchHealth(){
  try{
    const res = await fetch(api("/health"), { cache:"no-store" });
    const h = await res.json();
    setStatus(`พร้อม: N=${h.N ?? h.n_rows} · dim=${h.dim} · vocab=${h.vocab}`);
  }catch{
    setStatus("เชื่อมต่อเซิร์ฟเวอร์ไม่ได้");
  }
}

let inFlight = false;
async function doSearch(q){
  if (!q || q.trim().length < 1){ setStatus("พิมพ์คำค้นหาก่อน"); return; }
  if (inFlight) return;
  inFlight = true; setStatus("กำลังค้นหา...");

  appendUser(q);

  const alpha = Number($("#alpha")?.value || 0.30);
  const payload = {
    q, topk: 10, hybrid: true,
    alpha, min_score: 0.0,
    use_reranker: false, rerank_k: 0
  };

  try{
    const res = await fetch(api("/search"), {
      method:"POST",
      headers:{ "Content-Type":"application/json; charset=utf-8" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("HTTP "+res.status);
    const data = await res.json();

    if (data.guardrail?.triggered){
      appendBot(data.guardrail.message || "โปรดพิมพ์รายละเอียดเพิ่มเติม", null, "");
      renderAlternatives([]);
      setStatus("เสร็จ");
      return;
    }

    // รองรับทั้งโครงใหม่/เก่า
    const best = data.best ?? (Array.isArray(data.items) ? data.items[0] : null);
    const alts = data.alternatives ?? (Array.isArray(data.items) ? data.items.slice(1) : []);
    const logId = data.log_id || ""; // backend ควรส่งมาเพื่อเปิดปุ่ม

    if (best){
      const meta = {
        score: (best.score ?? 0).toFixed(3),
        cos:   (best.cos ?? 0).toFixed(3),
        kw:    (best.kw  ?? 0).toFixed(3),
        module: best.module || ""
      };
      appendBot(best.answer || best.text || "(ไม่พบคำตอบ)", meta, logId);
    }else{
      appendBot("ไม่พบคำตอบที่เหมาะสม", null, "");
    }
    renderAlternatives(alts);
    setStatus("เสร็จ");
  }catch(e){
    console.error(e);
    appendBot("ค้นหาล้มเหลว: " + (e.message || e), null, "");
    renderAlternatives([]);
    setStatus("ค้นหาล้มเหลว");
  }finally{
    inFlight = false;
  }
}

// ===== Init =====
document.addEventListener("DOMContentLoaded", () => {
  // slider ↔ badge
  const alpha = $("#alpha");
  const alphaValue = $("#alphaValue");
  if (alpha && alphaValue){
    const sync = () => alphaValue.textContent = Number(alpha.value).toFixed(2);
    alpha.addEventListener("input", sync);
    sync();
  }

  // health
  const btnHealth = $("#btn-health");
  if (btnHealth){ btnHealth.addEventListener("click", fetchHealth); }
  fetchHealth();

  // submit
  const form = $("#chatForm");
  const input = $("#q");
  if (form){
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const q = (input?.value || "").trim();
      if (!q) return;
      input.value = "";
      doSearch(q);
    });
  }
});
