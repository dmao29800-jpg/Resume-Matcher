/* ── API ──────────────────────────────────────────── */
const API_BASE = "http://127.0.0.1:5000/";

/* ── DOM refs ─────────────────────────────────────── */
const resumeInput   = document.getElementById("resume");
const fakeBrowse   = document.getElementById("fakeBrowse");
const dropZone     = document.getElementById("dropZone");
const dropInner    = document.getElementById("dropInner");
const fileNameEl   = document.getElementById("fileName");
const jdTextarea   = document.getElementById("jd");
const jdLenEl      = document.getElementById("jdLen");
const analyzeBtn   = document.getElementById("analyzeBtn");
const loadingEl    = document.getElementById("loading");
const resultEl     = document.getElementById("result");
const scoreNumEl   = document.getElementById("scoreNum");
const scoreFillEl  = document.getElementById("scoreFill");
const scoreBarFill = document.getElementById("scoreBarFill");
const scoreLabelEl = document.getElementById("scoreLabel");
const feedbackEl   = document.getElementById("feedback");
const resetBtn     = document.getElementById("resetBtn");

/* ── File selection ───────────────────────────────── */
fakeBrowse.addEventListener("click", () => resumeInput.click());

resumeInput.addEventListener("change", () => {
  const f = resumeInput.files[0];
  fileNameEl.textContent = f ? f.name : "尚未选择文件";
});

/* ── Drag & drop ──────────────────────────────────── */
["dragenter", "dragover"].forEach(ev =>
  dropZone.addEventListener(ev, e => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach(ev =>
  dropZone.addEventListener(ev, e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
  })
);
dropZone.addEventListener("drop", e => {
  const f = e.dataTransfer.files[0];
  if (f) {
    const dt = new DataTransfer();
    dt.items.add(f);
    resumeInput.files = dt.files;
    fileNameEl.textContent = f.name;
  }
});

/* ── JD char counter ──────────────────────────────── */
jdTextarea.addEventListener("input", () => {
  jdLenEl.textContent = jdTextarea.value.length;
});

/* ── Reset ────────────────────────────────────────── */
resetBtn.addEventListener("click", () => {
  resultEl.hidden = true;
  scoreFillEl.style.strokeDashoffset = 314.16;
  scoreBarFill.style.width = "0%";
  resumeInput.value = "";
  jdTextarea.value = "";
  jdLenEl.textContent = "0";
  fileNameEl.textContent = "尚未选择文件";
  feedbackEl.innerHTML = "";
});

/* ── Main flow ────────────────────────────────────── */
analyzeBtn.addEventListener("click", async () => {
  const resumeFile = resumeInput.files[0];
  const jdText    = jdTextarea.value.trim();

  if (!resumeFile) { alert("请先选择简历文件"); return; }
  if (!jdText)      { alert("请粘贴岗位描述（JD）"); return; }

  analyzeBtn.hidden = true;
  resultEl.hidden   = true;
  loadingEl.hidden  = false;

  const form = new FormData();
  form.append("resume", resumeFile);
  form.append("jd", jdText);

  try {
    const resp = await fetch(API_BASE + "match", { method: "POST", body: form });
    const data = await resp.json();

    loadingEl.hidden = true;
    showResult(data.score, data.suggestions);
  } catch (err) {
    loadingEl.hidden = true;
    analyzeBtn.hidden = false;
    console.error("详细错误信息:", err);
    alert("请求出错，请检查后端是否已启动（python app.py）");
  }
});

/* ── Render result ────────────────────────────────── */
function showResult(score, suggestions) {
  resultEl.hidden = false;
  analyzeBtn.hidden = false;

  // Ring circumference = 2π × 50 = 314.16
  const CIRC = 314.16;
  const offset = CIRC - (score / 100) * CIRC;

  // Animate ring
  requestAnimationFrame(() => {
    scoreFillEl.style.strokeDashoffset = offset;
    // Animate number
    animateCount(scoreNumEl, 0, score, 1200);
    // Bar fill
    scoreBarFill.style.width = score + "%";
    // Label
    scoreLabelEl.textContent = labelFor(score);
    // Color accent by score
    scoreFillEl.style.stroke = colorFor(score);
    scoreBarFill.style.background = colorFor(score);
  });

  // Feedback list
  feedbackEl.innerHTML = "";
  if (suggestions && suggestions.length) {
    suggestions.forEach(s => {
      const li = document.createElement("li");
      li.textContent = s;
      feedbackEl.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.textContent = "简历与 JD 整体匹配度良好，针对上述细节可进一步优化。";
    feedbackEl.appendChild(li);
  }
}

function labelFor(s) {
  if (s >= 80) return "匹配度 · 优秀";
  if (s >= 60) return "匹配度 · 良好";
  if (s >= 40) return "匹配度 · 一般";
  return "匹配度 · 偏低";
}

function colorFor(s) {
  if (s >= 70) return "#e07a5f";    // 暖橙
  if (s >= 45) return "#f4a261";    // 中橙
  return "#83c5be";                  // 薄荷绿
}

function animateCount(el, from, to, duration) {
  const start = performance.now();
  function tick(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3); // ease-out-cubic
    el.textContent = Math.round(from + (to - from) * ease);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
