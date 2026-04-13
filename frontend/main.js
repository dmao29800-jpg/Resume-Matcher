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
    animateCount(scoreNumEl, 0, score, 1200);
    scoreBarFill.style.width = score + "%";
    scoreLabelEl.textContent = labelFor(score);
    scoreFillEl.style.stroke = colorFor(score);
    scoreBarFill.style.background = colorFor(score);
  });

  // Feedback list — STAR 卡片
  feedbackEl.innerHTML = "";
  const list = Array.isArray(suggestions) ? suggestions : [];
  if (list.length === 0) {
    const li = document.createElement("li");
    li.textContent = "简历与 JD 整体匹配度良好，针对上述细节可进一步优化。";
    feedbackEl.appendChild(li);
    return;
  }

  list.forEach((item, i) => {
    const card = document.createElement("div");
    card.className = "star-card" + (i === 0 ? " open" : ""); // 默认展开第一条

    const tagClass = {
      "缺失":     "tag-missing",
      "关键词不足": "tag-keyword",
      "年限不足":  "tag-years",
      "语言匹配":  "tag-language",
      "优秀":      "tag-great",
    }[item.tag] || "tag-missing";

    const star = item.star || {};

    card.innerHTML = `
      <div class="star-card__head">
        <span class="star-card__clause">${escHtml(item.clause || item)}</span>
        <div class="star-card__meta">
          ${item.tag ? `<span class="star-card__tag ${tagClass}">${escHtml(item.tag)}</span>` : ""}
          <svg class="star-card__chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </div>
      </div>
      <div class="star-card__body">
        <div class="star-steps">
          ${star.S ? `<div class="star-step"><span class="star-step__letter S">S</span><p class="star-step__text"><strong>Situation 背景：</strong>${escHtml(star.S)}</p></div>` : ""}
          ${star.T ? `<div class="star-step"><span class="star-step__letter T">T</span><p class="star-step__text"><strong>Task 任务：</strong>${escHtml(star.T)}</p></div>` : ""}
          ${star.A ? `<div class="star-step"><span class="star-step__letter A">A</span><p class="star-step__text"><strong>Action 行动：</strong>${escHtml(star.A)}</p></div>` : ""}
          ${star.R ? `<div class="star-step"><span class="star-step__letter R">R</span><p class="star-step__text"><strong>Result 结果：</strong>${escHtml(star.R)}</p></div>` : ""}
        </div>
      </div>`;

    // 点击折叠/展开
    card.querySelector(".star-card__head").addEventListener("click", () => {
      card.classList.toggle("open");
    });

    feedbackEl.appendChild(card);
  });
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
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
