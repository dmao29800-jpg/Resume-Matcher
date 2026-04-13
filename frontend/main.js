/* ── API ──────────────────────────────────────────── */
// 开发环境使用本地地址，生产环境使用环境变量或相对路径
const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://127.0.0.1:5000/"
  : (window.BACKEND_URL || "https://resume-matcher-backend.railway.app/");

/* ── Pages ─────────────────────────────────────────── */
const inputPage   = document.getElementById("inputPage");
const loadingPage = document.getElementById("loadingPage");
const resultPage  = document.getElementById("resultPage");

/* ── DOM refs ─────────────────────────────────────── */
const resumeInput   = document.getElementById("resume");
const fakeBrowse    = document.getElementById("fakeBrowse");
const dropZone      = document.getElementById("dropZone");
const fileNameEl    = document.getElementById("fileName");
const uploadSuccess = document.getElementById("uploadSuccess");
const jdTextarea    = document.getElementById("jd");
const jdLenEl       = document.getElementById("jdLen");
const analyzeBtn    = document.getElementById("analyzeBtn");
const backBtn       = document.getElementById("backBtn");
const resetBtn      = document.getElementById("resetBtn");

const scoreNumEl    = document.getElementById("scoreNum");
const scoreFillEl   = document.getElementById("scoreFill");
const scoreBarFill  = document.getElementById("scoreBarFill");
const scoreLabelEl  = document.getElementById("scoreLabel");
const tipsCountEl   = document.getElementById("tipsCount");
const feedbackEl    = document.getElementById("feedback");

const loadingSteps = [
  document.getElementById("step1"),
  document.getElementById("step2"),
  document.getElementById("step3"),
  document.getElementById("step4")
];

/* ── File selection ───────────────────────────────── */
fakeBrowse.addEventListener("click", () => resumeInput.click());

function showUploadSuccess(filename) {
  fileNameEl.textContent = filename;
  fileNameEl.style.color = "#22c55e";
  fileNameEl.style.fontWeight = "600";
  uploadSuccess.hidden = false;
  dropZone.style.borderColor = "#22c55e";
  dropZone.style.background = "#f0fdf4";
}

function clearUploadSuccess() {
  fileNameEl.textContent = "尚未选择文件";
  fileNameEl.style.color = "";
  fileNameEl.style.fontWeight = "";
  uploadSuccess.hidden = true;
  dropZone.style.borderColor = "";
  dropZone.style.background = "";
}

resumeInput.addEventListener("change", () => {
  const f = resumeInput.files[0];
  if (f) {
    showUploadSuccess(f.name);
  } else {
    clearUploadSuccess();
  }
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
    showUploadSuccess(f.name);
  }
});

/* ── JD char counter ──────────────────────────────── */
jdTextarea.addEventListener("input", () => {
  jdLenEl.textContent = jdTextarea.value.length;
});

/* ── Page Navigation ──────────────────────────────── */
function showPage(page) {
  inputPage.hidden   = page !== "input";
  loadingPage.hidden = page !== "loading";
  resultPage.hidden  = page !== "result";
}

/* ── Loading Animation ────────────────────────────── */
let loadingInterval = null;

function startLoadingAnimation() {
  let step = 0;
  loadingSteps.forEach(el => {
    el.classList.remove("active", "done");
  });
  
  loadingInterval = setInterval(() => {
    if (step > 0) {
      loadingSteps[step - 1].classList.remove("active");
      loadingSteps[step - 1].classList.add("done");
    }
    if (step < 4) {
      loadingSteps[step].classList.add("active");
      step++;
    }
  }, 600);
}

function stopLoadingAnimation() {
  if (loadingInterval) {
    clearInterval(loadingInterval);
    loadingInterval = null;
  }
  loadingSteps.forEach(el => {
    el.classList.remove("active");
    el.classList.add("done");
  });
}

/* ── Back Button ──────────────────────────────────── */
backBtn.addEventListener("click", () => {
  showPage("input");
});

/* ── Reset Button ─────────────────────────────────── */
resetBtn.addEventListener("click", () => {
  // 重置输入
  resumeInput.value = "";
  jdTextarea.value = "";
  jdLenEl.textContent = "0";
  clearUploadSuccess();
  feedbackEl.innerHTML = "";
  
  // 返回输入页
  showPage("input");
});

/* ── Main flow ────────────────────────────────────── */
analyzeBtn.addEventListener("click", async () => {
  const resumeFile = resumeInput.files[0];
  const jdText     = jdTextarea.value.trim();

  if (!resumeFile) { alert("请先选择简历文件"); return; }
  if (!jdText)      { alert("请粘贴岗位描述（JD）"); return; }

  // 显示加载页
  showPage("loading");
  startLoadingAnimation();

  const form = new FormData();
  form.append("resume", resumeFile);
  form.append("jd", jdText);

  try {
    const resp = await fetch(API_BASE + "match", { method: "POST", body: form });
    const data = await resp.json();

    stopLoadingAnimation();
    
    // 短暂延迟后显示结果
    setTimeout(() => {
      showResult(data.score, data.suggestions);
      showPage("result");
    }, 400);
    
  } catch (err) {
    stopLoadingAnimation();
    console.error("详细错误信息:", err);
    alert("请求出错，请检查后端是否已启动（python app.py）");
    showPage("input");
  }
});

/* ── Render result ────────────────────────────────── */
function showResult(score, suggestions) {
  // Ring circumference = 2π × 50 = 314.16
  const CIRC = 314.16;
  const offset = CIRC - (score / 100) * CIRC;

  // Animate ring
  requestAnimationFrame(() => {
    scoreFillEl.style.strokeDashoffset = offset;
    animateCount(scoreNumEl, 0, score, 1200);
    scoreBarFill.style.width = score + "%";
    scoreLabelEl.textContent = labelFor(score);
    
    const c = colorFor(score);
    scoreFillEl.style.stroke = c;
    scoreBarFill.style.background = c;
    scoreNumEl.style.color = c;
  });

  // 渲染建议卡片
  feedbackEl.innerHTML = "";
  const list = Array.isArray(suggestions) ? suggestions : [];
  tipsCountEl.textContent = list.length + " 条";
  
  if (list.length === 0) {
    const li = document.createElement("li");
    li.className = "star-card";
    li.innerHTML = `
      <div class="star-card__head">
        <span class="star-card__clause">✨ 简历与 JD 匹配度良好，无需特别改进</span>
      </div>
    `;
    feedbackEl.appendChild(li);
    return;
  }

  list.forEach((item, i) => {
    const card = document.createElement("div");
    card.className = "star-card";

    const tagClass = {
      "缺失":       "tag-missing",
      "关键词不足": "tag-keyword",
      "年限不足":   "tag-years",
      "语言匹配":   "tag-language",
      "优秀":       "tag-great",
      "核心技能":   "tag-core",
      "年限":       "tag-years",
      "表达优化":   "tag-language",
      "量化不足":   "tag-keyword",
      "关联技能":   "tag-language",
      "技能缺失":   "tag-missing",
      "策略建议":   "tag-years",
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
  return "匹配度 · 有提升空间";
}

function colorFor(s) {
  if (s >= 80) return "#22c55e";
  if (s >= 60) return "#f59e0b";
  return "#ef4444";
}

function animateCount(el, from, to, duration) {
  const start = performance.now();
  function tick(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (to - from) * ease);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/* ── Contact Modal ─────────────────────────────────── */
const contactFab = document.getElementById("contactFab");
const contactModal = document.getElementById("contactModal");
const contactClose = document.getElementById("contactClose");
const copyEmailBtn = document.getElementById("copyEmail");
const copiedTip = document.getElementById("copiedTip");

contactFab.addEventListener("click", () => {
  contactModal.hidden = false;
});

contactClose.addEventListener("click", () => {
  contactModal.hidden = true;
});

contactModal.addEventListener("click", (e) => {
  if (e.target === contactModal) {
    contactModal.hidden = true;
  }
});

copyEmailBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText("1903300602@qq.com");
    copiedTip.hidden = false;
    setTimeout(() => {
      copiedTip.hidden = true;
    }, 2000);
  } catch (err) {
    // 降级方案
    const textArea = document.createElement("textarea");
    textArea.value = "1903300602@qq.com";
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("copy");
    document.body.removeChild(textArea);
    copiedTip.hidden = false;
    setTimeout(() => {
      copiedTip.hidden = true;
    }, 2000);
  }
});
