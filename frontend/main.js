/* ==== 配置 ==== */
// 把这里换成你部署好的后端地址（不要忘记末尾的斜杠）
const API_BASE = "http://127.0.0.1:5000/";
// ==== 工具函数 ====
function showResult(score, suggestions) {
  document.getElementById('score').textContent = score.toFixed(1);
  const fbDiv = document.getElementById('feedback');
  fbDiv.innerHTML = "";
  if (suggestions.length) {
    const ul = document.createElement('ul');
    suggestions.forEach(s => {
      const li = document.createElement('li');
      li.textContent = s;
      ul.appendChild(li);
    });
    fbDiv.appendChild(ul);
  } else {
    fbDiv.textContent = "简历基本匹配，无需重大修改。";
  }
  document.getElementById('result').hidden = false;
}

// ==== 主流程 ====
document.getElementById('analyzeBtn').addEventListener('click', async () => {
  const resumeFile = document.getElementById('resume').files[0];
  const jdText = document.getElementById('jd').value.trim();

  if (!resumeFile) { alert('请先选择简历文件'); return; }
  if (!jdText)   { alert('请粘贴岗位要求（JD）'); return; }

  const form = new FormData();
  form.append('resume', resumeFile);
  form.append('jd', jdText);

 // 在 main.js 中修改
try {
    const resp = await fetch(API_BASE + "match", {
      method: "POST",
      body: form
    });
    const data = await resp.json();   
    console.log("服务器返回的数据:", data); // 增加这一行，按 F12 就能看到具体内容
    showResult(data.score, data.suggestions);
} catch (e) {
    console.error("详细错误信息:", e); // 这里能看到到底哪里报错了
    alert('请求出错，请检查控制台 Console 的详细报错');
}
});