import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from matcher import match_score
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 允许的上传类型
ALLOWED_EXT = {".pdf", ".docx", ".txt"}

def allowed_file(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in ALLOWED_EXT

def read_file(file_path: str) -> str:
    """根据文件后缀读取文本内容（简化实现）。"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".pdf":
        from pdfminer.high_level import extract_text
        return extract_text(file_path)
    else:
        return ""

@app.route("/match", methods=["POST"])
def match():
    # 1️⃣ 简历文件
    if "resume" not in request.files:
        return jsonify({"error": "missing resume file"}), 400
    resume_file = request.files["resume"]
    if resume_file.filename == "" or not allowed_file(resume_file.filename):
        return jsonify({"error": "invalid resume file"}), 400

    # 2️⃣ JD 文本（直接表单字段）
    jd_text = request.form.get("jd", "").strip()
    if not jd_text:
        return jsonify({"error": "missing JD text"}), 400

    # 保存临时文件（使用系统临时目录，避免跨平台编码问题）
    tmp_dir = os.getenv("TEMP") or "/tmp"
    resume_path = os.path.join(tmp_dir, secure_filename(resume_file.filename))
    resume_file.save(resume_path)

    # 读取简历内容
    resume_text = read_file(resume_path)

    # 计算匹配度 + 建议
    score, suggestions = match_score(resume_text, jd_text)

    # 清理临时文件
    try:
        os.remove(resume_path)
    except OSError:
        pass

    return jsonify({
        "score": score,            # 0~100 的整数/小数
        "suggestions": suggestions  # List[Dict] STAR 格式
    })

# ==== 为调试提供简单主页 ====
@app.route("/", methods=["GET"])
def index():
    return "Resume‑Matcher API is running."

if __name__ == "__main__":
    # 0.0.0.0 让容器平台可对外访问
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))