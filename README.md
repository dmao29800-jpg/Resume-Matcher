# Resume Matcher - 简历 JD 匹配分析系统

一个智能简历与职位描述（JD）匹配分析工具，基于 TF-IDF 和语义相似度算法，生成 STAR 法则改进建议。

## 功能特性

- 📊 **智能评分** — 多维度评估简历与 JD 匹配度
- 🎯 **STAR 建议** — 按情境-任务-行动-结果生成结构化改进建议
- 🎨 **精美 UI** — 巴恩 + INS 风格，暖陶土橙主色调
- 📱 **响应式设计** — 支持桌面和移动端

## 技术栈

### 前端
- 纯 HTML/CSS/JavaScript
- 无框架依赖，轻量快速
- 渐进式披露交互

### 后端
- Python Flask
- TF-IDF + 语义相似度算法
- 支持 PDF/Word 简历解析

## 本地开发

### 前置要求
- Python 3.12+
- Node.js 18+（可选，用于本地服务器）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/dmao29800-jpg/Resume-Matcher.git
cd Resume-Matcher

# 安装后端依赖
cd backend
pip install -r requirements.txt

# 启动后端服务
python app.py

# 在另一个终端打开前端
cd ../frontend
# 方式1: 直接用浏览器打开 index.html
# 方式2: 使用本地服务器
npx serve .
```

## 部署

### Railway（后端）

1. 访问 [Railway](https://railway.app/)
2. 使用 GitHub 登录
3. 创建新项目 → 从 GitHub 导入 `Resume-Matcher`
4. 设置 Root Directory 为 `backend`
5. 自动部署

### Vercel（前端）

1. 访问 [Vercel](https://vercel.com/)
2. 使用 GitHub 登录
3. 导入 `Resume-Matcher` 仓库
4. 设置 Root Directory 为 `frontend`
5. 添加环境变量 `BACKEND_URL` 指向 Railway 后端地址

## 作者

初学者制作，使用中遇到的问题欢迎联系作者进行反馈！

📧 邮箱：1903300602@qq.com

## 许可证

MIT License
