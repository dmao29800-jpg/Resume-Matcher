"""
Resume–JD 匹配算法 v2
评分体系：
  - 技能关键词匹配（TF-IDF 加权）× 40%
  - JD 核心要求覆盖率 × 30%
  - 经验年限匹配 × 15%
  - 语义相似度（关键词共现）× 15%
最终得分归一化到 0~100。
"""

import re
import math
from collections import Counter
from typing import List, Dict, Tuple

# ── 停用词 ──────────────────────────────────────────────
_CN_STOP = {
    "的", "了", "在", "是", "我", "你", "他", "她", "它", "和", "与", "或",
    "但", "而", "于", "有", "这", "那", "个", "们", "为", "上", "下", "中",
    "对", "以", "到", "把", "被", "要", "会", "能", "可", "也", "很", "都",
    "将", "又", "再", "从", "所", "已", "应", "该", "等", "并", "或",
}
_EN_STOP = {
    "for", "with", "and", "or", "the", "a", "an", "to", "of", "in", "on",
    "by", "as", "at", "from", "or", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "can", "this",
    "that", "these", "those", "it", "its",
}

# ── 工具函数 ──────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """中英文混合分词，去除停用词。"""
    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower())
    return [t for t in tokens if t not in _CN_STOP and t not in _EN_STOP and len(t) > 1]


def _extract_counter(text: str, top_n: int = 100) -> Counter:
    return Counter(_tokenize(text)).most_common(top_n)


# ── TF-IDF（简化实现，不依赖外部模型）────────────────────

def _tf(tokens: List[str]) -> Dict[str, float]:
    """词频（归一化）。"""
    if not tokens:
        return {}
    freq = Counter(tokens)
    total = len(tokens)
    return {w: v / total for w, v in freq.items()}


def _idf(corpus: List[List[str]]) -> Dict[str, float]:
    """
    逆文档频率。
    corpus: 多个文档的 token 列表。
    """
    N = len(corpus)
    df = Counter()
    for doc in corpus:
        for w in set(doc):
            df[w] += 1
    return {w: math.log(N / (df[w] + 1)) + 1 for w in df}


def _tfidf(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
    """TF × IDF。"""
    return {w: tf[w] * idf.get(w, 1) for w in tf}


# ── 经验年限提取 ─────────────────────────────────────────

_YEAR_RE = re.compile(
    r"(\d+)[\-–年\s]*年|"
    r"(\d+)[\+＋]年|"
    r"经验\s*(\d+)[\-–年\s]*年|"
    r"(\d+)\s*(?:years?|yrs?)",
    re.IGNORECASE
)

def _extract_years(text: str) -> List[int]:
    """从文本中提取所有数字化的年限值（去重）。"""
    raw = _YEAR_RE.findall(text)
    nums = []
    for group in raw:
        for g in group:
            if g:
                n = int(g)
                if 1 <= n <= 30:   # 过滤掉明显的异常值
                    nums.append(n)
    return list(set(nums))


# ── JD 核心要求解析 ─────────────────────────────────────

_SKILL_RE = re.compile(
    r"(?:熟悉|掌握|熟练|精通|了解|使用|具备|擅长|有\s*\d+\s*年)|"
    r"(?:\d+\+?\s*年)|"
    r"[\w\+\#\.\-]+", re.IGNORECASE
)

# 强度词 → 级别标签
_LEVEL_TAGS = {
    "精通": "硬核",
    "熟练": "熟练",
    "熟悉": "了解",
    "掌握": "掌握",
    "了解": "了解",
}

def _parse_jd(jd: str) -> Dict:
    """
    从 JD 文本中提取：
      - 核心技能关键词（纯词或短语）
      - 要求年限
    """
    raw_tokens = _tokenize(jd)
    counter = Counter(raw_tokens)

    # 出现 ≥2 次的词更可能是核心技能
    core = {w for w, c in counter.items() if c >= 2}

    # 太短的英文词跳过
    core = {w for w in core if len(w) >= 2}

    years = _extract_years(jd)

    # ── 新增：解析具体要求条款 ───────────────────────────
    clauses = _extract_jd_clauses(jd)

    return {"core_keywords": core, "years": years, "clauses": clauses}


def _extract_jd_clauses(jd: str) -> List[Dict]:
    """
    按行/句拆解 JD，提取每个具体要求条款。
    返回形如 [{"text": "熟悉Python和Django", "skills": ["python","django"], "level": null}]
    """
    lines = re.split(r"[\n\r•·\-–—]", jd)
    clauses = []
    seen = set()

    for raw in lines:
        line = raw.strip()
        if len(line) < 6 or len(line) > 200:
            continue

        tokens = set(_tokenize(line))
        if len(tokens) < 2:
            continue

        # 提取强度词
        level = None
        for kw, tag in _LEVEL_TAGS.items():
            if kw in line:
                level = tag
                break

        # 提取技能词（3字符以上，且非纯数字）
        skills = [t for t in tokens if len(t) >= 2 and not t.isdigit()]
        if not skills:
            continue

        # 去重（同条款内）
        key = " ".join(sorted(skills))
        if key in seen:
            continue
        seen.add(key)

        clauses.append({
            "text": line,
            "skills": skills,
            "level": level,
        })

    return clauses


# ── 核心评分函数 ─────────────────────────────────────────

def _skills_coverage(resume_tokens: List[str], jd_core: set) -> Tuple[float, set, set]:
    """
    技能关键词覆盖率。
    返回: (得分 0~1, 命中的词集合, 未命中的词集合)
    """
    resume_unique = set(resume_tokens)
    # 关键词匹配时支持子串（JD 要求 "python" → 简历里有 "python3" 也能命中）
    hits = {w for w in jd_core
            if w in resume_unique or any(w in r for r in resume_unique if len(w) >= 3)}
    missed = jd_core - hits
    if not jd_core:
        return 0.0, set(), set()
    return len(hits) / len(jd_core), hits, missed


def _years_match(resume_text: str, jd_years: List[int]) -> float:
    """
    经验年限匹配度。
    有 JD 年限要求时 → 简历年限 >= JD 要求为满分，不足按比例；
    无 JD 年限时 → 简历年限 >= 3 年给满分。
    """
    resume_years = _extract_years(resume_text)
    if not resume_years:
        return 0.5  # 无法判断时给中间值

    max_resume_year = max(resume_years)

    if not jd_years:
        return 1.0 if max_resume_year >= 3 else max_resume_year / 3

    min_jd_year = min(jd_years)
    if max_resume_year >= min_jd_year:
        return 1.0
    return max_resume_year / min_jd_year


def _jd_coverage_score(resume_text: str, jd: str) -> float:
    """
    JD 行覆盖率：统计简历中涵盖了 JD 多少个句子级别的要求。
    简化：用 JD 中每 3 句话作为一个需求单元。
    """
    # 把 JD 按标点和换行拆成句子
    sentences = re.split(r"[。；\n\.；]", jd)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if not sentences:
        return 0.5

    resume_tokens_set = set(_tokenize(resume_text))
    hits = 0
    for s in sentences:
        stokens = set(_tokenize(s))
        # 如果这句话的核心词有一半以上出现在简历里，视为覆盖
        if not stokens:
            continue
        covered = sum(1 for t in stokens if t in resume_tokens_set)
        if covered / len(stokens) >= 0.4:
            hits += 1

    return hits / len(sentences)


def _keyword_tfidf_score(resume: str, jd: str) -> float:
    """
    TF-IDF 加权关键词得分。
    把 JD 和简历看作两个文档，计算 JD 关键词在简历中的 TF-IDF 权重。
    """
    resume_tokens = _tokenize(resume)
    jd_tokens = _tokenize(jd)

    idf = _idf([resume_tokens, jd_tokens])
    resume_tf = _tf(resume_tokens)
    jd_tf = _tf(jd_tokens)

    resume_tfidf = _tfidf(resume_tf, idf)
    jd_tfidf = _tfidf(jd_tf, idf)

    # 取 JD 中 TF-IDF 权重最高的 20 个词
    top_jd = sorted(jd_tfidf.items(), key=lambda x: -x[1])[:20]
    if not top_jd:
        return 0.5

    resume_unique = set(resume_tokens)
    score = 0.0
    for w, w_idf in top_jd:
        if w in resume_unique or any(w in r for r in resume_unique if len(w) >= 3):
            score += w_idf

    # 归一化：最高分是所有 top_jd 的 idf 总和
    max_score = sum(idf_val for _, idf_val in top_jd)
    return min(score / max_score, 1.0) if max_score > 0 else 0.5


# ── 主匹配函数 ─────────────────────────────────────────

def match_score(resume: str, jd: str) -> Tuple[float, List[Dict]]:
    """
    计算简历与 JD 的综合匹配度（0~100）。

    评分权重：
      技能关键词覆盖率  × 40%
      TF-IDF 加权得分   × 30%
      经验年限匹配      × 15%
      JD 需求覆盖率    × 15%

    返回：
      score        (float)  综合得分 0~100
      suggestions  (List[Dict])  STAR 法则结构化建议
    """
    resume_tokens = _tokenize(resume)
    resume_unique = set(resume_tokens)

    # 解析 JD
    jd_parsed  = _parse_jd(jd)
    jd_core    = jd_parsed["core_keywords"]
    jd_years   = jd_parsed["years"]
    clauses    = jd_parsed["clauses"]

    # 子项得分
    cov_score, hits, missed = _skills_coverage(resume_tokens, jd_core)
    tfidf_score             = _keyword_tfidf_score(resume, jd)
    years_score             = _years_match(resume, jd_years)
    jd_cov_score            = _jd_coverage_score(resume, jd)

    # 加权总分
    total = (
        cov_score    * 0.40 +
        tfidf_score  * 0.30 +
        years_score  * 0.15 +
        jd_cov_score * 0.15
    )
    score = round(total * 100, 1)

    # ── STAR 法则生成改进建议 ─────────────────────────────
    suggestions = _build_star_suggestions(resume, resume_unique, clauses,
                                           jd_core, jd_years,
                                           cov_score, tfidf_score,
                                           years_score, jd_cov_score)

    return score, suggestions


def _build_star_suggestions(
    resume: str, resume_unique: set,
    clauses: List[Dict], jd_core: set, jd_years: List[int],
    cov_score: float, tfidf_score: float,
    years_score: float, jd_cov_score: float
) -> List[Dict]:
    """
    用 STAR 法则为每条未覆盖的 JD 条款生成结构化改进建议：
      S – Situation   背景：在什么项目/业务场景下
      T – Task        任务：负责什么，具体目标是什么
      A – Action      行动：用了什么技术栈、怎么做的
      R – Result      结果：量化成果（如 QPS 提升 X%，响应降低 Yms）
    """

    def clause_hit(clause: Dict) -> bool:
        """判断简历是否覆盖了该条款的核心技能。"""
        for sk in clause["skills"]:
            if sk in resume_unique or any(sk in r for r in resume_unique if len(sk) >= 3):
                return True
        return False

    suggestions = []

    # ① 按 STAR 法则逐条生成未覆盖的 JD 条款建议
    for clause in clauses:
        if clause_hit(clause):
            continue

        skills_txt = "、".join(f"「{s}」" for s in clause["skills"][:4])
        level_hint = f"（要求：{clause['level']}）" if clause["level"] else ""

        suggestion = {
            "clause":  clause["text"],
            "tag":     "缺失",
            "skill":   clause["skills"][0] if clause["skills"] else "",
            "star": {
                "S": f"在你参与过的项目或工作经历中，涉及过 {skills_txt} {level_hint} 的业务场景。",
                "T": f"你需要在该项目中承担 / 主导 {skills_txt} 相关的技术实现或优化工作。",
                "A": (
                    f"使用 {skills_txt} 完成具体开发或优化，可描述："
                    "采用了什么方案、解决了什么技术难点、如何保证质量。"
                ),
                "R": (
                    f"给出可量化的结果，例如："
                    "系统性能提升 X%、日均处理请求量达 N 条、稳定性保持 99.9% 以上。"
                ),
            }
        }
        suggestions.append(suggestion)

    # ② 整体层面建议
    if cov_score < 0.5:
        top_missing = sorted(jd_core - {w for w in jd_core
                     if any(w in r for r in resume_unique)},
                     key=lambda x: -len(x))[:3]
        if top_missing:
            missing_txt = "、".join(f"「{w}」" for w in top_missing)
            suggestions.append({
                "clause":  missing_txt,
                "tag":     "关键词不足",
                "skill":   top_missing[0],
                "star": {
                    "S": "回顾你过往的项目经历，找出与缺失关键词相关的场景。",
                    "T": "明确你在该场景中需要体现的技能方向。",
                    "A": "在简历中用具体行为动词（设计、开发、优化）描述技术动作。",
                    "R": "附上量化指标（提升幅度、规模、收益）。",
                }
            })

    if years_score < 0.7 and jd_years:
        min_y = min(jd_years)
        suggestions.append({
            "clause":  f"JD 要求 ≥ {min_y} 年经验",
            "tag":     "年限不足",
            "skill":   "工作经验",
            "star": {
                "S": f"回顾累计工作年限是否达到 {min_y} 年（含实习、兼职、项目经历）。",
                "T": "明确总工作年限后，在简历开头或技能摘要中清晰呈现。",
                "A": "将年限写在工作经历标题旁（如「3 年 Python 后端开发经验」），或写入自我介绍。",
                "R": "招聘系统可自动识别，HR 第一眼即可确认达标。",
            }
        })

    if tfidf_score < 0.35:
        suggestions.append({
            "clause":  "简历用语与 JD 关键词不一致",
            "tag":     "语言匹配",
            "skill":   "简历表述",
            "star": {
                "S": "JD 中反复出现的核心词汇（如 React、API 设计、MySQL）代表招聘方最看重的技能。",
                "T": "将简历中对应的技术描述改为与 JD 一致的用语。",
                "A": (
                    "逐条对照 JD 高频词，把简历里的「用了 XX 技术」"
                    "改为 JD 中使用的专业词汇（如「开发」→「构建」「设计」）。"
                ),
                "R": "用语对齐后，ATS 系统和 HR 扫描都能快速识别匹配度。",
            }
        })

    if not suggestions:
        suggestions.append({
            "clause":  "简历整体匹配度良好",
            "tag":     "优秀",
            "skill":   "—",
            "star": {
                "S": "你的简历已覆盖 JD 核心技能需求，语言表达与岗位要求一致。",
                "T": "可针对细节进一步打磨，争取更高分。",
                "A": "检查每段经历是否都有量化的 R（结果）描述；确保时间线清晰连贯。",
                "R": "优秀简历往往赢在「结果的量化」——数字是最有说服力的表达。",
            }
        })

    return suggestions
