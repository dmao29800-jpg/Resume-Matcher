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

    # 补充：太短的英文词跳过（如"js"、"py"单个字符）
    core = {w for w in core if len(w) >= 2}

    years = _extract_years(jd)
    return {"core_keywords": core, "years": years}


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

def match_score(resume: str, jd: str) -> Tuple[float, List[str]]:
    """
    计算简历与 JD 的综合匹配度（0~100）。

    评分权重：
      技能关键词覆盖率  × 40%
      TF-IDF 加权得分   × 30%
      经验年限匹配      × 15%
      JD 需求覆盖率    × 15%
    """
    resume_tokens = _tokenize(resume)

    # 解析 JD
    jd_parsed = _parse_jd(jd)
    jd_core = jd_parsed["core_keywords"]
    jd_years = jd_parsed["years"]

    # 子项得分
    cov_score, hits, missed = _skills_coverage(resume_tokens, jd_core)
    tfidf_score = _keyword_tfidf_score(resume, jd)
    years_score = _years_match(resume, jd_years)
    jd_cov_score = _jd_coverage_score(resume, jd)

    # 加权总分
    total = (
        cov_score    * 0.40 +
        tfidf_score  * 0.30 +
        years_score  * 0.15 +
        jd_cov_score * 0.15
    )

    # 归一化到 0~100
    score = round(total * 100, 1)

    # ── 生成改进建议 ──────────────────────────────────────
    suggestions = []

    if cov_score < 0.6:
        top_missing = sorted(missed, key=lambda x: -len(x))[:5]
        for kw in top_missing:
            suggestions.append(f"建议在简历中补充或强化「{kw}」相关技能描述")

    if years_score < 0.6 and jd_years:
        min_y = min(jd_years)
        suggestions.append(
            f"JD 要求至少 {min_y} 年经验，"
            f"建议在简历中明确写出工作年限以提升匹配度"
        )

    if tfidf_score < 0.4:
        suggestions.append(
            "简历与 JD 的整体用语匹配度偏低，"
            "建议参考 JD 关键词，适当调整简历描述方式"
        )

    if jd_cov_score < 0.5:
        suggestions.append(
            "简历覆盖的 JD 需求点较少，"
            "建议逐条对照 JD 要求，在简历中补充对应内容"
        )

    if not suggestions:
        suggestions.append("简历整体匹配度良好，针对上述细节可进一步优化。")

    return score, suggestions
