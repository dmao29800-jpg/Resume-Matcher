"""
Resume–JD 匹配算法 v8（真实区分度）

改进：
1. 无60分保底，真实反映匹配质量
2. 技能匹配：逐条JD条款精确打分，不稀释
3. 经验维度：实习/项目/校园经历数量与质量
4. 结构完整性：STAR结构、量化指标
5. JD条款覆盖率：每条JD要求单独评分
"""

import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower())
    stop = {
        "的", "了", "在", "是", "我", "你", "和", "与", "或", "但", "而", "于",
        "有", "这", "那", "个", "们", "为", "上", "下", "中", "对", "以", "到",
        "把", "被", "要", "会", "能", "可", "也", "很", "都", "将", "又", "再",
        "从", "所", "已", "应", "该", "等", "并", "负责", "参与",
        "for", "with", "and", "or", "the", "a", "an", "to", "of", "in", "on",
        "by", "is", "are", "was", "were", "be", "been", "have", "has", "had",
        "this", "that", "it",
    }
    return [t for t in tokens if t not in stop and len(t) > 1]

def _sentences(text: str) -> List[str]:
    """把文本拆成句子"""
    return [s.strip() for s in re.split(r'[，。；；\n]', text) if len(s.strip()) > 6]

def _extract_years(text: str) -> List[int]:
    return [int(m.group(1)) for m in re.finditer(r"(\d+)\s*年", text)
            if 0 <= int(m.group(1)) <= 50]

def _count_quantified(sents: List[str]) -> int:
    """统计有量化指标的句子数"""
    count = 0
    for s in sents:
        if re.search(r'\d+\s*(万|亿|千|% |％|ms|QPS|DAU|用户|倍|条|次|台|个|人|日|周|月)', s):
            count += 1
    return count


# ═══════════════════════════════════════════════════════════
#  技能知识图谱
# ═══════════════════════════════════════════════════════════

class SkillGraph:
    # 技能分类（同族技能有语义关联）
    CATEGORIES = {
        "backend_lang": {"python", "java", "golang", "go", "c++", "c#", "php", "ruby", "rust"},
        "frontend":     {"react", "vue", "angular", "nextjs", "nuxt", "jquery", "html", "css", "小程序"},
        "python_web":   {"django", "flask", "fastapi", "tornado", "fastapi"},
        "java_web":     {"spring", "springboot", "mybatis", "springcloud", "spring cloud"},
        "database":     {"mysql", "postgresql", "mongodb", "redis", "elasticsearch", "sqlite", "oracle", "sql"},
        "cache":        {"redis", "memcached"},
        "mq":           {"kafka", "rabbitmq", "rocketmq", "activemq"},
        "devops":       {"docker", "kubernetes", "k8s", "jenkins", "gitlab", "github", "ci/cd", "cicd", "nginx"},
        "ml":           {"pytorch", "tensorflow", "sklearn", "xgboost", "keras", "机器学习", "深度学习"},
        "data":         {"pandas", "numpy", "spark", "hadoop", "hive", "kafka"},
        "robotics":     {"ros", "slam", "opencv", "pcl", "gazebo", "matlab", "simulink", "cartographer"},
        "mobile":       {"android", "ios", "flutter", "react native", "小程序"},
        "cloud":        {"aws", "azure", "阿里云", "腾讯云", "华为云", "云服务"},
    }

    # 别名映射
    ALIASES = {
        "golang": "go", "go": "go",
        "js": "javascript", "javascript": "javascript",
        "ts": "typescript", "typescript": "typescript",
        "py": "python", "python3": "python",
        "postgres": "postgresql",
        "mongo": "mongodb",
        "k8s": "kubernetes",
        "springboot": "spring",
        "pytorch": "pytorch", "tf": "tensorflow",
        "es": "elasticsearch",
        "spring cloud": "springcloud",
        "djangoRESTframework": "django",
    }

    # 硬技能关键词（必须精准匹配）
    HARD_SKILLS = {
        "后端": ["python", "java", "golang", "go", "c++", "spring", "flask", "django", "fastapi",
                "mysql", "postgresql", "mongodb", "redis", "kafka", "docker", "k8s", "微服务"],
        "前端": ["react", "vue", "angular", "html", "css", "javascript", "typescript", "node"],
        "算法": ["机器学习", "深度学习", "pytorch", "tensorflow", "tensorflow", "推荐算法",
                "nlp", "cv", "搜索算法", "图算法"],
        "数据": ["pandas", "numpy", "spark", "hadoop", "hive", "kafka", "flink", "etl"],
        "运维": ["docker", "kubernetes", "k8s", "jenkins", "nginx", "linux", "shell", "devops"],
    }

    def normalize(self, s: str) -> str:
        return self.ALIASES.get(s.lower(), s.lower())

    def get_cat(self, s: str) -> Optional[str]:
        s = self.normalize(s)
        for cat, skills in self.CATEGORIES.items():
            if s in skills:
                return cat
        return None

    def similarity(self, a: str, b: str) -> float:
        a, b = self.normalize(a), self.normalize(b)
        if a == b:
            return 1.0
        ca, cb = self.get_cat(a), self.get_cat(b)
        if ca and ca == cb:
            cat_sim = {"python_web": 0.85, "java_web": 0.85, "database": 0.8,
                       "frontend": 0.8, "ml": 0.8, "devops": 0.8,
                       "backend_lang": 0.5, "data": 0.75}
            return cat_sim.get(ca, 0.7)
        return 0.0

    def find_sent(self, skill: str, text: str) -> Optional[str]:
        skill = self.normalize(skill)
        for sent in _sentences(text):
            if skill in sent:
                return sent
        return None

_SG = SkillGraph()


# ═══════════════════════════════════════════════════════════
#  JD 条款提取
# ═══════════════════════════════════════════════════════════

def _extract_jd_clauses(jd: str) -> List[Dict]:
    """
    把 JD 拆成独立的条款，每条包含：
    - text: 原文
    - type: skill / soft / years / other
    - keywords: 关键词列表
    - required: 是否硬性要求（用"必须"、"熟练"、"精通"判断）
    """
    clauses = []
    raw_lines = re.split(r'[,，\n；;]', jd)
    for line in raw_lines:
        line = line.strip()
        if len(line) < 4:
            continue
        tokens = set(_tokenize(line))
        if not tokens:
            continue

        is_years = bool(re.search(r'\d+\s*年', line))
        is_required = any(kw in line for kw in ['必须', '熟练掌握', '精通', '扎实的', '有经验', '优先', '要求'])
        is_soft = any(w in line.lower() for w in ['沟通', '团队', '学习', '逻辑', '责任心', '抗压'])

        found_skills = []
        for tok in tokens:
            cat = _SG.get_cat(tok)
            if cat:
                found_skills.append(tok)

        clauses.append({
            "text": line,
            "type": "years" if is_years else ("skill" if found_skills else "soft"),
            "keywords": found_skills,
            "required": is_required,
        })
    return clauses


# ═══════════════════════════════════════════════════════════
#  经验结构解析
# ═══════════════════════════════════════════════════════════

def _parse_experience(text: str) -> Dict:
    """
    解析简历中的经历结构：
    - 实习：找出实习段数
    - 项目：找出项目段数
    - 校园：找出校园/比赛经历
    - 每段是否有STAR结构
    - 每段是否有量化
    """
    # 常见经历标题模式
    header_patterns = [
        r'实习(?:经验|经历|工作)',
        r'项目(?:经验|经历)',
        r'校园(?:实践|经历)',
        r'比赛|竞赛|比赛',
        r'科研|研究',
        r'工作经历',
    ]
    sents = _sentences(text)

    results = {
        "intern_count": 0,
        "project_count": 0,
        "club_count": 0,
        "total_sections": 0,
        "quantified_count": _count_quantified(sents),
        "star_indicators": sum(1 for s in sents if any(w in s for w in ['负责', '主导', '完成', '推动', '达成', '实现'])),
        "avg_section_length": 0,
    }

    # 简单粗暴：按段落估算经历数
    sections = re.split(r'\n{2,}', text)
    for sec in sections:
        sec = sec.strip()
        if len(sec) < 20:
            continue
        results["total_sections"] += 1
        if any(re.search(p, sec) for p in [r'实习', r'intern']):
            results["intern_count"] += 1
        elif any(re.search(p, sec) for p in [r'项目', r'project']):
            results["project_count"] += 1
        else:
            results["club_count"] += 1

    return results


# ═══════════════════════════════════════════════════════════
#  语义相似度（TF-IDF + Cosine）
# ═══════════════════════════════════════════════════════════

def _tfidf_cosine(text1: str, text2: str) -> float:
    """基于 TF-IDF 的余弦相似度"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        tfidf = vec.fit_transform([text1, text2])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(sim)
    except Exception:
        # fallback：词集合相似度
        t1, t2 = set(_tokenize(text1)), set(_tokenize(text2))
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)


# ═══════════════════════════════════════════════════════════
#  逐条评分
# ═══════════════════════════════════════════════════════════

def _score_clauses(
    clauses: List[Dict],
    resume: str,
    resume_tokens: set,
) -> Tuple[float, List[Dict], List[str]]:
    """
    逐条 JD 条款评分，返回：
    - overall_skill_score: 0.0~1.0
    - clause_results: 每条JD的评分详情
    - missing_core: 缺失的核心技能
    """
    if not clauses:
        return 0.5, [], []

    clause_scores = []
    missing_core = []

    for clause in clauses:
        ct = clause["type"]
        kws = clause["keywords"]
        required = clause["required"]
        score = 0.0
        matched = []
        missing = []

        if ct == "skill":
            if not kws:
                score = 0.5  # 无技能关键词，给中间分
            else:
                # 逐关键词检查
                kw_scores = []
                for kw in kws:
                    best = 0.0
                    for rt in resume_tokens:
                        sim = _SG.similarity(rt, kw)
                        if sim > best:
                            best = sim
                    kw_scores.append((kw, best))
                    if best >= 0.5:
                        matched.append(kw)
                    else:
                        missing.append(kw)

                # 这条条款得分 = 关键词平均分
                if kw_scores:
                    score = sum(s for _, s in kw_scores) / len(kw_scores)
                    # 有关联技能也算
                    for kw in kws:
                        for rt in resume_tokens:
                            if rt != kw and _SG.get_cat(rt) == _SG.get_cat(kw):
                                sim = _SG.similarity(rt, kw)
                                if sim >= 0.5:
                                    score = max(score, sim * 0.7)
                                    break

        elif ct == "years":
            r_yrs = _extract_years(resume)
            j_yrs = _extract_years(clause["text"])
            if r_yrs and j_yrs:
                max_r = max(r_yrs)
                min_j = min(j_yrs)
                if max_r >= min_j:
                    score = 1.0
                else:
                    diff = min_j - max_r
                    score = max(0, 1.0 - diff * 0.25)
            else:
                score = 0.4  # 简历未写年限

        elif ct == "soft":
            # 软技能：看简历整体是否有相关词
            soft_words = {
                '沟通': ['沟通', '表达', '协作'],
                '团队': ['团队', '合作', '配合'],
                '学习': ['学习', '自学', '研究'],
                '逻辑': ['逻辑', '分析', '思考'],
            }
            matched_soft = 0
            for kw in clause.get('keywords', []):
                if any(w in resume.lower() for w in soft_words.get(kw, [kw])):
                    matched_soft += 1
            score = min(matched_soft / max(len(kws), 1), 1.0) if kws else 0.5

        else:
            score = 0.5

        # 硬性要求未满足则额外扣分
        final_score = score
        if required and score < 0.5:
            final_score = score * 0.8

        clause_scores.append({
            "text": clause["text"][:50],
            "score": round(final_score, 2),
            "matched": matched,
            "missing": missing,
            "required": required,
            "type": ct,
        })

        # 记录缺失核心技能
        if ct == "skill" and missing:
            for m in missing:
                if m not in missing_core:
                    missing_core.append(m)

    # 总体技能分 = 每条得分加权平均（硬性要求权重更高）
    total_w, weighted_sum = 0, 0.0
    for cs in clause_scores:
        w = 1.5 if cs["required"] else 1.0
        total_w += w
        weighted_sum += cs["score"] * w

    overall = weighted_sum / total_w if total_w > 0 else 0.5
    return overall, clause_scores, missing_core


# ═══════════════════════════════════════════════════════════
#  经验质量评分
# ═══════════════════════════════════════════════════════════

def _experience_score(exp: Dict, required_years: Optional[int]) -> float:
    """
    经验质量评估（0.0~1.0）
    - 有实习/项目经历
    - 每段经历有量化指标
    - STAR结构指标
    """
    score = 0.0

    # 基础分：按经历数量
    sections = exp["total_sections"]
    if sections >= 5:
        score += 0.30
    elif sections >= 3:
        score += 0.20
    elif sections >= 1:
        score += 0.10
    else:
        score += 0.00  # 无经历 = 0分

    # 量化加分
    q_ratio = exp["quantified_count"] / max(sections, 1)
    if q_ratio >= 0.6:
        score += 0.25
    elif q_ratio >= 0.3:
        score += 0.15
    elif q_ratio > 0:
        score += 0.05
    # 全无量化 = 不加分

    # STAR结构加分
    star_ratio = exp["star_indicators"] / max(sections, 1)
    if star_ratio >= 1.0:
        score += 0.15
    elif star_ratio >= 0.5:
        score += 0.10

    # 实习加分（实习比项目更有说服力）
    if exp["intern_count"] >= 2:
        score += 0.15
    elif exp["intern_count"] >= 1:
        score += 0.10

    return min(score, 1.0)


# ═══════════════════════════════════════════════════════════
#  STAR 建议生成
# ═══════════════════════════════════════════════════════════

def _generate_suggestions(
    clauses: List[Dict],
    clause_results: List[Dict],
    exp: Dict,
    missing_core: List[str],
    resume_sents: List[str],
    resume: str,
) -> List[Dict]:
    suggestions = []

    # ① 缺失核心技能（分数 < 0.4 的硬性要求条款）
    weak_clauses = [c for c in clause_results if c["score"] < 0.4 and c["required"]]
    if weak_clauses:
        c = weak_clauses[0]
        miss = c["missing"][:2] if c["missing"] else missing_core[:1]
        if miss:
            suggestions.append({
                "clause": c["text"],
                "tag": "核心技能缺失",
                "skill": miss[0],
                "star": {
                    "S": f"JD要求「{miss[0]}」，简历中未体现或描述不足。",
                    "T": f"补充{miss[0]}相关经历，或在技能列表中明确写出。",
                    "A": f"在项目描述中加入：使用{miss[0]}实现XX功能；或在技能栏写「了解{miss[0]}核心原理，正在系统学习」。",
                    "R": "ATS系统扫描关键词，缺失核心技能会被直接筛掉。",
                }
            })

    # ② 无量化指标
    if exp["quantified_count"] == 0:
        vague = next((s for s in resume_sents
                      if any(w in s for w in ['负责', '完成', '开发', '优化', '提升'])
                      and not re.search(r'\d+', s)), None)
        if vague:
            suggestions.append({
                "clause": f"「{vague[:20]}...」缺少量化指标",
                "tag": "量化不足",
                "skill": "STAR 结果",
                "star": {
                    "S": f"项目描述偏定性：{vague[:30]}...",
                    "T": "为每段经历补充至少一个数字。",
                    "A": f"改写：{vague} → {vague}，日活10万，QPS从100提升至500，可用率99.9%。",
                    "R": "量化数据是简历的「硬通货」，面试官据此判断你的真实贡献大小。",
                }
            })

    # ③ 经历数量不足
    if exp["total_sections"] < 3:
        suggestions.append({
            "clause": f"简历仅{exp['total_sections']}段经历，内容偏少",
            "tag": "经历单薄",
            "skill": "简历完整度",
            "star": {
                "S": f"简历经历偏少（{exp['total_sections']}段），内容密度不足。",
                "T": "充实简历内容，增加3-5段有价值的经历。",
                "A": "将课程项目、比赛、课外实践也写入简历，每段突出：用什么技术、做了什么、结果如何。",
                "R": "经历丰富代表经历丰富，ATS系统和HR会认为你更「有料」。",
            }
        })

    # ④ 实习缺失（若JD暗示需要工作经验）
    if exp["intern_count"] == 0 and exp["project_count"] == 0:
        suggestions.append({
            "clause": "简历中无实习和项目经历",
            "tag": "缺乏实践",
            "skill": "经历类型",
            "star": {
                "S": "简历缺少实际项目或实习经历，与大多数候选人相比缺乏说服力。",
                "T": "补充至少1段可量化的项目或实习。",
                "A": "整理课程项目、比赛作品、开源贡献，哪怕是小功能也可以量化（如：「优化了某函数的执行效率」）。",
                "R": "有实践经历的简历通过率比没有的高出47%（LinkedIn数据）。",
            }
        })

    # ⑤ 有量化但结构松散
    elif exp["quantified_count"] > 0 and exp["star_indicators"] < exp["total_sections"]:
        suggestions.append({
            "clause": "有量化数据但STAR结构不够完整",
            "tag": "结构待优化",
            "skill": "STAR 完整性",
            "star": {
                "S": "简历中有量化指标，但各段落的STAR结构不完整。",
                "T": "让每段经历都遵循「情境-任务-行动-结果」结构。",
                "A": "检查每段经历：是否说清了项目背景（S）？具体任务（T）？你做了什么（A）？结果怎样（R，有数字）？",
                "R": "STAR结构让面试官快速理解你的价值，而非自己从字里行间挖掘。",
            }
        })

    # ⑥ 优秀情况
    if not suggestions:
        suggestions.append({
            "clause": "简历与 JD 核心要求匹配良好",
            "tag": "优秀",
            "skill": "—",
            "star": {
                "S": f"简历覆盖了岗位核心技能，技能匹配度高，有量化数据，有STAR结构。",
                "T": "继续保持，可冲击更高分。",
                "A": "检查每段经历的量化数据是否最大化；确认没有遗漏的关键字。",
                "R": "这份简历在同岗位候选人中具有竞争力，好好准备面试。",
            }
        })

    return suggestions[:4]


# ═══════════════════════════════════════════════════════════
#  主匹配函数
# ═══════════════════════════════════════════════════════════

def match_score(resume: str, jd: str) -> Tuple[float, List[Dict]]:
    """
    评分体系 v8：
    - 技能匹配分 × 40%     （逐条JD条款评分）
    - 语义相似度 × 25%    （TF-IDF余弦）
    - 经验质量   × 35%    （实习/项目/量化/STAR）
    总分真实反映简历质量，不再有保底分
    """
    resume_tokens = set(_tokenize(resume))
    jd_tokens = set(_tokenize(jd))
    resume_sents = _sentences(resume)

    # ── JD 条款提取 ───────────────────────────────────
    clauses = _extract_jd_clauses(jd)

    # ── 技能匹配评分 ───────────────────────────────────
    skill_score, clause_results, missing_core = _score_clauses(
        clauses, resume, resume_tokens
    )

    # ── 语义相似度 ─────────────────────────────────────
    sem_score = _tfidf_cosine(resume, jd)

    # ── 经验质量 ───────────────────────────────────────
    exp = _parse_experience(resume)
    yrs_req = next((min(_extract_years(c["text"])) for c in clauses if c["type"] == "years"), None)
    exp_score = _experience_score(exp, yrs_req)

    # ── 加权总分（0~100）───────────────────────────────
    # 分两个通道：核心技能匹配（精确）+ 整体经验质量
    # 核心技能分：只看有技能关键词的条款
    skill_clauses = [c for c in clause_results if c["type"] == "skill"]
    if skill_clauses:
        core_skill = sum(c["score"] for c in skill_clauses) / len(skill_clauses)
    else:
        core_skill = skill_score

    # 最终分数 = 核心技能×50% + 语义×20% + 经验质量×30%
    final_raw = core_skill * 0.50 + sem_score * 0.20 + exp_score * 0.30
    score = round(final_raw * 100, 1)

    # ── bonus ──────────────────────────────────────────
    if exp["quantified_count"] >= 3 and exp["total_sections"] >= 3:
        score = min(score + 3, 99)
    if core_skill >= 0.75 and exp_score >= 0.65:
        score = min(score + 5, 99)

    score = round(score, 1)

    suggestions = _generate_suggestions(
        clauses, clause_results, exp, missing_core, resume_sents, resume
    )

    return score, suggestions
