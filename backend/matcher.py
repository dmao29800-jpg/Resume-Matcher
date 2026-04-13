"""
Resume–JD 匹配算法 v9
科学性改进：
1. Sentence-BERT Embedding 语义匹配（接通已加载的 BERT 模型）
2. 硬技能惩罚项（required 技能缺失 → 一票否决）
3. 动词能级权重（高级动词 ×1.2，低级动词 ×0.5）
"""

import re
from typing import List, Dict, Tuple, Optional

# ═══════════════════════════════════════════════════════════
#  全局 Embedding 模型（单例，延迟加载）
# ═══════════════════════════════════════════════════════════

_EMBEDDER = None

def _get_embedder():
    """延迟加载 Sentence-BERT 模型"""
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            print(f"[Embedding] 模型加载失败: {e}，降级为 TF-IDF")
            _EMBEDDER = "fallback"
    return _EMBEDDER


def _embedding_score(resume: str, jd: str) -> float:
    """
    用 Sentence-BERT 计算简历与 JD 的语义相似度。
    逻辑：JD 每条条款与简历所有句子两两做余弦相似度，取最大值的平均。
    """
    model = _get_embedder()

    if model == "fallback":
        return _tfidf_cosine(resume, jd)

    try:
        jd_clauses = [c.strip() for c in re.split(r'[,，\n；;]', jd) if len(c.strip()) > 6]
        resume_sents = [s.strip() for s in re.split(r'[，。；；\n]', resume) if len(s.strip()) > 6]

        if not jd_clauses or not resume_sents:
            return _tfidf_cosine(resume, jd)

        # encode 两端（不传 normalize，手动计算余弦相似度）
        emb_jd = model.encode(jd_clauses)
        emb_rs = model.encode(resume_sents)

        # 余弦相似度：手动 L2 normalize 后做 dot product
        def _norm(x):
            return x / (x ** 2).sum(axis=1, keepdims=True) ** 0.5
        emb_jd_n = _norm(emb_jd)
        emb_rs_n = _norm(emb_rs)
        sim_matrix = emb_jd_n @ emb_rs_n.T  # (n_jd, n_rs)

        # 每条 JD 取其与简历最相似的句子的分数
        clause_best = sim_matrix.max(axis=1).tolist()

        # 综合得分：对各条款加权平均（硬性条款权重更高）
        weights = []
        for clause in jd_clauses:
            w = 1.5 if any(k in clause for k in ['必须', '熟练', '精通', '扎实的', '要求', '优先']) else 1.0
            weights.append(w)
        w_sum = sum(weights)
        score = sum(c * w for c, w in zip(clause_best, weights)) / w_sum
        return float(score)

    except Exception as e:
        print(f"[Embedding] 推理失败: {e}，降级 TF-IDF")
        return _tfidf_cosine(resume, jd)


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
    return [s.strip() for s in re.split(r'[，。；；\n]', text) if len(s.strip()) > 6]

def _extract_years(text: str) -> List[int]:
    return [int(m.group(1)) for m in re.finditer(r"(\d+)\s*年", text)
            if 0 <= int(m.group(1)) <= 50]

def _count_quantified(sents: List[str]) -> int:
    count = 0
    for s in sents:
        if re.search(r'\d+\s*(万|亿|千|% |％|ms|QPS|DAU|用户|倍|条|次|台|个|人|日|周|月|%|qps|dau)', s, re.I):
            count += 1
    return count


# ═══════════════════════════════════════════════════════════
#  动词能级权重表
# ═══════════════════════════════════════════════════════════

VERB_POWER = {
    # 高级动词 ×1.2
    "high": {
        "架构", "重构", "主导", "调优", "落地", "设计", "规划", "治理",
        "优化", "升级", "演进", "创新", "孵化", "自研", "原创",
        "cto", "技术负责人", "首席", "决策", "把关", "统筹",
        "全链路", "端到端", "搭建", "构建", "建立",
    },
    # 低级动词 ×0.5
    "low": {
        "了解", "协助", "学习", "参与", "接触", "熟悉", "看过", "用过",
        "写过", "写过", "学过", "研究过", "简单", "初步",
    },
}

def _verb_power(sent: str) -> float:
    """返回句子的动词能级乘数（默认 1.0）"""
    high = any(v in sent for v in VERB_POWER["high"])
    low = any(v in sent for v in VERB_POWER["low"])
    if high:
        return 1.2
    if low:
        return 0.5
    return 1.0


# ═══════════════════════════════════════════════════════════
#  技能知识图谱
# ═══════════════════════════════════════════════════════════

class SkillGraph:
    CATEGORIES = {
        "backend_lang": {"python", "java", "golang", "go", "c++", "c#", "php", "ruby", "rust"},
        "frontend":     {"react", "vue", "angular", "nextjs", "nuxt", "jquery", "html", "css", "小程序"},
        "python_web":   {"django", "flask", "fastapi", "tornado"},
        "java_web":     {"spring", "springboot", "mybatis", "springcloud", "spring cloud"},
        "database":     {"mysql", "postgresql", "mongodb", "redis", "elasticsearch", "sqlite", "oracle", "sql"},
        "cache":        {"redis", "memcached"},
        "mq":           {"kafka", "rabbitmq", "rocketmq", "activemq"},
        "devops":       {"docker", "kubernetes", "k8s", "jenkins", "gitlab", "github", "ci/cd", "cicd", "nginx"},
        "ml":           {"pytorch", "tensorflow", "sklearn", "xgboost", "keras", "机器学习", "深度学习"},
        "data":         {"pandas", "numpy", "spark", "hadoop", "hive", "kafka", "flink"},
        "robotics":     {"ros", "slam", "opencv", "pcl", "gazebo", "matlab", "simulink", "cartographer"},
        "mobile":       {"android", "ios", "flutter", "react native", "小程序"},
        "cloud":        {"aws", "azure", "阿里云", "腾讯云", "华为云"},
    }

    ALIASES = {
        "golang": "go", "go": "go",
        "js": "javascript", "javascript": "javascript",
        "ts": "typescript", "typescript": "typescript",
        "py": "python", "python3": "python",
        "postgres": "postgresql", "mongo": "mongodb",
        "k8s": "kubernetes", "springboot": "spring",
        "pytorch": "pytorch", "tf": "tensorflow",
        "es": "elasticsearch", "spring cloud": "springcloud",
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
    sents = _sentences(text)
    results = {
        "intern_count": 0,
        "project_count": 0,
        "club_count": 0,
        "total_sections": 0,
        "quantified_count": _count_quantified(sents),
        "star_indicators": 0,
        "weighted_verbs": 0.0,  # 动词能级加权和
    }

    sections = re.split(r'\n{2,}', text)
    for sec in sections:
        sec = sec.strip()
        if len(sec) < 20:
            continue
        results["total_sections"] += 1
        sec_lower = sec.lower()

        if any(re.search(p, sec_lower) for p in [r'实习', r'intern']):
            results["intern_count"] += 1
        elif any(re.search(p, sec_lower) for p in [r'项目', r'project']):
            results["project_count"] += 1
        else:
            results["club_count"] += 1

        # 动词能级加权
        sec_sents = _sentences(sec)
        for s in sec_sents:
            power = _verb_power(s)
            if any(w in s for w in ['负责', '主导', '完成', '推动', '达成', '实现']):
                results["weighted_verbs"] += power
                results["star_indicators"] += 1

    return results


# ═══════════════════════════════════════════════════════════
#  TF-IDF 余弦（Embedding 降级用）
# ═══════════════════════════════════════════════════════════

def _tfidf_cosine(text1: str, text2: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        tfidf = vec.fit_transform([text1, text2])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        t1, t2 = set(_tokenize(text1)), set(_tokenize(text2))
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)


# ═══════════════════════════════════════════════════════════
#  逐条技能评分
# ═══════════════════════════════════════════════════════════

def _score_clauses(
    clauses: List[Dict],
    resume: str,
    resume_tokens: set,
) -> Tuple[float, List[Dict], List[str]]:
    if not clauses:
        return 0.5, [], []

    clause_scores = []
    missing_core = []

    for clause in clauses:
        ct = clause["type"]
        kws = clause["keywords"]
        required = clause["required"]
        score = 0.0
        matched, missing = [], []

        if ct == "skill":
            if not kws:
                score = 0.5
            else:
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

                if kw_scores:
                    score = sum(s for _, s in kw_scores) / len(kw_scores)
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
                score = 1.0 if max_r >= min_j else max(0, 1.0 - (min_j - max_r) * 0.25)
            else:
                score = 0.4

        elif ct == "soft":
            soft_map = {
                '沟通': ['沟通', '表达', '协作'],
                '团队': ['团队', '合作', '配合'],
                '学习': ['学习', '自学', '研究'],
                '逻辑': ['逻辑', '分析', '思考'],
            }
            matched_soft = sum(
                1 for kw in kws if any(w in resume.lower() for w in soft_map.get(kw, [kw]))
            )
            score = min(matched_soft / max(len(kws), 1), 1.0) if kws else 0.5

        else:
            score = 0.5

        # 硬性要求扣分
        final_score = score * 0.8 if (required and score < 0.5) else score

        clause_scores.append({
            "text": clause["text"][:50],
            "score": round(final_score, 2),
            "matched": matched,
            "missing": missing,
            "required": required,
            "type": ct,
        })

        if ct == "skill" and missing:
            for m in missing:
                if m not in missing_core:
                    missing_core.append(m)

    total_w, weighted_sum = 0, 0.0
    for cs in clause_scores:
        w = 1.5 if cs["required"] else 1.0
        total_w += w
        weighted_sum += cs["score"] * w

    overall = weighted_sum / total_w if total_w > 0 else 0.5
    return overall, clause_scores, missing_core


# ═══════════════════════════════════════════════════════════
#  经验质量评分（动词能级版）
# ═══════════════════════════════════════════════════════════

def _experience_score(exp: Dict, required_years: Optional[int]) -> float:
    score = 0.0
    sections = exp["total_sections"]

    # ── 基础分：经历数量 ─────────────────────────────────
    if sections >= 5:
        score += 0.25
    elif sections >= 3:
        score += 0.18
    elif sections >= 1:
        score += 0.10
    else:
        score += 0.00

    # ── 量化加分 ───────────────────────────────────────
    q_ratio = exp["quantified_count"] / max(sections, 1)
    if q_ratio >= 0.6:
        score += 0.25
    elif q_ratio >= 0.3:
        score += 0.15
    elif q_ratio > 0:
        score += 0.05

    # ── STAR + 动词能级（关键改进）────────────────────────
    star_ratio = exp["star_indicators"] / max(sections, 1)
    wv = exp["weighted_verbs"]
    if star_ratio >= 1.0 and wv >= sections * 1.2:
        score += 0.20  # 动词质量高，STAR 完整
    elif star_ratio >= 0.6:
        score += 0.12
    elif star_ratio >= 0.3:
        score += 0.06

    # ── 实习加分 ───────────────────────────────────────
    if exp["intern_count"] >= 2:
        score += 0.15
    elif exp["intern_count"] >= 1:
        score += 0.10

    # ── 年限补足 ───────────────────────────────────────
    if required_years:
        r_yrs = 0  # 简单估算
        if r_yrs >= required_years:
            score = min(score + 0.10, 1.0)

    return min(score, 1.0)


# ═══════════════════════════════════════════════════════════
#  面试题预测（基于 JD 关键词触发）
# ═══════════════════════════════════════════════════════════

def _generate_interview_tips(clauses: List[Dict], clause_results: List[Dict], resume: str) -> Optional[str]:
    """根据 JD 覆盖情况，生成面试准备提示"""
    covered_cats = set()
    for cs in clause_results:
        for kw in cs.get("matched", []):
            cat = _SG.get_cat(kw)
            if cat:
                covered_cats.add(cat)

    tips = []
    for cat in covered_cats:
        if cat == "database":
            tips.append("复习：索引原理、事务隔离级别、慢查询优化、分库分表方案。")
        elif cat == "devops":
            tips.append("复习：Docker 网络原理、K8s Pod 调度机制、CICD 流水线设计。")
        elif cat == "mq":
            tips.append("复习：Kafka 分区策略、消费者组重平衡、消息丢失/重复处理。")
        elif cat == "ml":
            tips.append("复习：模型选型依据、特征工程思路、线上效果评估（A/B测试）。")
        elif cat == "python_web":
            tips.append("复习：FastAPI vs Django 区别、异步编程、ORM 性能调优。")
        elif cat == "backend_lang":
            tips.append("复习：Goroutine 调度、GC 调优、内存泄漏排查、并发安全。")

    if tips:
        return tips[0]  # 每轮只给一条，避免信息过载
    return None


# ═══════════════════════════════════════════════════════════
#  STAR 建议生成（增强版）
# ═══════════════════════════════════════════════════════════

def _generate_suggestions(
    clauses: List[Dict],
    clause_results: List[Dict],
    exp: Dict,
    missing_core: List[str],
    resume_sents: List[str],
    resume: str,
    sem_score: float,
) -> List[Dict]:
    suggestions = []
    skill_clauses = [c for c in clause_results if c["type"] == "skill"]

    # ── ① 核心技能缺失（一票否决触发）────────────────────
    weak_required = [c for c in skill_clauses if c["score"] < 0.3 and c["required"]]
    if weak_required:
        c = weak_required[0]
        miss = c["missing"][:2] if c["missing"] else missing_core[:1]
        if miss:
            suggestions.append({
                "clause": c["text"],
                "tag": "⚠️ 核心技能缺失",
                "skill": miss[0],
                "severity": "high",
                "star": {
                    "S": f"JD 明确要求「{miss[0]}」，但简历中未体现。",
                    "T": f"将「{miss[0]}」相关经历前置到简历显眼位置。",
                    "A": f"若缺少直接经历：在技能栏写明；在项目描述中用「使用{miss[0]}实现XX功能」方式补充。",
                    "R": "ATS 系统筛简历时，缺失核心关键字直接淘汰，绝无机会进入人工环节。",
                }
            })

    # ── ② 语义匹配低 → JD 与简历差距大 ─────────────────
    if sem_score < 0.25:
        suggestions.append({
            "clause": f"简历内容与 JD 整体语义差距较大（{sem_score:.0%} 相似度）",
            "tag": "🔄 内容方向偏差",
            "skill": "简历定位",
            "severity": "medium",
            "star": {
                "S": f"简历的措辞与 JD 岗位描述语言相差较大，HR 第一眼可能认为「不匹配」。",
                "T": "调整简历措辞，向 JD 关键词靠拢。",
                "A": "将简历中的项目描述改写：用 JD 的词汇描述相同经历（如「写接口」→「提供 RPC 服务」）。",
                "R": "即使是同一段经历，不同的描述方式会让 ATS 给出截然不同的匹配分。",
            }
        })

    # ── ③ 无量化指标 ──────────────────────────────────
    if exp["quantified_count"] == 0:
        vague = next((s for s in resume_sents
                      if any(w in s for w in ['负责', '完成', '开发', '优化', '提升'])
                      and not re.search(r'\d+', s)), None)
        if vague:
            suggestions.append({
                "clause": f"「{vague[:20]}...」全为定性描述，无量化指标",
                "tag": "📊 量化不足",
                "skill": "STAR 结果",
                "severity": "medium",
                "star": {
                    "S": f"项目描述停留在定性层面：{vague[:30]}...",
                    "T": "每段经历补充至少一个数字。",
                    "A": f"改写示例：{vague} → 在该项目中，日活10万→50万，QPS从80→500，P99延迟从200ms→40ms。",
                    "R": "量化数据是简历的「货币」，面试官凭此判断你的真实贡献规模。",
                }
            })

    # ── ④ 经历单薄 ──────────────────────────────────
    if exp["total_sections"] < 3:
        suggestions.append({
            "clause": f"简历仅 {exp['total_sections']} 段经历，内容密度不足",
            "tag": "📝 经历单薄",
            "skill": "简历完整度",
            "severity": "medium",
            "star": {
                "S": f"经历偏少（{exp['total_sections']}段），与同岗位竞争者相比内容单薄。",
                "T": "目标：简历充实到 4-6 段有实质内容的经历。",
                "A": "将课程项目、比赛、开源贡献、课外实践全部整理入简历，每段附技术栈+动作+数字结果。",
                "R": "经历丰富的简历在 ATS 通过率比内容单薄的高出 2-3 倍。",
            }
        })

    # ── ⑤ 动词能级过低 ────────────────────────────────
    avg_verb_power = exp["weighted_verbs"] / max(exp["star_indicators"], 1)
    if avg_verb_power < 0.7:
        suggestions.append({
            "clause": "简历中大量使用低级动词（了解/协助/参与）",
            "tag": "🎯 动词能级偏低",
            "skill": "STAR 行动",
            "severity": "low",
            "star": {
                "S": "简历中多段经历使用「了解」「协助」「参与」等低能量动词，面试官会认为你的参与度不高。",
                "T": "将低级动词替换为高级动词，强化你的主导角色。",
                "A": "了解→掌握；协助→配合；参与→主导；写过→实现/落地/自研。",
                "R": "动词决定面试官对你「角色定位」的第一印象——主角还是配角。",
            }
        })

    # ── ⑥ 优秀情况 ──────────────────────────────────
    if not suggestions:
        suggestions.append({
            "clause": "简历与 JD 核心要求高度匹配",
            "tag": "✨ 优秀",
            "skill": "—",
            "severity": "none",
            "star": {
                "S": "简历覆盖了岗位核心技能，有量化数据，STAR 结构完整，动词能级高。",
                "T": "保持当前质量，冲刺面试。",
                "A": "根据匹配到的技能类别，准备对应面试题。",
                "R": "这份简历在 ATS 和 HR 初筛中具有强竞争力。",
            }
        })

    return suggestions[:4]


# ═══════════════════════════════════════════════════════════
#  主匹配函数
# ═══════════════════════════════════════════════════════════

def match_score(resume: str, jd: str) -> Tuple[float, List[Dict]]:
    """
    v9 评分体系：
    - 核心技能匹配 × 55%   （仅含技能关键词的 JD 条款）
    - 语义相似度   × 20%   （Sentence-BERT Embedding）
    - 经验质量     × 25%   （经历数量+量化+动词能级）
    ─────────────────────────────────
    硬技能惩罚项：required 技能 < 0.3 → 总分 × 0.7
    """
    resume_tokens = set(_tokenize(resume))
    resume_sents = _sentences(resume)

    clauses = _extract_jd_clauses(jd)
    skill_score, clause_results, missing_core = _score_clauses(clauses, resume, resume_tokens)
    sem_score = _embedding_score(resume, jd)
    exp = _parse_experience(resume)
    yrs_req = next((min(_extract_years(c["text"])) for c in clauses if c["type"] == "years"), None)
    exp_score = _experience_score(exp, yrs_req)

    # 核心技能分：只看含技能的 JD 条款
    skill_clauses = [c for c in clause_results if c["type"] == "skill"]
    core_skill = sum(c["score"] for c in skill_clauses) / len(skill_clauses) if skill_clauses else skill_score

    # 加权总分
    final_raw = core_skill * 0.55 + sem_score * 0.20 + exp_score * 0.25

    # ── 硬技能惩罚项（Negative Scoring）─────────────────
    fail_required = any(c["score"] < 0.3 and c["required"] for c in skill_clauses)
    if fail_required:
        final_raw *= 0.7   # 一票否决：总分打七折
        penalty_flag = True
    else:
        penalty_flag = False

    score = round(final_raw * 100, 1)

    # ── bonus ───────────────────────────────────────────
    if exp["quantified_count"] >= 3 and exp["total_sections"] >= 3:
        score = min(score + 2, 99)
    if core_skill >= 0.75 and exp_score >= 0.65:
        score = min(score + 4, 99)
    if penalty_flag:
        score = max(score, 5)   # 被惩罚后最低 5 分

    score = round(score, 1)

    suggestions = _generate_suggestions(
        clauses, clause_results, exp, missing_core, resume_sents, resume, sem_score
    )

    # ── 面试提示（附加字段）─────────────────────────────
    interview_tip = _generate_interview_tips(clauses, clause_results, resume)

    return score, suggestions
