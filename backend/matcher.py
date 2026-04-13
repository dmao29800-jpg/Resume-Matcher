"""
Resume–JD 匹配算法 v9.1
核心改进（相比 v9）：
1. Domain Mismatch 强惩罚（核心技能覆盖率 < 40% → 经验分近乎清零）
2. 自适应权重分配（技能覆盖越高 → 技能分权重越高；覆盖低时经验分被压制）
3. TF-IDF + 技能覆盖率 联合语义匹配（轻量，无外部依赖）
4. BERT Embedding 作为可选升级（本地/云端有 GPU 时启用）
5. 动词能级权重（高级动词 ×1.2，低级动词 ×0.5）
6. 硬技能惩罚项（required 技能 < 0.3 → 总分 × 0.7）
"""

import re
import os
from typing import List, Dict, Tuple, Optional

# BERT 模型路径（Railway 环境预设，或本地调试时注释掉 USE_BERT=False）
USE_BERT = os.environ.get("USE_BERT", "false").lower() in ("true", "1", "yes")

# ═══════════════════════════════════════════════════════════
#  语义相似度（默认 TF-IDF + 覆盖率；USE_BERT=true 时启用 BERT）
# ═══════════════════════════════════════════════════════════

_EMBEDDER = None

def _get_embedder():
    """仅在 USE_BERT=true 时才尝试加载 BERT，避免冷启动阻塞"""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    if not USE_BERT:
        _EMBEDDER = "disabled"
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        _EMBEDDER = "fallback"
    return _EMBEDDER


def _semantic_score(resume: str, jd: str, clauses: List[Dict],
                    clause_results: List[Dict]) -> float:
    """
    语义匹配得分 = 0.7×TF-IDF + 0.3×技能覆盖率辅助修正
    - 如果 USE_BERT=true 且 BERT 可用：BERT 接管，TF-IDF 降为 0.3 权重
    - 默认：纯 TF-IDF，轻量快速，无外部依赖
    """
    # TF-IDF 基础分（总是计算）
    tfidf = _tfidf_cosine(resume, jd)

    # 技能覆盖率辅助（对 JD 条款的语义吻合度做二次修正）
    skill_cls = [c for c in clause_results if c["type"] == "skill"]
    if skill_cls:
        cov = sum(1 for c in skill_cls if c["score"] >= 0.4) / len(skill_cls)
        skill_helper = cov  # 0~1 的覆盖率映射到 [0,1]
    else:
        skill_helper = tfidf  # 无技能条款时跟随 TF-IDF

    model = _get_embedder()

    # BERT 路径（仅 Railway 预装 GPU 环境启用）
    if model not in ("disabled", "fallback", None):
        try:
            jd_clauses_text = [c.strip() for c in re.split(r'[,，\n；;]', jd)
                               if len(c.strip()) > 6]
            resume_sents = [s.strip() for s in re.split(r'[，。；；\n]', resume)
                            if len(s.strip()) > 6]
            if jd_clauses_text and resume_sents:
                emb_jd = model.encode(jd_clauses_text)
                emb_rs = model.encode(resume_sents)
                # cosine sim: normalized dot product
                def _n(x):
                    return x / (x**2).sum(axis=1, keepdims=True)**0.5
                sims = (_n(emb_jd) @ _n(emb_rs).T).max(axis=1)
                bert_score = float(sims.mean())
                # BERT 0.7 + TF-IDF 0.2 + 技能覆盖率 0.1
                return 0.70 * bert_score + 0.20 * tfidf + 0.10 * skill_helper
        except Exception:
            pass  # BERT 推理失败，降级到纯 TF-IDF

    # 默认路径：TF-IDF 主导，技能覆盖率辅助修正
    # JD 中无技术条款（如纯 PM）时，更多依赖 TF-IDF
    if skill_cls:
        # 技术导向 JD：TF-IDF 0.7 + 覆盖率 0.3
        return 0.70 * tfidf + 0.30 * skill_helper
    else:
        # 非技术导向 JD（PM/运营等）：TF-IDF 为主，0.9 权重
        return 0.90 * tfidf + 0.10 * skill_helper


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
    """
    提取 JD 条款，支持多种分隔符：
    - 中文编号：①②③④⑤ / 1、2、3、 / 1. 2. 3.
    - 英文编号：1. 2. 3. / (1) (2) (3)
    - 标点分隔：,，\n；;
    """
    # 多级分隔：先按段落/编号拆，再按标点拆
    raw_lines = re.split(
        r'(?:(?<=[a-zA-Z0-9])[\n]+|(?<=[^a-zA-Z0-9\n])[\n]{1,2}|(?<=[)）])[\n]+)',
        jd
    )
    # 进一步按各种编号格式拆分
    split_pattern = r'[(（]?\d{1,2}[)）]?\s*[,，、；;]?\s*'
    all_lines = []
    for line in raw_lines:
        sub = re.split(split_pattern, line)
        all_lines.extend([s.strip() for s in sub if s.strip()])
    # 兜底：再按标点拆一遍还没拆干净的
    final_lines = []
    for line in all_lines:
        parts = re.split(r'[,，；;\n]', line)
        final_lines.extend([p.strip() for p in parts if p.strip()])

    clauses = []
    seen = set()
    for line in final_lines:
        line = line.strip()
        if len(line) < 5:
            continue
        # 去重（同义行）
        key = re.sub(r'\s+', '', line)[:30]
        if key in seen:
            continue
        seen.add(key)

        tokens = set(_tokenize(line))
        if not tokens:
            continue

        is_years = bool(re.search(r'\d+\s*年', line))
        is_required = any(kw in line for kw in ['必须', '熟练掌握', '精通', '扎实的', '有经验', '优先', '要求', '至少', '博士', '硕士'])

        # 软技能关键词
        soft_words = ['沟通', '表达', '协作', '团队', '协调', '学习', '逻辑', '责任心',
                      '抗压', '开朗', '感染', '职业规划', '学生社团', '敏捷', 'Scrum',
                      '领导力', '主动性', '适应性']
        is_soft = any(w in line.lower() for w in soft_words)

        # NLP/AI 专业关键词（特殊识别，不依赖 CATEGORIES）
        nlp_keywords = {
            'nlp', 'natural language', '自然语言', '文本分类', '语义理解',
            '知识图谱', '情感分析', '对话系统', '对话', '文本生成',
            '深度学习', '机器学习', 'ml', 'ai', '人工智能',
            'tensorflow', 'pytorch', 'caffe', 'mxnet', 'keras',
            'bert', 'gpt', 'transformer', 'llm', '大模型', '大语言',
            '信号处理', '模式识别', '图像识别', 'cv', 'computer vision',
            '强化学习', 'reinforcement', '推荐系统', '推荐算法',
        }
        found_nlp = [tok for tok in tokens if tok.lower() in nlp_keywords]

        # 技术栈关键词（复用 SkillGraph）
        found_skills = found_nlp[:]
        for tok in tokens:
            if tok in seen:
                continue
            cat = _SG.get_cat(tok)
            if cat:
                found_skills.append(tok)

        # 兜底：中文技术动词/名词检测（扩展识别）
        tech_verb_nouns = {
            'python', 'java', 'go', 'golang', 'c++', 'c#', 'php', 'ruby', 'rust',
            'docker', 'kubernetes', 'k8s', 'redis', 'mysql', 'postgres', 'mongodb',
            'elasticsearch', 'nginx', 'jenkins', 'ci/cd', 'cicd',
            '微服务', '高并发', '架构', '重构', '优化', '调优', '分布式',
            '中间件', '消息队列', '缓存', '数据库', '存储', '搜索',
            '后端', '前端', '全栈', '客户端', '服务端',
            'tensorflow', 'pytorch', 'caffe', 'mxnet', 'keras', 'xgboost',
            'fastapi', 'django', 'flask', 'spring', 'springboot', 'springboot',
            'tornado', 'gin', 'beego', 'echo',
            'kafka', 'rabbitmq', 'rocketmq', 'activemq', 'pulsar',
            'vue', 'react', 'angular', '小程序', 'uniapp',
            '算法', '模型', '训练', '推理', '部署', '上线',
            'linux', 'unix', 'windows', 'shell', 'bash', 'awk',
            'tcp', 'udp', 'http', 'https', 'grpc', 'thrift', 'websocket',
            'oauth', 'jwt', 'ssl', 'tls', '加密', '安全',
            '测试', '单元测试', '集成测试', '自动化测试',
            'devops', 'sre', '监控', '日志', '链路追踪',
        }
        found_verbs = [tok for tok in tokens if tok.lower() in tech_verb_nouns]
        found_skills.extend([v for v in found_verbs if v not in found_skills])

        # 加分项/注意事项 降权
        is_bonus = any(w in line for w in ['加分', '优先', '注意', '有则', '优先考虑'])

        if found_skills:
            ctype = "nlp_skill" if found_nlp else "skill"
        elif is_years:
            ctype = "years"
        elif is_soft:
            ctype = "soft"
        else:
            ctype = "general"

        clauses.append({
            "text": line,
            "type": ctype,
            "keywords": found_skills,
            "required": is_required and not is_bonus,
            "is_bonus": is_bonus,
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
) -> Tuple[float, List[Dict], List[str], float]:
    """
    返回：(加权总分, 条款结果列表, 缺失技能, Domain_Mismatch系数 0~1)
    Domain Mismatch：简历技术密度高但 JD 技术要求少 → 系数 < 1（惩罚）
    """
    if not clauses:
        return 0.5, [], [], 1.0

    # ── Domain Mismatch 检测（精准版）────────────────
    # 仅在 JD 明确是软技能/PM 导向时惩罚（soft 条款 > 60%）
    # 不惩罚有大量 general 条款但整体偏技术的 JD（如"微服务架构"算技术要求）
    resume_tech_tokens = set()
    for tok in resume_tokens:
        if _SG.get_cat(tok) or tok in {
            'python', 'java', 'go', 'golang', 'c++', 'c#', 'php', 'ruby', 'rust',
            'docker', 'kubernetes', 'k8s', 'redis', 'mysql', 'postgres', 'mongodb',
            'nlp', 'ml', 'ai', 'tensorflow', 'pytorch', 'fastapi', 'django',
            'spring', 'kafka', 'rabbitmq', 'flask', 'vue', 'react', 'angular',
            '后端', '前端', '全栈', '微服务', '高并发', '分布式',
        }:
            resume_tech_tokens.add(tok)
    resume_tech_ratio = len(resume_tech_tokens) / max(len(resume_tokens), 1)

    # 仅统计 soft 条款占比（general 条款不参与 mismatch 计算）
    jd_soft = sum(1 for c in clauses if c["type"] == "soft")
    jd_total = len(clauses)
    jd_soft_ratio = jd_soft / max(jd_total, 1)

    # Domain Mismatch：JD 软技能占比 > 60% 且简历技术密集 → 强惩罚
    if resume_tech_ratio >= 0.25 and jd_soft_ratio > 0.6:
        mismatch_penalty = 0.2   # 简历是技术岗，JD 明确是软技能岗
    else:
        mismatch_penalty = 1.0  # 正常评估（general 条款不触发惩罚）

    # ── 逐条评分 ───────────────────────────────────────
    clause_scores = []
    missing_core = []

    for clause in clauses:
        ct = clause["type"]
        kws = clause["keywords"]
        required = clause["required"]
        is_bonus = clause.get("is_bonus", False)
        score = 0.0
        matched, missing = [], []

        if ct in ("skill", "nlp_skill"):
            if not kws:
                score = 0.4
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
                    # 关联技能加成
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
                score = 0.3

        elif ct == "soft":
            # 软技能：全文扫描 clause 里的关键词是否出现在简历中
            soft_map = {
                '沟通': ['沟通', '表达', '协作', '协调', '善于沟通'],
                '团队': ['团队', '合作', '配合', '协作'],
                '学习': ['学习', '自学', '研究'],
                '逻辑': ['逻辑', '分析', '思考'],
                '开朗': ['开朗', '乐观', '积极', '阳光'],
                '敏捷': ['敏捷', 'scrum', '看板', 'sprint'],
                '职业': ['职业规划', '职业发展', '长期发展'],
                '社团': ['社团', '学生组织', '干部', '学生会', '职务'],
            }
            soft_hits = 0
            active_soft_keys = [kw for kw in soft_map if kw in clause["text"]]
            for kw in active_soft_keys:
                variants = soft_map[kw]
                if any(v in resume for v in variants):
                    soft_hits += 1
            soft_score = min(soft_hits / max(len(active_soft_keys), 1), 1.0)
            if soft_score == 0:
                soft_score = 0.1  # 没有任何软技能证据，极低分
            score = soft_score

        else:  # general clause
            # 通用条款：简历技术岗 vs 通用 JD 的匹配度
            general_tech_hints = ['开发', '编程', '代码', '技术', '软件', '系统',
                                  '算法', '数据', '后台', '前端']
            general_soft_hints = ['开朗', '乐观', '抗压', '感染', '社团', '规划']
            r_lower = resume.lower()
            has_tech = any(h in r_lower for h in general_tech_hints)
            has_soft = any(h in r_lower for h in general_soft_hints)
            if has_tech and not has_soft:
                score = 0.35  # 简历技术岗，JD 通用岗 → 中低分
            elif has_soft:
                score = 0.5   # 简历有软技能，JD 有软技能要求 → 中等
            else:
                score = 0.4   # 默认

        # 权重调整
        if is_bonus:
            final_score = score * 0.6  # 加分项权重降低
        elif required and score < 0.4:
            final_score = score * 0.6  # 硬性要求不满足时大幅减分
        else:
            final_score = score

        clause_scores.append({
            "text": clause["text"][:50],
            "score": round(final_score, 2),
            "matched": matched,
            "missing": missing,
            "required": required,
            "type": ct,
        })

        if ct in ("skill", "nlp_skill") and missing:
            for m in missing:
                if m not in missing_core:
                    missing_core.append(m)

    total_w, weighted_sum = 0, 0.0
    for cs in clause_scores:
        w = 1.5 if cs["required"] else (0.6 if clause["is_bonus"] else 1.0)
        total_w += w
        weighted_sum += cs["score"] * w

    overall = weighted_sum / total_w if total_w > 0 else 0.5
    return overall, clause_scores, missing_core, mismatch_penalty


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
    v9 评分体系（修正版）：
    ─────────────────────────────────────────────────────────
    核心原则：分数 = JD 导向，简历质量只在技能覆盖后才起作用

    评分维度：
    - 核心技能匹配 × 自适应权重  （技能覆盖率决定权重占比）
    - 经验质量     × 自适应权重  （技能覆盖不足时几乎不计分）
    - 语义相似度   × 20%         （Sentence-BERT Embedding）
    ─────────────────────────────────────────────────────────
    Domain Mismatch 惩罚：核心技能覆盖率 < 40% → 强惩罚
    硬技能惩罚项：required 技能 < 0.3 → 额外 × 0.7
    """
    resume_tokens = set(_tokenize(resume))
    resume_sents = _sentences(resume)

    clauses = _extract_jd_clauses(jd)
    skill_score, clause_results, missing_core, mismatch = _score_clauses(clauses, resume, resume_tokens)
    sem_score = _semantic_score(resume, jd, clauses, clause_results)
    exp = _parse_experience(resume)
    yrs_req = next((min(_extract_years(c["text"])) for c in clauses if c["type"] == "years"), None)
    exp_score = _experience_score(exp, yrs_req)

    # ── 核心技能覆盖率（关键指标）──────────────────────
    tech_clauses = [c for c in clause_results if c["type"] in ("skill", "nlp_skill")]
    soft_clauses  = [c for c in clause_results if c["type"] == "soft"]

    if tech_clauses:
        tech_scores  = [c["score"] for c in tech_clauses]
        core_coverage = sum(1 for s in tech_scores if s >= 0.4) / len(tech_scores)
        core_skill = sum(tech_scores) / len(tech_scores)
    else:
        core_coverage = 0.0   # JD 有技术条款但简历无匹配 -> 覆盖率=0，触发硬地板
        core_skill = 0.0

    if soft_clauses:
        soft_skill = sum(c["score"] for c in soft_clauses) / len(soft_clauses)
    else:
        soft_skill = 0.5  # 无软技能条款时默认中等

    # 技术导向型 JD（tech > 50%）：技能分以技术条款为主
    # 非技术导向 JD（tech <= 50%）：技能分 = 技术×0.6 + 软技能×0.4
    jd_tech_ratio = len(tech_clauses) / max(len(clause_results), 1)
    if tech_clauses:
        if jd_tech_ratio >= 0.5:
            weighted_skill = core_skill
        elif jd_tech_ratio >= 0.3:
            weighted_skill = core_skill * 0.5 + soft_skill * 0.5
        else:
            weighted_skill = soft_skill
    else:
        weighted_skill = soft_skill

    # ── 技能覆盖率惩罚（Domain Mismatch）────────────────
    if core_coverage < 0.4:
        exp_score *= 0.1
        sem_score  *= 0.3
    elif core_coverage < 0.7:
        exp_score *= 0.5
        sem_score  *= 0.7

    # ── 技术岗硬地板（找到 final_raw 定义后追加）────────────────
    
    # ── 自适应权重分配 ────────────────────────────────
    # 技能覆盖越高 → 技能分权重越高；覆盖越低 → 经验分权重被压制
    skill_weight = max(core_coverage, 0.3)   # 最低 0.3，避免完全无技能时经验分主导
    exp_weight   = min(1.0 - skill_weight, 0.4)  # 经验分最多占 40%
    sem_weight   = 0.20
    total_weight = skill_weight + exp_weight + sem_weight

    final_raw = (weighted_skill * skill_weight
                 + exp_score * exp_weight
                 + sem_score  * sem_weight) / total_weight

    # ── Domain Mismatch 惩罚（核心修复）────────────────
    # 简历满是技术栈但 JD 无技术要求 → 直接压低分数
    final_raw *= mismatch

    # ── 技术岗硬地板（核心修复）────────────────────────
    # JD 有技术条款但简历完全没有技术匹配（core_coverage=0）
    # → 最终分数直接压在 1-5 区间，避免软技能救场
    if tech_clauses and core_coverage == 0.0:
        final_raw = min(final_raw, 0.05)

    # ── 硬技能惩罚项（Negative Scoring）────────────────
    fail_required = any(c["score"] < 0.3 and c["required"] for c in tech_clauses)
    if fail_required:
        final_raw *= 0.7
        penalty_flag = True
    else:
        penalty_flag = False

    score = round(final_raw * 100, 1)

    # ── bonus（仅在有一定技能覆盖时才生效）─────────────
    if core_coverage >= 0.5:
        if exp["quantified_count"] >= 2 and exp["total_sections"] >= 2:
            score = min(score + 2, 99)
        if core_skill >= 0.7 and exp_score >= 0.5:
            score = min(score + 3, 99)
    if penalty_flag:
        score = max(score, 5)

    score = round(score, 1)

    suggestions = _generate_suggestions(
        clauses, clause_results, exp, missing_core, resume_sents, resume, sem_score
    )

    interview_tip = _generate_interview_tips(clauses, clause_results, resume)

    return score, suggestions
