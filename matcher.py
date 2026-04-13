"""
Resume–JD 匹配算法 v7（具体改进建议 + 原文扫描）

核心升级：
1. 扫描简历原文，定位具体句子
2. 每条建议给出「原文 → 修改后」对照
3. 不再套模板，全部基于实际检测结果生成
"""

import re
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional

# ═══════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower())
    stop = {"的", "了", "在", "是", "我", "你", "和", "与", "或", "但", "而", "于", "有", "这", "那", "个", "们", "为", "上", "下", "中", "对", "以", "到", "把", "被", "要", "会", "能", "可", "也", "很", "都", "将", "又", "再", "从", "所", "已", "应", "该", "等", "并", "for", "with", "and", "or", "the", "a", "an", "to", "of", "in", "on", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "this", "that", "it"}
    return [t for t in tokens if t not in stop and len(t) > 1]

def _extract_years(text: str) -> List[int]:
    return [int(m.group(1)) for m in re.finditer(r"(\d+)\s*年", text) if 0 <= int(m.group(1)) <= 50]

def _sentences(text: str) -> List[str]:
    """把文本拆成句子（更宽松的阈值）"""
    return [s.strip() for s in re.split(r'[，。；\n]', text) if len(s.strip()) > 4]


# ═══════════════════════════════════════════════════════════
#  技能知识图谱
# ═══════════════════════════════════════════════════════════

class SkillGraph:
    CATEGORIES = {
        "backend_lang": {"python", "java", "golang", "go", "c++", "c#", "php", "ruby", "rust"},
        "frontend": {"react", "vue", "angular", "nextjs", "nuxt"},
        "python_web": {"django", "flask", "fastapi", "tornado"},
        "java_web": {"spring", "springboot", "mybatis"},
        "database": {"mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch", "sqlite"},
        "cache": {"redis", "memcached"},
        "mq": {"kafka", "rabbitmq", "rocketmq"},
        "devops": {"docker", "kubernetes", "k8s", "jenkins", "gitlab", "github"},
        "ml": {"pytorch", "tensorflow", "sklearn", "xgboost", "keras"},
        "robotics": {"ros", "slam", "opencv", "pcl", "gazebo", "matlab", "simulink"},
        "transport": {"v2x", "交通流", "信号控制", "车路协同"},
    }

    ALIASES = {
        "golang": "go", "go": "go",
        "js": "javascript", "javascript": "javascript", "es6": "javascript",
        "ts": "typescript", "typescript": "typescript",
        "py": "python", "python3": "python",
        "postgres": "postgresql",
        "mongo": "mongodb",
        "k8s": "kubernetes", "kubernetes": "kubernetes",
        "es": "elasticsearch",
        "springboot": "spring", "spring": "spring",
        "pytorch": "pytorch", "tf": "tensorflow",
    }

    # 同类技能内部相似度
    CAT_SIM = {
        "python_web": 0.85, "java_web": 0.85, "database": 0.75,
        "frontend": 0.8, "ml": 0.8, "devops": 0.8,
        "backend_lang": 0.5,
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
            return self.CAT_SIM.get(ca, 0.8)
        return 0.0

    def find_in_text(self, skill: str, text: str) -> Optional[str]:
        """在文本中找包含某技能的句子"""
        skill = self.normalize(skill)
        for sent in _sentences(text):
            if skill in sent:
                return sent
        return None

_SG = SkillGraph()


# ═══════════════════════════════════════════════════════════
#  模糊词检测 & 量化建议库
# ═══════════════════════════════════════════════════════════

# 模糊词 → 可能的量化方向
VAGUE_PAIRS = [
    # (模糊词, 性能方向, 业务方向)
    ("优化了", "响应时间", "吞吐量"),
    ("提升了", "性能提升%", "转化率提升%"),
    ("改进了", "处理速度", "准确率"),
    ("完善了", "功能完整性", "用户满意度"),
    ("解决了", "问题数", "稳定性"),
    ("开发了", "功能模块数", "服务用户数"),
    ("负责了", "项目规模", "团队人数"),
    ("维护了", "系统稳定性", "SLA"),
    ("搭建了", "架构规模", "覆盖用户"),
    ("完成了", "交付物数量", "收益指标"),
    ("使用了", "技术方案", "业务场景"),
]

# 常见量化单位组合
QUANT_EXAMPLES = [
    "响应时间从{old}降至{new}，提升{ratio}倍",
    "QPS从{old}提升至{new}",
    "日活{DAU}万，支持{count}并发用户",
    "数据处理量达{vol}万条/天",
    "覆盖{users}万用户，稳定性{SLA}",
    "准确率从{old}%提升至{new}%",
]


# ═══════════════════════════════════════════════════════════
#  岗位画像
# ═══════════════════════════════════════════════════════════

def _detect_role(jd: str) -> Tuple[str, Dict]:
    """检测JD类型，返回角色和权重"""
    jd_l = jd.lower()
    profiles = {
        "dev":       (["开发", "工程师", "后端", "前端", "api", "系统"], 0.45, 0.25, 0.20, 0.10),
        "pm":        (["产品", "经理", "需求", "调研", "用户", "mvp"],  0.25, 0.40, 0.15, 0.20),
        "ai_pm":     (["ai", "算法", "模型", "机器学习", "产品"],       0.30, 0.35, 0.15, 0.20),
        "research":  (["算法", "研究", "论文", "模型", "优化", "顶会"], 0.35, 0.30, 0.15, 0.20),
    }
    best_role, best_score = "dev", 0
    best_weights = profiles["dev"]
    for role, (keywords, ws, wse, wy, wi) in profiles.items():
        score = sum(2 for kw in keywords if kw in jd_l)
        if score > best_score:
            best_role, best_score = role, score
            best_weights = (ws, wse, wy, wi)
    return best_role, {
        "skill": best_weights[0], "semantic": best_weights[1],
        "years": best_weights[2], "intensity": best_weights[3]
    }


# ═══════════════════════════════════════════════════════════
#  语义相似度（降级版，不依赖外部模型）
# ═══════════════════════════════════════════════════════════

def _semantic_score(resume: str, jd: str) -> float:
    """基于词集重叠的语义评分"""
    r_tokens = set(_tokenize(resume))
    j_tokens = set(_tokenize(jd))
    if not j_tokens:
        return 0.5
    # 扩展同义词
    def expand(t):
        return {_SG.normalize(t)}
    r_exp = set()
    for t in r_tokens:
        r_exp.update(expand(t))
    j_exp = set()
    for t in j_tokens:
        j_exp.update(expand(t))
    overlap = len(r_exp & j_exp)
    base = overlap / max(len(j_exp), 3)
    return min(base * 1.3, 1.0)


# ═══════════════════════════════════════════════════════════
#  实体密度（项目强度）
# ═══════════════════════════════════════════════════════════

def _intensity_score(text: str) -> float:
    sents = _sentences(text)
    if not sents:
        return 0.3
    total = 0.0
    for s in sents:
        q = len(re.findall(r'\d+', s))
        t = sum(1 for cat in _SG.CATEGORIES.values() for w in cat if w in s.lower())
        v = sum(1 for w in ['提升','降低','优化','搭建','开发','维护','管理'] if w in s)
        if q == 0 and t == 0:
            continue
        score = min((q * 0.2 + t * 0.3 + v * 0.15) / (len(s)/50), 1.0)
        total += score
    return min(total / max(len(sents), 1) * 1.5, 1.0)


# ═══════════════════════════════════════════════════════════
#  年限匹配 + 过匹配检测
# ═══════════════════════════════════════════════════════════

def _years_check(resume: str, jd: str, role: str) -> Tuple[float, Optional[str]]:
    r_yrs = _extract_years(resume)
    j_yrs = _extract_years(jd)
    if not j_yrs:
        return 1.0, None
    if not r_yrs:
        return 0.6, None
    max_r = max(r_yrs)
    min_j = min(j_yrs)
    if max_r >= min_j:
        # 初级岗申请但经验过多
        if min_j <= 1 and max_r >= 3:
            return 0.95, "该岗位偏初级，你的经验较丰富，建议在简历中突出「学习能力」而非「资历」。"
        return 1.0, None
    diff = min_j - max_r
    return max(0.5, 1.0 - diff * 0.15), None


# ═══════════════════════════════════════════════════════════
#  核心：生成具体改进建议
# ═══════════════════════════════════════════════════════════

def _generate_suggestions(
    resume: str,
    jd: str,
    role: str,
    skill_matches: List[Tuple],
    missing_skills: List[str],
    years_score: float,
    years_warning: Optional[str],
    years_gap: Optional[int],
    years_required: Optional[int],
) -> List[Dict]:
    """基于实际检测，生成具体的改进建议"""
    suggestions = []
    sents = _sentences(resume)

    # ── 检测①：模糊描述（无量化指标）────────────────────
    vague_sents = []
    result_verbs = ['提升', '优化', '改进', '完善', '解决', '开发', '负责', '完成', '维护', '搭建', '使用', '降低', '增加', '减少']
    for sent in sents:
        has_quant = bool(re.search(r'\d+\s*(万|亿|千|%|％|ms|QPS|DAU|用户|倍|条|次|台|个)', sent))
        if has_quant:
            continue
        has_result_verb = any(w in sent for w in result_verbs)
        if has_result_verb and len(sent) > 5:
            vague_sents.append(sent)

    # 排序：优化/提升类句子优先
    priority = ['优化', '提升', '改进', '降低', '增加', '减少']
    vague_sents.sort(key=lambda s: -sum(1 for w in priority if w in s))

    if vague_sents:
        # 取第一个模糊句作为示例
        example = vague_sents[0]
        # 根据上下文猜量化方向
        if any(w in example for w in ['优化','改进','提升']):
            rewrite = example + '，响应时间从800ms降至120ms，性能提升5倍'
        elif any(w in example for w in ['开发','搭建','负责']):
            rewrite = example + '，日活10万，支持500并发'
        elif any(w in example for w in ['维护','管理']):
            rewrite = example + '，SLA达99.9%，可用性提升40%'
        else:
            rewrite = example + '，QPS从100提升至800，处理量提升8倍'

        suggestions.append({
            "clause": f"「{example[:20]}...」缺少量化指标",
            "tag": "量化不足",
            "skill": "STAR 结果",
            "example_from": example,
            "example_to": rewrite,
            "star": {
                "S": f"简历中有描述偏定性的句子：{example[:30]}...",
                "T": "为每个项目补充至少一个量化指标。",
                "A": f"原文：{example}\n改写：{rewrite}",
                "R": "量化数据是面试通行证，HR和ATS系统都会高看你。",
            }
        })

    # ── 检测②：同类技能未关联（JD要A，简历有B，A和B同类）────
    jd_tokens = set(_tokenize(jd))
    resume_tokens = set(_tokenize(resume))
    related_done = False
    for jd_skill in jd_tokens:
        if related_done:
            break
        cat = _SG.get_cat(jd_skill)
        if not cat:
            continue
        for res_skill in resume_tokens:
            if res_skill == jd_skill:
                continue
            res_cat = _SG.get_cat(res_skill)
            if res_cat == cat and _SG.similarity(res_skill, jd_skill) >= 0.5:
                # 找到了！看看简历中有没有提到 res_skill 的句子
                related_sent = _SG.find_in_text(res_skill, resume)
                cat_name = {"python_web":"Python Web框架","java_web":"Java框架",
                           "database":"数据库","ml":"ML框架"}.get(cat, cat)

                suggestions.append({
                    "clause": f"JD 要「{jd_skill}」，你有「{res_skill}」（{cat_name}同类）",
                    "tag": "关联技能",
                    "skill": jd_skill,
                    "star": {
                        "S": f"「{res_skill}」和「{jd_skill}」同属{cat_name}，你已有扎实基础。",
                        "T": "在简历中关联两者，展示技术迁移能力。",
                        "A": f"{'在「' + related_sent[:15] + '...」后补充' if related_sent else '在项目描述中'}：同时对{jd_skill}有实践，学习成本低，可快速上手。",
                        "R": "招聘方看到你有同类技术基础，会降低学习成本预期。",
                    }
                })
                related_done = True
                break

    # ── 检测③：纯缺失技能（简历完全没有）──────────────────
    core_missing = [s for s in missing_skills if _SG.get_cat(s)]
    shown_jd_skills = {s.get("jd_skill") for s in suggestions}
    for skill in core_missing[:2]:
        if skill in shown_jd_skills:
            continue
        # 找简历中最相关的已有技能作为锚点
        best_anchor = None
        best_sim = 0
        for res_skill in resume_tokens:
            sim = _SG.similarity(res_skill, skill)
            if sim > best_sim:
                best_sim = sim
                best_anchor = res_skill

        anchor_sent = _SG.find_in_text(best_anchor, resume) if best_anchor else None
        suggestions.append({
            "clause": f"简历未提及「{skill}」，JD 明确要求",
            "tag": "技能缺失",
            "skill": skill,
            "star": {
                "S": f"「{skill}」是JD明确要求的技术栈，但简历中完全没有出现。",
                "T": "在简历中找到可关联的项目，或补充学习经历。",
                "A": f"{'在「' + anchor_sent[:15] + '...」的描述中' if anchor_sent else '在项目描述中'}加入：使用{skill}完成XX功能；或写「了解{skill}核心原理，正在系统学习」。",
                "R": "关键词出现在简历中，ATS系统才能识别并推送给HR。",
            }
        })

    # ── 检测④：年限不达标 ──────────────────────────────
    if years_score < 0.8 and years_gap and years_required:
        suggestions.append({
            "clause": f"JD 要求 {years_required} 年，简历约 {years_required - years_gap} 年",
            "tag": "年限不足",
            "skill": "工作年限",
            "star": {
                "S": f"简历年限与JD要求存在 {years_gap} 年差距。",
                "T": "明确写出所有相关工作年限（含实习、兼职、比赛）。",
                "A": "在简历顶部或个人简介写：「X年XX领域经验」；若总年限达标，将年限写得更显眼。",
                "R": "ATS系统据此判断是否进入下一轮，不要让HR自己算。",
            }
        })

    # ── 检测⑤：过匹配 ──────────────────────────────────
    if years_warning:
        suggestions.append({
            "clause": "你的经验较丰富，申请的是初级岗位",
            "tag": "策略建议",
            "skill": "简历侧重点",
            "star": {
                "S": years_warning,
                "T": "让招聘方看到你的适配度，而非单纯的资历。",
                "A": "在简历中弱化年限数字，突出「快速学习」「可塑性强」「转岗动机」等标签。",
                "R": "降低「overqualified」顾虑，增加面试机会。",
            }
        })

    # ── 优秀：没有明显问题 ─────────────────────────────
    if not suggestions:
        suggestions.append({
            "clause": "简历与 JD 核心要求匹配良好",
            "tag": "优秀",
            "skill": "—",
            "star": {
                "S": f"简历覆盖了{role}岗位所需的核心技能，无明显缺失项。",
                "T": "可继续打磨细节，冲击更高分。",
                "A": "检查是否有更多量化结果可补充；确认每段经历都有STAR结构。",
                "R": "保持优化，面试邀约率会持续提升。",
            }
        })

    return suggestions[:4]


# ═══════════════════════════════════════════════════════════
#  主匹配函数
# ═══════════════════════════════════════════════════════════

def match_score(resume: str, jd: str) -> Tuple[float, List[Dict]]:
    """
    评分体系 v7：
    - 技能匹配 × 动态权重（岗位类型决定）
    - 语义相似度 × 动态权重
    - 年限匹配 × 动态权重
    - 实体密度 × 动态权重
    基础分 60，加表现分 40
    """
    role, w = _detect_role(jd)
    resume_tokens = set(_tokenize(resume))
    jd_tokens = set(_tokenize(jd))

    # 技能匹配
    matched_score_sum = 0.0
    skill_matches = []
    missing = []
    for js in jd_tokens:
        best = 0.0
        best_pair = (None, None)
        for rs in resume_tokens:
            sim = _SG.similarity(rs, js)
            if sim > best:
                best = sim
                best_pair = (rs, js)
        if best >= 0.5:
            matched_score_sum += best
            skill_matches.append((best_pair[0], best_pair[1], best))
        else:
            missing.append(js)

    skill_score = min(matched_score_sum / max(len(jd_tokens), 3) * 1.3, 1.0)

    # 语义 & 密度
    sem_score = _semantic_score(resume, jd)
    int_score = _intensity_score(resume)

    # 年限
    yrs_score, yrs_warn = _years_check(resume, jd, role)
    r_yrs = _extract_years(resume)
    j_yrs = _extract_years(jd)
    yrs_gap = None
    if j_yrs and r_yrs and yrs_score < 0.8:
        yrs_gap = min(j_yrs) - max(r_yrs)
        if yrs_gap < 0:
            yrs_gap = None

    # 加权
    final = skill_score * w["skill"] + sem_score * w["semantic"] + yrs_score * w["years"] + int_score * w["intensity"]
    score = 60 + final * 40

    # bonus
    core_matches = [m for m in skill_matches if m[2] >= 0.8]
    if len(core_matches) >= 4:
        score = min(98, score + 5)
    elif len(core_matches) >= 2:
        score = min(95, score + 3)

    score = round(score, 1)

    suggestions = _generate_suggestions(
        resume, jd, role,
        skill_matches, missing,
        yrs_score, yrs_warn,
        yrs_gap, min(j_yrs) if j_yrs else None,
    )

    return score, suggestions
