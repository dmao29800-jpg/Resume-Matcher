import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from matcher import match_score, _parse_jd_weighted, _skill_semantic_match

resume = """
张明，3年Python后端开发经验，熟悉Django框架。
使用MySQL和Redis，负责用户系统设计与API开发。
有分布式系统和高并发场景经验。
"""

jd = """
岗位要求：
1. 精通Python后端开发
2. 熟悉Django或FastAPI框架
3. 熟练使用MySQL数据库
4. 有Redis缓存使用经验
5. 本科及以上学历
6. 了解Golang优先
"""

# Debug JD parsing
parsed = _parse_jd_weighted(jd)
print("=== JD Weighted Skills ===")
for s, w, l, ln in parsed["weighted_skills"]:
    print(f"  skill={s}, weight={w:.2f}, level={l}, line={ln[:30]}")
print(f"  years={parsed['years']}")
print(f"  lines count={len(parsed['lines'])}")

# Debug skill matching
skill_score, details = _skill_semantic_match(resume, parsed["weighted_skills"])
print(f"\n=== Skill Match Score: {skill_score:.3f} ===")
for d in details:
    print(f"  {d['skill']}: matched={d['matched']}, method={d['method']}, weight={d['weight']:.2f}")

# Full score
score, suggestions = match_score(resume, jd)
print(f"\n=== Final Score: {score} ===")
for i, s in enumerate(suggestions, 1):
    tag = s.get('tag', '')
    clause = s.get('clause', '')[:50]
    print(f"  Suggestion {i}: [{tag}] {clause}")
