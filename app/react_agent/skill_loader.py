"""
能力 B：离线技能（SKILL.md）加载与在线注入。

匹配策略（当前版本）：关键词包含匹配。
  - 解析 SKILL.md 的“## 触发条件”段，按 /、，、, 、；等分隔为关键词
  - query 命中关键词数量最多者胜出；最多注入 1 个技能
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.conf.experience_config import experience_config
from app.core.logger import logger
from app.utils.path_util import PROJECT_ROOT

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)


def _skill_dir() -> Path:
    raw = (experience_config.skill_dir or "./skills").strip()
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _section(text: str, title: str) -> str:
    """提取指定二级标题下的内容（到下一个 ## 或文末）"""
    pattern = re.compile(rf"^##\s+{re.escape(title)}\s*$(.*?)(?=^##\s+|\Z)", re.M | re.S)
    m = pattern.search(text)
    return (m.group(1).strip() if m else "")


def parse_skill(text: str, path: Path) -> Dict[str, Any]:
    """解析 SKILL.md 为 {name, trigger, content, keywords, path}"""
    name_match = re.search(r"^#\s+Skill:\s*(.+)$", text, re.M)
    if not name_match:
        name_match = re.search(r"^#\s+(.+)$", text, re.M)
    name = (name_match.group(1).strip() if name_match else path.stem)
    trigger = _section(text, "触发条件")
    keywords = [k.strip() for k in re.split(r"[/、，,；;\s]+", trigger) if k.strip()]
    # 额外抽取错误码（如 E3、A1B2）
    keywords += re.findall(r"[A-Za-z]\d{1,4}", trigger)
    keywords = list(dict.fromkeys(keywords))
    return {
        "name": name,
        "trigger": trigger,
        "content": text.strip(),
        "keywords": keywords,
        "path": str(path),
    }


def load_skills() -> List[Dict[str, Any]]:
    """加载 skills 目录下的全部 SKILL.md；目录不存在返回空列表"""
    directory = _skill_dir()
    if not directory.exists():
        return []
    skills = []
    for path in sorted(directory.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
            skills.append(parse_skill(text, path))
        except Exception as e:
            logger.warning(f"[SkillLoader] 读取技能失败：{path}，{e}")
    return skills


def select_skill(query: str) -> Optional[Dict[str, Any]]:
    """按关键词命中数量选择 1 个技能；未命中返回 None"""
    if not experience_config.skill_injection_enabled:
        return None
    query_lower = (query or "").lower()
    best = None
    best_score = 0
    for skill in load_skills():
        score = sum(1 for kw in skill["keywords"] if kw and kw.lower() in query_lower)
        if score > best_score:
            best, best_score = skill, score
    if best:
        logger.info(f"[SkillLoader] 命中技能：{best['name']}（关键词命中 {best_score} 个）")
    return best


def build_skill_block(skill: Optional[Dict[str, Any]]) -> str:
    """把技能渲染为【可用技能】块"""
    if not skill:
        return ""
    return (
        "【可用技能】（结构化沉淀，优先参考）\n\n"
        + skill.get("content", "").strip()
    )
