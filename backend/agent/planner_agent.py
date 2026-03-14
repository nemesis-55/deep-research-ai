"""
Planner Agent — Deep Research AI

Design pattern: Strategy — encapsulates research plan generation behind a single
public method.  Uses DeepSeek-R1 (role="planner") for structured JSON output.
Falls back to a deterministic template plan if LLM output cannot be parsed.
"""
import json
import logging
import re
from typing import List

from backend.config_loader import get
from backend.model_loader import generate_text

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_PLANNER_PROMPT = """\
You are an elite research director at a top-tier intelligence firm.
Create a DEEP, EXHAUSTIVE research plan — like a Bloomberg analyst or Gemini Deep Research.

User Query: {query}

Generate 8-12 highly specific, non-overlapping research sub-tasks that together produce
a world-class comprehensive report. Cover ALL of:
- Core facts, history, and foundational background
- Financial data, revenue, market cap, key metrics
- Recent news, announcements, developments (last 6-12 months)
- Key people: leadership, founders, executives
- Competitors, market position, industry landscape
- Future plans, roadmap, partnerships, expansions
- Expert opinions, analyst ratings, third-party assessments
- Regulatory, legal, or ethical considerations
- Industry trends and macro environment
- Specific statistics, data points, and numbers

Return ONLY valid JSON — no text before or after:
{{
  "research_plan": [
    "Highly specific task 1",
    "Highly specific task 2"
  ]
}}

Each task must be a SPECIFIC, ACTIONABLE search query. Be extremely precise."""

_FALLBACK_TASKS = [
    "Background, founding history and overview of: {query}",
    "Financial performance, revenue, valuation and key metrics for: {query}",
    "Recent news, announcements and major developments about: {query} in 2024-2025",
    "Key leadership, founders, executives and their backgrounds at: {query}",
    "Main competitors, market share and competitive positioning of: {query}",
    "Future plans, product roadmap, expansions and partnerships of: {query}",
    "Expert analyst opinions, ratings and third-party assessments of: {query}",
    "Controversies, regulatory issues, legal challenges facing: {query}",
    "Industry trends, market dynamics and macro environment affecting: {query}",
    "Key statistics, user numbers, growth metrics and data points for: {query}",
    "Technology stack, infrastructure and technical innovation of: {query}",
    "Business model, revenue streams and monetization strategy of: {query}",
]


# ── Agent ─────────────────────────────────────────────────────────────────────

class PlannerAgent:
    """Generates a structured multi-task research plan using the reasoning model."""

    def generate_plan(self, query: str) -> List[str]:
        logger.info(f"[Planner] Generating plan for: {query}")
        try:
            raw  = generate_text(_PLANNER_PROMPT.format(query=query), max_new_tokens=1024, role="planner")
            plan = self._parse(raw)
            if plan:
                logger.info(f"[Planner] {len(plan)}-task plan generated.")
                return plan
        except Exception as e:
            logger.warning(f"[Planner] Generation failed: {e}")

        logger.warning("[Planner] Using fallback plan.")
        return [t.format(query=query) for t in _FALLBACK_TASKS]

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, raw: str) -> List[str]:
        """Robustly extract a task list from raw LLM output."""
        # 1. Try JSON extraction
        try:
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                plan = data.get("research_plan", [])
                if isinstance(plan, list) and len(plan) >= 3:
                    return [str(t).strip() for t in plan if str(t).strip()]
        except json.JSONDecodeError:
            pass

        # 2. Fallback: extract numbered or bulleted lines
        lines = []
        for line in raw.splitlines():
            line = line.strip()
            m = re.match(r"^\d+[.)]\s+(.+)$", line)
            if m:
                lines.append(m.group(1).strip())
            elif line.startswith(("- ", "* ", "• ")):
                lines.append(line[2:].strip())

        return lines if len(lines) >= 3 else []
