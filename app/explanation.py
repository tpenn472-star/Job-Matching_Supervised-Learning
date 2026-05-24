import json
import re
from typing import Any, Dict, List, Optional


def _strip_json_fence(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


class GenAIExplanationService:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        self.load_error = None

        if not api_key:
            self.load_error = "GENAI_API_KEY is not configured."
            return

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except Exception as exc:
            self.load_error = str(exc)
            self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def generate_explanation(
        self,
        resume_text: str,
        mode: str,
        target_role: Optional[str],
        user_friendly_score: Dict[str, Any],
        top_items: List[Dict[str, Any]],
        profile_insights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.model is None:
            return {
                "enabled": False,
                "generated": False,
                "error": self.load_error,
                "summary": "",
                "strengths": [],
                "gaps": [],
                "recommendations": [],
                "feedback": "",
            }

        compact_items = []
        for item in top_items[:5]:
            evidence = item.get("evidence", {}) or {}
            compact_items.append({
                "job_role": item.get("job_role"),
                "fit_score": item.get("fit_score"),
                "structured_score": item.get("structured_score"),
                "ranking_score": item.get("ranking_score"),
                "matched_skills": evidence.get("matched_skills", []),
                "missing_skills": evidence.get("missing_skills", []),
                "matched_tools": evidence.get("matched_tools", []),
                "missing_tools": evidence.get("missing_tools", []),
                "matched_domains": evidence.get("matched_domains", []),
                "missing_domains": evidence.get("missing_domains", []),
            })

        accepted_profile = {}
        if profile_insights:
            accepted_profile = profile_insights.get("accepted_for_scoring", {}) or {}

        prompt = f"""
You are an AI career matching assistant.

Generate a concise explanation for a job matching result.
Do not change the scores. Use the provided scores and evidence only.
Return ONLY valid JSON with this exact schema:

{{
  "summary": "string",
  "strengths": ["string"],
  "gaps": ["string"],
  "recommendations": ["string"],
  "feedback": "string"
}}

Context:
- Mode: {mode}
- Target role or role hint: {target_role}
- User-friendly score:
{json.dumps(user_friendly_score, ensure_ascii=False, indent=2)}

Top matching jobs:
{json.dumps(compact_items, ensure_ascii=False, indent=2)}

Accepted profile signals:
{json.dumps(accepted_profile, ensure_ascii=False, indent=2)}

Rules:
- Use English language.
- Be honest and grounded in the evidence.
- If score is moderate or low, explain the gap clearly.
- Do not mention NER, taxonomy, internal model, or implementation details.
- Do not invent skills that are not shown in the evidence.
- Keep the summary and feedback concise.
"""

        try:
            response = self.model.generate_content(prompt)
            text = _strip_json_fence(getattr(response, "text", "") or "")
            parsed = json.loads(text)

            return {
                "enabled": True,
                "generated": True,
                "error": None,
                "summary": str(parsed.get("summary", "")),
                "strengths": list(parsed.get("strengths", [])),
                "gaps": list(parsed.get("gaps", [])),
                "recommendations": list(parsed.get("recommendations", [])),
                "feedback": str(parsed.get("feedback", "")),
            }

        except Exception as exc:
            return {
                "enabled": True,
                "generated": False,
                "error": str(exc),
                "summary": "",
                "strengths": [],
                "gaps": [],
                "recommendations": [],
                "feedback": "",
            }