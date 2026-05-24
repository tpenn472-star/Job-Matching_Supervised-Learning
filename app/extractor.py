import html
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
    "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "you", "your",
    "or", "this", "these", "those", "their", "they", "we", "our", "into", "while", "also", "such",
    "role", "job", "candidate", "responsible", "requirements", "required", "skills", "experience",
}


def clean_text(text) -> str:
    if text is None:
        text = ""
    else:
        try:
            if pd.isna(text):
                text = ""
        except Exception:
            pass
    text = str(text).lower()
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#./()\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_role_light(role) -> str:
    return re.sub(r"\s+", " ", clean_text(role)).strip()


def tokenize_for_overlap(text) -> Set[str]:
    tokens = re.findall(r"[a-z0-9+#.]+", clean_text(text))
    return {tok for tok in tokens if len(tok) > 2 and tok not in STOPWORDS}


def pipe_to_set(value) -> Set[str]:
    if value is None:
        return set()
    try:
        if pd.isna(value):
            return set()
    except Exception:
        pass
    return {str(x).strip().lower() for x in str(value).split("|") if str(x).strip()}


def set_to_pipe(values: Iterable[str]) -> str:
    return "|".join(sorted({str(x).strip().lower() for x in values if str(x).strip()}))


def to_list(values: Iterable[str]) -> List[str]:
    return sorted({str(v).strip().lower() for v in values if str(v).strip()})


def safe_ratio(numerator, denominator) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


class TaxonomyExtractor:
    def __init__(self, taxonomy_path: Path):
        self.taxonomy = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
        self.normalization = {
            clean_text(k): clean_text(v)
            for k, v in self.taxonomy.get("normalization_dictionary", {}).items()
        }

        cv_tax = self.taxonomy.get("cv_taxonomy", {})
        job_tax = self.taxonomy.get("job_description_taxonomy", {})

        self.skills_vocab = self._clean_vocab(set(cv_tax.get("skills", [])) | set(job_tax.get("required_skills", [])))
        self.tools_vocab = self._clean_vocab(set(cv_tax.get("tools_technologies", [])) | set(job_tax.get("required_tools", [])))
        self.domain_vocab = self._clean_vocab(set(cv_tax.get("domain_keywords", [])) | set(job_tax.get("domains", [])))
        self.education_vocab = self._clean_vocab(set(cv_tax.get("education_levels", [])) | set(job_tax.get("required_education", [])))
        self.seniority_vocab = self._clean_vocab(set(cv_tax.get("seniority_levels", [])) | set(job_tax.get("seniority_levels", [])))
        self.project_vocab = self._clean_vocab(set(cv_tax.get("project_experience_terms", [])))
        self.cert_vocab = self._clean_vocab(set(cv_tax.get("certifications", [])))
        self.target_role_vocab = self._clean_vocab(
            set(cv_tax.get("target_roles", [])) | set(cv_tax.get("work_experience_titles", []))
        )

        self.skill_alias = self._build_alias_map(self.skills_vocab)
        self.tool_alias = self._build_alias_map(self.tools_vocab)
        self.domain_alias = self._build_alias_map(self.domain_vocab)
        self.education_alias = self._build_alias_map(self.education_vocab)
        self.seniority_alias = self._build_alias_map(self.seniority_vocab)
        self.project_alias = self._build_alias_map(self.project_vocab)
        self.cert_alias = self._build_alias_map(self.cert_vocab)
        self.target_role_alias = self._build_alias_map(self.target_role_vocab)

        self.skill_re = self._compile_regex(self.skill_alias)
        self.tool_re = self._compile_regex(self.tool_alias)
        self.domain_re = self._compile_regex(self.domain_alias)
        self.education_re = self._compile_regex(self.education_alias)
        self.seniority_re = self._compile_regex(self.seniority_alias)
        self.project_re = self._compile_regex(self.project_alias)
        self.cert_re = self._compile_regex(self.cert_alias)
        self.target_role_re = self._compile_regex(self.target_role_alias)

        self.education_rank = {
            "": -1,
            "no degree": 0,
            "high school": 1,
            "ged": 1,
            "vocational diploma": 2,
            "diploma": 2,
            "associate": 2,
            "associate degree": 2,
            "bachelor": 3,
            "bachelor degree": 3,
            "bachelors degree": 3,
            "undergraduate degree": 3,
            "master": 4,
            "master degree": 4,
            "masters degree": 4,
            "mba": 4,
            "msc": 4,
            "ma": 4,
            "phd": 5,
            "doctorate": 5,
            "postdoctoral": 6,
        }

        self.seniority_rank = {
            "": -1,
            "intern": 0,
            "trainee": 0,
            "apprentice": 0,
            "entry level": 1,
            "junior": 1,
            "associate": 1,
            "mid level": 2,
            "intermediate": 2,
            "experienced": 2,
            "senior": 3,
            "staff": 3,
            "principal": 4,
            "lead": 4,
            "manager": 4,
            "senior manager": 5,
            "director": 5,
            "head": 5,
            "vp": 6,
            "vice president": 6,
            "chief": 6,
            "c level": 6,
            "executive": 6,
        }

    @staticmethod
    def _clean_vocab(values: Iterable[str]) -> Set[str]:
        return {clean_text(v) for v in values if clean_text(v)}

    def _build_alias_map(self, canonical_terms: Iterable[str]) -> Dict[str, str]:
        canonical_terms = {clean_text(t) for t in canonical_terms if clean_text(t)}
        alias_to_canonical = {term: term for term in canonical_terms}
        for alias, canonical in self.normalization.items():
            if canonical in canonical_terms and alias:
                alias_to_canonical[alias] = canonical
        keep_short = {"r", "c", "go", "js", "ts", "py", "qa", "hr", "ai", "ml", "dl", "bi", "ux", "ui"}
        return {a: c for a, c in alias_to_canonical.items() if len(a) > 2 or a in keep_short}

    def _compile_regex(self, alias_map: Dict[str, str]):
        aliases = sorted(alias_map.keys(), key=len, reverse=True)
        if not aliases:
            return None
        return re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(a) for a in aliases) + r")(?![a-z0-9])", re.I)

    def _extract(self, text: str, regex, alias_map: Dict[str, str]) -> Set[str]:
        if regex is None:
            return set()
        found = set()
        for match in regex.finditer(clean_text(text)):
            canonical = alias_map.get(clean_text(match.group(1)))
            if canonical:
                found.add(canonical)
        return found

    def _choose_ranked(self, terms: Iterable[str], rank_map: Dict[str, int]) -> str:
        ranked = [(rank_map.get(str(t).strip().lower(), -1), str(t).strip().lower()) for t in terms]
        ranked = [x for x in ranked if x[0] >= 0]
        return max(ranked)[1] if ranked else ""

    def extract_years(self, text: str):
        t = clean_text(text)
        nums = []
        for match in re.finditer(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)", t):
            try:
                nums.append(int(match.group(1)))
            except Exception:
                pass
        if nums:
            return max(nums)
        if re.search(r"fresh graduate|recent graduate|entry level|entry-level|intern|internship|beginner", t):
            return 0
        if re.search(r"junior", t):
            return 1
        if re.search(r"mid level|mid-level|intermediate", t):
            return 3
        if re.search(r"senior|lead|principal|manager|director|chief", t):
            return 5
        return np.nan

    def extract_target_role_from_text(self, text: str) -> str:
        raw = " " + re.sub(r"\s+", " ", str(text)) + " "
        patterns = [
            r"target role:\s*([^\.\n]+)",
            r"seeking (?:a|an)?\s*([^\.\n]+?)\s+(?:role|position)",
            r"applying for\s*([^\.\n]+?)(?:\.|$)",
            r"interested in\s*([^\.\n]+?)(?:\.|$)",
            r"career objective:\s*([^\.\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw, flags=re.IGNORECASE)
            if match:
                value = re.sub(r"\s+", " ", match.group(1)).strip(" .,-").lower()
                if 2 <= len(value) <= 100:
                    return value
        roles_found = self._extract(text, self.target_role_re, self.target_role_alias)
        return sorted(roles_found)[0] if roles_found else ""

    def extract_cv_profile(self, resume_text: str) -> Dict:
        skills = self._extract(resume_text, self.skill_re, self.skill_alias)
        tools = self._extract(resume_text, self.tool_re, self.tool_alias) | (skills & self.tools_vocab)
        domains = self._extract(resume_text, self.domain_re, self.domain_alias)
        projects = self._extract(resume_text, self.project_re, self.project_alias)
        certs = self._extract(resume_text, self.cert_re, self.cert_alias)
        education_terms = self._extract(resume_text, self.education_re, self.education_alias)
        seniority_terms = self._extract(resume_text, self.seniority_re, self.seniority_alias)

        return {
            "cv_skills": skills,
            "cv_tools_technologies": tools,
            "cv_domain_keywords": domains,
            "cv_project_experience": projects,
            "cv_certifications": certs,
            "cv_education_level": self._choose_ranked(education_terms, self.education_rank),
            "cv_seniority_level": self._choose_ranked(seniority_terms, self.seniority_rank),
            "cv_years_experience": self.extract_years(resume_text),
            "cv_target_role": self.extract_target_role_from_text(resume_text),
        }

    def extract_job_features(self, job_role: str, job_description: str) -> Dict:
        text = f"{job_role} {job_description}"
        skills = self._extract(text, self.skill_re, self.skill_alias)
        tools = self._extract(text, self.tool_re, self.tool_alias) | (skills & self.tools_vocab)
        domains = self._extract(text, self.domain_re, self.domain_alias)
        responsibilities = self._extract(text, self.project_re, self.project_alias)
        education_terms = self._extract(text, self.education_re, self.education_alias)
        seniority_terms = self._extract(text, self.seniority_re, self.seniority_alias)
        return {
            "job_required_skills": set_to_pipe(skills),
            "job_required_education": self._choose_ranked(education_terms, self.education_rank),
            "job_required_years_experience": self.extract_years(text),
            "job_responsibilities": set_to_pipe(responsibilities),
            "job_required_tools": set_to_pipe(tools),
            "job_domain": set_to_pipe(domains),
            "job_seniority_level": self._choose_ranked(seniority_terms, self.seniority_rank),
        }

    # Backward-compatible helpers for older code paths.
    def extract_cv(self, resume_text: str) -> Dict:
        profile = self.extract_cv_profile(resume_text)
        return {
            "skills": profile["cv_skills"],
            "tools": profile["cv_tools_technologies"],
            "domains": profile["cv_domain_keywords"],
            "education": profile["cv_education_level"],
            "seniority": profile["cv_seniority_level"],
            "years_experience": profile["cv_years_experience"],
        }

    def extract_job(self, job_role: str, job_description: str) -> Dict:
        features = self.extract_job_features(job_role, job_description)
        return {
            "skills": pipe_to_set(features["job_required_skills"]),
            "tools": pipe_to_set(features["job_required_tools"]),
            "domains": pipe_to_set(features["job_domain"]),
            "education": features["job_required_education"],
            "seniority": features["job_seniority_level"],
            "years_experience": features["job_required_years_experience"],
        }
