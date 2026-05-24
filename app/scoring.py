import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from app.extractor import (
    clean_text,
    jaccard_similarity,
    pipe_to_set,
    safe_ratio,
    set_to_pipe,
    tokenize_for_overlap,
    to_list,
)


DETAIL_NUMERIC_COLUMNS = [
    "structured_skill_match_ratio",
    "structured_tool_match_ratio",
    "structured_domain_match_ratio",
    "structured_experience_match",
    "structured_education_match",
    "structured_project_match_ratio",
    "structured_responsibility_overlap",
    "structured_seniority_match",
    "structured_certification_match",
    "semantic_proxy_score",
    "lexical_jaccard",
    "profile_quality_signal",
    "job_title_similarity",
    "role_keyword_overlap",
    "resume_skill_count",
    "job_skill_count",
    "skill_match_count",
]

RATIO_COLUMNS = [
    "structured_skill_match_ratio",
    "structured_tool_match_ratio",
    "structured_domain_match_ratio",
    "structured_experience_match",
    "structured_education_match",
    "structured_project_match_ratio",
    "structured_responsibility_overlap",
    "structured_seniority_match",
    "structured_certification_match",
    "semantic_proxy_score",
    "profile_quality_signal",
    "lexical_jaccard",
    "job_title_similarity",
    "role_keyword_overlap",
]

EXPLANATION_COLUMNS = [
    "matched_skills",
    "missing_skills",
    "matched_tools",
    "missing_tools",
    "matched_domains",
    "missing_domains",
]


def _is_missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return value is None


def education_match_from_terms(cv_education, job_education, education_rank: dict) -> float:
    cv_education = "" if _is_missing(cv_education) else str(cv_education).strip().lower()
    job_education = "" if _is_missing(job_education) else str(job_education).strip().lower()

    if not job_education:
        return 0.5
    if not cv_education:
        return 0.0

    cv_rank = education_rank.get(cv_education, -1)
    job_rank = education_rank.get(job_education, -1)

    if cv_rank < 0 or job_rank < 0:
        return 0.5
    if cv_rank >= job_rank:
        return 1.0
    if job_rank - cv_rank == 1:
        return 0.6
    return 0.0


def experience_match_from_years(cv_years, job_years) -> float:
    if _is_missing(job_years):
        return 0.5
    if _is_missing(cv_years):
        return 0.0

    cv_years = float(cv_years)
    job_years = float(job_years)

    if cv_years >= job_years:
        return 1.0
    if job_years <= 0:
        return 1.0
    return max(0.0, min(cv_years / job_years, 1.0))


def seniority_match_from_cv_profile(cv_profile: dict, job_seniority, seniority_rank: dict) -> float:
    job_seniority = "" if _is_missing(job_seniority) else str(job_seniority).strip().lower()

    if not job_seniority:
        return 0.5

    cv_seniority = cv_profile.get("cv_seniority_level", "")

    if not cv_seniority:
        text = f"{cv_profile.get('cv_target_role', '')}".lower()
        if re.search(r"intern|internship|trainee|apprentice", text):
            cv_seniority = "intern"
        elif re.search(r"fresh graduate|entry level|entry-level|junior|associate", text):
            cv_seniority = "entry level"
        elif re.search(r"mid level|mid-level|intermediate|experienced", text):
            cv_seniority = "mid level"
        elif re.search(r"senior|sr", text):
            cv_seniority = "senior"
        elif re.search(r"lead|principal|manager|head|chief|director", text):
            cv_seniority = "lead"

    if not cv_seniority:
        return 0.0

    cv_rank = seniority_rank.get(cv_seniority, -1)
    job_rank = seniority_rank.get(job_seniority, -1)

    if cv_rank < 0 or job_rank < 0:
        return 0.5
    if cv_rank >= job_rank:
        return 1.0
    if job_rank - cv_rank == 1:
        return 0.6
    return 0.0


def _get_structured_set(job_row, structured_col: str, fallback_terms: Optional[Iterable[str]] = None) -> set[str]:
    if structured_col in job_row and not _is_missing(job_row.get(structured_col)):
        values = pipe_to_set(job_row.get(structured_col, ""))
        if values:
            return values
    return set(fallback_terms or [])


def compute_pair_features_v5(resume_text: str, job_row, extractor, cv_profile: Optional[dict] = None) -> dict:
    """Feature engineering V5 yang disalin dari notebook inference."""
    job_role = str(job_row.get("job_role", ""))
    job_description = str(job_row.get("job_description", ""))

    resume_clean = clean_text(resume_text)
    job_text_clean = clean_text(job_role + " " + job_description)
    job_role_clean = clean_text(job_role)

    if cv_profile is None:
        cv_profile = extractor.extract_cv_profile(resume_text)

    fallback_job = None
    required_structured_cols = [
        "job_required_skills",
        "job_required_tools",
        "job_domain",
        "job_responsibilities",
        "job_required_education",
        "job_required_years_experience",
        "job_seniority_level",
    ]
    if any(col not in job_row for col in required_structured_cols):
        fallback_job = extractor.extract_job_features(job_role, job_description)

    cv_skills = cv_profile["cv_skills"]
    job_skills = _get_structured_set(
        job_row,
        "job_required_skills",
        pipe_to_set(fallback_job.get("job_required_skills", "")) if fallback_job else None,
    )

    cv_tools = cv_profile["cv_tools_technologies"]
    job_tools = _get_structured_set(
        job_row,
        "job_required_tools",
        pipe_to_set(fallback_job.get("job_required_tools", "")) if fallback_job else None,
    )

    cv_domains = cv_profile["cv_domain_keywords"]
    job_domains = _get_structured_set(
        job_row,
        "job_domain",
        pipe_to_set(fallback_job.get("job_domain", "")) if fallback_job else None,
    )

    cv_projects = cv_profile["cv_project_experience"]
    job_responsibilities = _get_structured_set(
        job_row,
        "job_responsibilities",
        pipe_to_set(fallback_job.get("job_responsibilities", "")) if fallback_job else None,
    )

    matched_skills = cv_skills & job_skills
    matched_tools = cv_tools & job_tools
    matched_domains = cv_domains & job_domains
    project_overlap_set = (cv_projects | cv_domains) & (job_domains | job_responsibilities)

    structured_skill_match_ratio = safe_ratio(len(matched_skills), len(job_skills))
    structured_skill_coverage_cv_to_job = safe_ratio(len(matched_skills), len(cv_skills))

    structured_tool_match_ratio = safe_ratio(len(matched_tools), len(job_tools))
    structured_domain_match_ratio = safe_ratio(len(matched_domains), len(job_domains))
    structured_project_match_ratio = safe_ratio(len(project_overlap_set), len(job_domains | job_responsibilities))
    structured_responsibility_overlap = safe_ratio(len(cv_projects & job_responsibilities), len(job_responsibilities))

    job_required_education = job_row.get("job_required_education", "")
    if fallback_job and not str(job_required_education).strip():
        job_required_education = fallback_job.get("job_required_education", "")

    job_required_years_experience = job_row.get("job_required_years_experience", np.nan)
    if fallback_job and _is_missing(job_required_years_experience):
        job_required_years_experience = fallback_job.get("job_required_years_experience", np.nan)

    job_seniority_level = job_row.get("job_seniority_level", "")
    if fallback_job and not str(job_seniority_level).strip():
        job_seniority_level = fallback_job.get("job_seniority_level", "")

    structured_education_match = education_match_from_terms(
        cv_profile["cv_education_level"],
        job_required_education,
        extractor.education_rank,
    )

    structured_experience_match = experience_match_from_years(
        cv_profile["cv_years_experience"],
        job_required_years_experience,
    )

    structured_seniority_match = seniority_match_from_cv_profile(
        cv_profile,
        job_seniority_level,
        extractor.seniority_rank,
    )

    structured_certification_match = 0.5

    structured_match_score = (
        0.35 * structured_skill_match_ratio
        + 0.15 * structured_tool_match_ratio
        + 0.15 * structured_domain_match_ratio
        + 0.10 * structured_experience_match
        + 0.08 * structured_education_match
        + 0.07 * structured_project_match_ratio
        + 0.05 * structured_seniority_match
        + 0.03 * structured_certification_match
        + 0.02 * structured_responsibility_overlap
    )

    resume_tokens = tokenize_for_overlap(resume_clean)
    job_tokens = tokenize_for_overlap(job_text_clean)
    role_tokens = tokenize_for_overlap(job_role_clean)

    lexical_jaccard = jaccard_similarity(resume_tokens, job_tokens)
    job_title_similarity = safe_ratio(len(role_tokens & resume_tokens), len(role_tokens))
    role_keyword_overlap = safe_ratio(len(role_tokens & (resume_tokens | job_tokens)), len(role_tokens))

    semantic_proxy_score = (
        0.28 * structured_skill_match_ratio
        + 0.22 * structured_skill_coverage_cv_to_job
        + 0.18 * job_title_similarity
        + 0.14 * role_keyword_overlap
        + 0.10 * lexical_jaccard
        + 0.04 * structured_education_match
        + 0.04 * structured_experience_match
    )
    profile_enrichment = cv_profile.get("_profile_enrichment", {})
    profile_quality_signal = float(profile_enrichment.get("profile_quality_signal", 50.0)) / 100.0
    return {
        "Resume": resume_text,
        "Job Roles": job_role,
        "Job Description": job_description,
        "resume_clean": resume_clean,
        "job_text_clean": job_text_clean,
        # V4-compatible features
        "resume_skill_count": len(cv_skills),
        "job_skill_count": len(job_skills),
        "skill_match_count": len(matched_skills),
        "skill_match_ratio": structured_skill_match_ratio,
        "skill_coverage_resume_to_job": structured_skill_coverage_cv_to_job,
        "lexical_jaccard": lexical_jaccard,
        "job_title_similarity": job_title_similarity,
        "role_keyword_overlap": role_keyword_overlap,
        "education_match": structured_education_match,
        "experience_match": structured_experience_match,
        "resume_word_count": len(resume_clean.split()),
        "job_word_count": len(job_text_clean.split()),
        "semantic_proxy_score": semantic_proxy_score,
        # V5 structured features
        "structured_cv_skill_count": len(cv_skills),
        "structured_job_skill_count": len(job_skills),
        "structured_matched_skill_count": len(matched_skills),
        "structured_missing_skill_count": max(len(job_skills - cv_skills), 0),
        "structured_skill_match_ratio": structured_skill_match_ratio,
        "structured_skill_coverage_cv_to_job": structured_skill_coverage_cv_to_job,
        "structured_cv_tool_count": len(cv_tools),
        "structured_job_tool_count": len(job_tools),
        "structured_matched_tool_count": len(matched_tools),
        "structured_missing_tool_count": max(len(job_tools - cv_tools), 0),
        "structured_tool_match_ratio": structured_tool_match_ratio,
        "structured_cv_domain_count": len(cv_domains),
        "structured_job_domain_count": len(job_domains),
        "structured_matched_domain_count": len(matched_domains),
        "structured_domain_match_ratio": structured_domain_match_ratio,
        "structured_project_match_ratio": structured_project_match_ratio,
        "structured_responsibility_overlap": structured_responsibility_overlap,
        "structured_education_match": structured_education_match,
        "structured_experience_match": structured_experience_match,
        "structured_seniority_match": structured_seniority_match,
        "structured_certification_match": structured_certification_match,
        "structured_match_score": structured_match_score,
        "profile_quality_signal": profile_quality_signal,
        # Explanation fields
        "matched_skills": set_to_pipe(matched_skills),
        "missing_skills": set_to_pipe(job_skills - cv_skills),
        "matched_tools": set_to_pipe(matched_tools),
        "missing_tools": set_to_pipe(job_tools - cv_tools),
        "matched_domains": set_to_pipe(matched_domains),
        "missing_domains": set_to_pipe(job_domains - cv_domains),
    }


def prepare_raw_inference_frame(
    resume_text: str,
    candidate_jobs: pd.DataFrame,
    extractor,
    cv_profile: Optional[dict] = None,
) -> pd.DataFrame:
    if cv_profile is None:
        cv_profile = extractor.extract_cv_profile(resume_text)

    rows = [
        compute_pair_features_v5(
            resume_text,
            job_row,
            extractor=extractor,
            cv_profile=cv_profile,
        )
        for _, job_row in candidate_jobs.iterrows()
    ]

    raw_frame = pd.DataFrame(rows)

    if raw_frame.empty:
        return raw_frame

    # Notebook mengisi missing numeric features berdasarkan feature_config sebelum normalisasi.
    numeric_like_cols = raw_frame.select_dtypes(include=["number"]).columns.tolist()

    for col in numeric_like_cols:
        raw_frame[col] = (
            pd.to_numeric(raw_frame[col], errors="coerce")
            .fillna(0.0)
            .astype(np.float32)
        )

    return raw_frame


def add_core_scores(result: pd.DataFrame, raw_features: pd.DataFrame, probabilities: Optional[np.ndarray] = None) -> pd.DataFrame:
    result = result.copy().reset_index(drop=True)
    raw_features = raw_features.reset_index(drop=True)

    if probabilities is None:
        probabilities = raw_features["structured_match_score"].clip(0, 1).to_numpy(dtype=float)

    result["match_probability"] = np.asarray(probabilities, dtype=float)
    result["fit_score"] = (result["match_probability"].clip(0, 1) * 100).round(2)
    result["structured_score"] = (raw_features["structured_match_score"].clip(0, 1) * 100).round(2)

    profile_signal = (
        raw_features["profile_quality_signal"]
        if "profile_quality_signal" in raw_features.columns
        else pd.Series([0.5] * len(result))
    )

    profile_signal = (
        pd.to_numeric(profile_signal, errors="coerce")
        .fillna(0.5)
        .clip(0, 1)
    )

    profile_quality_score = (profile_signal * 100).round(2)

    # User-facing score per job recommendation.
    result["user_match_score"] = (
        0.70 * result["fit_score"]
        + 0.25 * result["structured_score"]
        + 0.05 * profile_quality_score
    ).clip(0, 100).round(2)

    # Internal ranking score for sorting.
    base_ranking = (
        0.75 * result["fit_score"]
        + 0.25 * result["structured_score"]
    )

    profile_multiplier = 0.95 + 0.05 * profile_signal

    result["ranking_score"] = (
        base_ranking * profile_multiplier
    ).clip(0, 100).round(2)

    for col in DETAIL_NUMERIC_COLUMNS:
        if col in raw_features.columns:
            result[col] = raw_features[col].values

    for col in EXPLANATION_COLUMNS:
        if col in raw_features.columns:
            result[col] = raw_features[col].values

    for col in RATIO_COLUMNS:
        if col in result.columns:
            result[col + "_pct"] = (
                pd.to_numeric(result[col], errors="coerce")
                .fillna(0)
                .clip(0, 1)
                * 100
            ).round(2)

    return result


def build_user_score_summary(scored_df: pd.DataFrame, top_k: int = 5) -> dict:
    if scored_df.empty:
        return {
            "fit_score_best": 0.0,
            "fit_score_average_top_k": 0.0,
            "fit_score_average_all": 0.0,
            "structured_score_best": 0.0,
            "structured_score_average_top_k": 0.0,
            "ranking_score_best": 0.0,
            "ranking_score_average_top_k": 0.0,
            "final_user_score": 0.0,
            "user_friendly_score": {
                "overall_score": 0.0,
                "fit_score": {
                    "score": 0.0,
                    "weight": 0.70,
                    "source": "average_top_k",
                    "description": "Average AI fit score from the top matched jobs.",
                },
                "structured_score": {
                    "score": 0.0,
                    "weight": 0.25,
                    "source": "best_evidence",
                    "description": "Best evidence-based score from matched skills, tools, domains, experience, and education.",
                },
                "profile_quality_score": {
                    "score": None,
                    "weight": 0.05,
                    "source": "profile_enrichment",
                    "description": "Resume profile completeness signal extracted from CV.",
                },
            },
            "top_k_used": 0,
            "total_candidates_scored": 0,
            "score_policy": "No candidate scored.",
        }

    scored_df = scored_df.copy().sort_values("ranking_score", ascending=False).reset_index(drop=True)
    top_df = scored_df.head(top_k)

    fit_score_best = round(float(scored_df["fit_score"].max()), 2)
    fit_score_average_top_k = round(float(top_df["fit_score"].mean()), 2)
    fit_score_average_all = round(float(scored_df["fit_score"].mean()), 2)

    structured_score_best = round(float(scored_df["structured_score"].max()), 2)
    structured_score_average_top_k = round(float(top_df["structured_score"].mean()), 2)

    ranking_score_best = round(float(scored_df["ranking_score"].max()), 2)
    ranking_score_average_top_k = round(float(top_df["ranking_score"].mean()), 2)

    profile_quality_score = None
    if "profile_quality_signal_pct" in top_df.columns:
        profile_quality_score = round(
            float(
                pd.to_numeric(
                    top_df["profile_quality_signal_pct"],
                    errors="coerce",
                )
                .fillna(0)
                .mean()
            ),
            2,
        )

    profile_quality_for_score = (
        profile_quality_score
        if profile_quality_score is not None
        else 50.0
    )

    # Main user-facing score:
    # 70% average top_k model fit
    # 25% best structured evidence
    # 5% profile quality from enrichment/NER
    final_user_score = round(
        0.70 * fit_score_average_top_k
        + 0.25 * structured_score_best
        + 0.05 * profile_quality_for_score,
        2,
    )

    return {
        "fit_score_best": fit_score_best,
        "fit_score_average_top_k": fit_score_average_top_k,
        "fit_score_average_all": fit_score_average_all,
        "structured_score_best": structured_score_best,
        "structured_score_average_top_k": structured_score_average_top_k,
        "ranking_score_best": ranking_score_best,
        "ranking_score_average_top_k": ranking_score_average_top_k,
        "final_user_score": final_user_score,
        "user_friendly_score": {
            "overall_score": final_user_score,
            "fit_score": {
                "score": fit_score_average_top_k,
                "weight": 0.70,
                "source": "average_top_k",
                "description": "Average AI fit score from the top matched jobs.",
            },
            "structured_score": {
                "score": structured_score_best,
                "weight": 0.25,
                "source": "best_evidence",
                "description": "Best evidence-based score from matched skills, tools, domains, experience, and education.",
            },
            "profile_quality_score": {
                "score": profile_quality_score,
                "weight": 0.05,
                "source": "profile_enrichment",
                "description": "Resume profile completeness signal extracted from CV.",
            },
        },
        "top_k_used": int(len(top_df)),
        "total_candidates_scored": int(len(scored_df)),
        "score_policy": (
            "Use user_friendly_score.overall_score as the main user-facing score. "
            "It combines average top_k AI fit score, best structured evidence score, "
            "and a small profile quality signal. Individual jobs are still sorted using ranking_score."
        ),
    }

def pipe_string_to_list(value) -> list[str]:
    return to_list(pipe_to_set(value))


def scored_dataframe_to_records(scored_df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for _, row in scored_df.iterrows():
        item = row.to_dict()
        for col in EXPLANATION_COLUMNS:
            item[col] = pipe_string_to_list(item.get(col, ""))
        # Pydantic/JSON tidak suka NaN.
        for key, value in list(item.items()):
            if _is_missing(value):
                item[key] = None
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                item[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                item[key] = float(value)
        records.append(item)
    return records


# Backward-compatible function name. Dipakai hanya jika ada kode lama yang masih memanggilnya.
def calculate_pair_score(resume_text: str, job_role: str, job_description: str, cv: dict, job: dict, education_rank: dict) -> dict:
    cv_skills, job_skills = cv["skills"], job["skills"]
    cv_tools, job_tools = cv["tools"], job["tools"]
    cv_domains, job_domains = cv["domains"], job["domains"]

    matched_skills = cv_skills & job_skills
    matched_tools = cv_tools & job_tools
    matched_domains = cv_domains & job_domains

    skill_ratio = safe_ratio(len(matched_skills), len(job_skills))
    tool_ratio = safe_ratio(len(matched_tools), len(job_tools))
    domain_ratio = safe_ratio(len(matched_domains), len(job_domains))
    exp_ratio = experience_match_from_years(cv["years_experience"], job["years_experience"])
    edu_ratio = education_match_from_terms(cv["education"], job["education"], education_rank)
    lexical = jaccard_similarity(tokenize_for_overlap(resume_text), tokenize_for_overlap(f"{job_role} {job_description}"))

    structured_score = (
        0.40 * skill_ratio
        + 0.18 * tool_ratio
        + 0.16 * domain_ratio
        + 0.12 * exp_ratio
        + 0.09 * edu_ratio
        + 0.05 * lexical
    )
    fit_score = structured_score
    ranking_score = 0.75 * fit_score + 0.25 * structured_score

    return {
        "fit_score": round(fit_score * 100, 2),
        "structured_score": round(structured_score * 100, 2),
        "ranking_score": round(ranking_score * 100, 2),
        "structured_skill_match_ratio_pct": round(skill_ratio * 100, 2),
        "structured_tool_match_ratio_pct": round(tool_ratio * 100, 2),
        "structured_domain_match_ratio_pct": round(domain_ratio * 100, 2),
        "structured_experience_match_pct": round(exp_ratio * 100, 2),
        "structured_education_match_pct": round(edu_ratio * 100, 2),
        "matched_skills": to_list(matched_skills),
        "missing_skills": to_list(job_skills - cv_skills),
        "matched_tools": to_list(matched_tools),
        "missing_tools": to_list(job_tools - cv_tools),
        "matched_domains": to_list(matched_domains),
        "missing_domains": to_list(job_domains - cv_domains),
    }
