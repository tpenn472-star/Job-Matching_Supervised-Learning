from typing import List, Optional
from pydantic import BaseModel, Field

class CVRoleScoreRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    selected_role: str = Field(..., min_length=2)


class JobRecommendationRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    role_hint: Optional[str] = None


class MatchedEvidence(BaseModel):
    matched_skills: List[str]
    missing_skills: List[str]
    matched_tools: List[str]
    missing_tools: List[str]
    matched_domains: List[str]
    missing_domains: List[str]


class JobScoreItem(BaseModel):
    job_role: str
    fit_score: float
    structured_score: float
    ranking_score: float
    structured_skill_match_ratio_pct: float
    structured_tool_match_ratio_pct: float
    structured_domain_match_ratio_pct: float
    structured_experience_match_pct: float
    structured_education_match_pct: float
    evidence: MatchedEvidence
    job_description: str

    # Extra detail agar output FastAPI bisa mengikuti notebook V5.
    match_probability: Optional[float] = None
    structured_project_match_ratio_pct: Optional[float] = None
    structured_responsibility_overlap_pct: Optional[float] = None
    structured_seniority_match_pct: Optional[float] = None
    structured_certification_match_pct: Optional[float] = None
    semantic_proxy_score_pct: Optional[float] = None
    lexical_jaccard_pct: Optional[float] = None
    job_title_similarity_pct: Optional[float] = None
    role_keyword_overlap_pct: Optional[float] = None
    resume_skill_count: Optional[float] = None
    job_skill_count: Optional[float] = None
    skill_match_count: Optional[float] = None
    role_group: Optional[str] = None
    role_family: Optional[str] = None


class CVRoleScoreResponse(BaseModel):
    selected_role: str
    total_job_descriptions_used: int
    fit_score_best: float
    fit_score_average_top_k: float
    fit_score_average_all: float
    structured_score_best: float
    ranking_score_best: float
    top_k_used: Optional[int] = None
    total_candidates_scored: Optional[int] = None
    score_policy: Optional[str] = None
    top_matches: List[JobScoreItem]


class JobRecommendationResponse(BaseModel):
    total_candidates_scored: int
    top_n: int
    fit_score_best: Optional[float] = None
    fit_score_average_top_k: Optional[float] = None
    fit_score_average_all: Optional[float] = None
    structured_score_best: Optional[float] = None
    ranking_score_best: Optional[float] = None
    top_k_used: Optional[int] = None
    score_policy: Optional[str] = None
    recommendations: List[JobScoreItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    catalog_rows: int
    message: str
    model_load_error: Optional[str] = None
