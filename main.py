from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.config import settings
from app.pdf_utils import extract_text_from_pdf_bytes
from app.schemas import (
    CVRoleScoreRequest,
    CVRoleScoreResponse,
    HealthResponse,
    JobRecommendationRequest,
    JobRecommendationResponse,
    JobScoreItem,
    MatchedEvidence,
)
from app.service import service


app = FastAPI(
    title=settings.project_name,
    version="1.2.0",
    description=(
        "Evalify Job Matching V5 API. Scoring follows the V5 notebook inference: "
        "model probability as fit_score, structured_match_score as structured_score, "
        "and ranking_score = 0.75 * fit_score + 0.25 * structured_score."
    ),
)


def _as_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def to_response_item(item: dict) -> JobScoreItem:
    return JobScoreItem(
        job_role=str(item.get("job_role", "")),
        fit_score=float(item.get("fit_score", 0.0)),
        structured_score=float(item.get("structured_score", 0.0)),
        ranking_score=float(item.get("ranking_score", 0.0)),
        structured_skill_match_ratio_pct=float(item.get("structured_skill_match_ratio_pct", 0.0)),
        structured_tool_match_ratio_pct=float(item.get("structured_tool_match_ratio_pct", 0.0)),
        structured_domain_match_ratio_pct=float(item.get("structured_domain_match_ratio_pct", 0.0)),
        structured_experience_match_pct=float(item.get("structured_experience_match_pct", 0.0)),
        structured_education_match_pct=float(item.get("structured_education_match_pct", 0.0)),
        evidence=MatchedEvidence(
            matched_skills=item.get("matched_skills", []) or [],
            missing_skills=item.get("missing_skills", []) or [],
            matched_tools=item.get("matched_tools", []) or [],
            missing_tools=item.get("missing_tools", []) or [],
            matched_domains=item.get("matched_domains", []) or [],
            missing_domains=item.get("missing_domains", []) or [],
        ),
        job_description=str(item.get("job_description", "")),
        match_probability=_as_float_or_none(item.get("match_probability")),
        structured_project_match_ratio_pct=_as_float_or_none(item.get("structured_project_match_ratio_pct")),
        structured_responsibility_overlap_pct=_as_float_or_none(item.get("structured_responsibility_overlap_pct")),
        structured_seniority_match_pct=_as_float_or_none(item.get("structured_seniority_match_pct")),
        structured_certification_match_pct=_as_float_or_none(item.get("structured_certification_match_pct")),
        semantic_proxy_score_pct=_as_float_or_none(item.get("semantic_proxy_score_pct")),
        lexical_jaccard_pct=_as_float_or_none(item.get("lexical_jaccard_pct")),
        job_title_similarity_pct=_as_float_or_none(item.get("job_title_similarity_pct")),
        role_keyword_overlap_pct=_as_float_or_none(item.get("role_keyword_overlap_pct")),
        resume_skill_count=_as_float_or_none(item.get("resume_skill_count")),
        job_skill_count=_as_float_or_none(item.get("job_skill_count")),
        skill_match_count=_as_float_or_none(item.get("skill_match_count")),
        role_group=item.get("role_group"),
        role_family=item.get("role_family"),
    )


@app.get("/", response_model=HealthResponse)
def root():
    message = "Evalify AI API is running. Open /docs for Swagger UI."
    if not service.model_loaded:
        message += " Model is not loaded; API will fall back to structured score."
    return HealthResponse(
        status="ok",
        model_loaded=service.model_loaded,
        catalog_rows=len(service.catalog),
        message=message,
        model_load_error=service.model_load_error,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return root()



@app.post("/v1/cv-score", response_model=CVRoleScoreResponse)
def cv_score(payload: CVRoleScoreRequest):
    return _score_cv_text(
        resume_text=payload.resume_text,
        selected_role=payload.selected_role,
        top_k=5,
    )

@app.post("/v1/recommend-jobs", response_model=JobRecommendationResponse)
def recommend_jobs(payload: JobRecommendationRequest):
    return _recommend_jobs_text(
        resume_text=payload.resume_text,
        top_n=10,
        role_hint=payload.role_hint,
    )

@app.post("/v1/cv-score/pdf", response_model=CVRoleScoreResponse)
async def cv_score_pdf(
    file: UploadFile = File(..., description="CV/resume PDF file."),
    selected_role: str = Form(..., description="Target role selected by user."),
):
    _validate_pdf_file(file)
    pdf_bytes = await file.read()
    resume_text = extract_text_from_pdf_bytes(pdf_bytes)
    return _score_cv_text(resume_text=resume_text, selected_role=selected_role, top_k=5)


@app.post("/v1/recommend-jobs/pdf", response_model=JobRecommendationResponse)
async def recommend_jobs_pdf(
    file: UploadFile = File(..., description="CV/resume PDF file."),
    role_hint: Optional[str] = Form(None, description="Optional role keyword to narrow recommendation."),
):
    _validate_pdf_file(file)
    pdf_bytes = await file.read()
    resume_text = extract_text_from_pdf_bytes(pdf_bytes)
    return _recommend_jobs_text(resume_text=resume_text, top_n=10, role_hint=role_hint)

def _validate_pdf_file(file: UploadFile) -> None:
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()

    if not filename.endswith(".pdf") and content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")


def _score_cv_text(resume_text: str, selected_role: str, top_k: int) -> CVRoleScoreResponse:
    try:
        result = service.score_cv_for_role(resume_text, selected_role, top_k)
        return CVRoleScoreResponse(
            selected_role=result["selected_role"],
            total_job_descriptions_used=result["total_job_descriptions_used"],
            fit_score_best=result["fit_score_best"],
            fit_score_average_top_k=result["fit_score_average_top_k"],
            fit_score_average_all=result["fit_score_average_all"],
            structured_score_best=result["structured_score_best"],
            ranking_score_best=result["ranking_score_best"],
            top_k_used=result.get("top_k_used"),
            total_candidates_scored=result.get("total_candidates_scored"),
            score_policy=result.get("score_policy"),
            top_matches=[to_response_item(x) for x in result["top_matches"]],
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _recommend_jobs_text(
    resume_text: str,
    top_n: int,
    role_hint: Optional[str] = None,
) -> JobRecommendationResponse:
    result = service.recommend_jobs(resume_text, top_n, role_hint)
    return JobRecommendationResponse(
        total_candidates_scored=result["total_candidates_scored"],
        top_n=result["top_n"],
        fit_score_best=result.get("fit_score_best"),
        fit_score_average_top_k=result.get("fit_score_average_top_k"),
        fit_score_average_all=result.get("fit_score_average_all"),
        structured_score_best=result.get("structured_score_best"),
        ranking_score_best=result.get("ranking_score_best"),
        top_k_used=result.get("top_k_used"),
        score_policy=result.get("score_policy"),
        recommendations=[to_response_item(x) for x in result["recommendations"]],
    )
