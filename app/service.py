from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from app.config import settings
from app.extractor import TaxonomyExtractor, clean_text, normalize_role_light, pipe_to_set
from app.scoring import (
    add_core_scores,
    build_user_score_summary,
    prepare_raw_inference_frame,
    scored_dataframe_to_records,
)


class JobMatchingService:
    def __init__(self):
        self.extractor = TaxonomyExtractor(settings.taxonomy_path)
        self.model_load_error: Optional[str] = None
        self.model = self._load_model()
        self.model_loaded = self.model is not None
        self.catalog = self._build_or_load_structured_catalog(self._load_raw_catalog())

    def _load_model(self):
        if not settings.use_model_if_available:
            return None

        model_path = Path(settings.model_path)
        feature_config_path = Path(settings.feature_config_path)
        if not model_path.exists() or not feature_config_path.exists():
            missing = []
            if not model_path.exists():
                missing.append(str(model_path))
            if not feature_config_path.exists():
                missing.append(str(feature_config_path))
            self.model_load_error = "Model/feature config not found: " + ", ".join(missing)
            return None

        try:
            from app.model_adapter import V5ModelAdapter

            return V5ModelAdapter(model_path, feature_config_path)
        except Exception as exc:  # pragma: no cover - depends on local TF/model artifact
            self.model_load_error = str(exc)
            return None

    def _load_raw_catalog(self) -> pd.DataFrame:
        catalog_path = settings.job_catalog_path if Path(settings.job_catalog_path).exists() else settings.fallback_job_catalog_path
        catalog_path = Path(catalog_path)
        df = pd.read_csv(catalog_path)

        rename_map = {}
        if "Job Description" in df.columns and "job_description" not in df.columns:
            rename_map["Job Description"] = "job_description"
        if "Job Roles" in df.columns and "job_role" not in df.columns:
            rename_map["Job Roles"] = "job_role"
        if "nama_role" in df.columns and "job_role" not in df.columns:
            rename_map["nama_role"] = "job_role"
        df = df.rename(columns=rename_map)

        missing = {"job_role", "job_description"} - set(df.columns)
        if missing:
            raise ValueError(f"Job catalog missing columns: {missing}")

        if "role_group" not in df.columns:
            df["role_group"] = df["job_role"].apply(clean_text)
        if "role_family" not in df.columns:
            df["role_family"] = "unknown"

        required_raw_catalog_columns = ["job_role", "job_description", "role_group", "role_family"]
        df = df[required_raw_catalog_columns].copy()

        for col in required_raw_catalog_columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

        df = df[df["job_role"].ne("") & df["job_description"].ne("")].copy()
        df = df.drop_duplicates(subset=["job_role", "job_description"], keep="first").reset_index(drop=True)

        df["source_dataset"] = "unique_job_role_descriptions_v5"
        df["source_file"] = catalog_path.name
        df["source_row_id"] = np.arange(len(df))
        df["role_key"] = df["job_role"].apply(normalize_role_light)
        return df

    def _build_or_load_structured_catalog(self, raw_catalog: pd.DataFrame, force_rebuild: bool = False) -> pd.DataFrame:
        cache_path = Path(settings.structured_job_catalog_cache_path)
        required_structured_cols = [
            "job_role",
            "job_description",
            "role_group",
            "role_family",
            "source_dataset",
            "source_file",
            "source_row_id",
            "role_key",
            "job_required_skills",
            "job_required_education",
            "job_required_years_experience",
            "job_responsibilities",
            "job_required_tools",
            "job_domain",
            "job_seniority_level",
        ]

        if cache_path.exists() and not force_rebuild:
            cached = pd.read_csv(cache_path)
            missing = [col for col in required_structured_cols if col not in cached.columns]
            if not missing:
                catalog = cached[required_structured_cols].copy()
                for col in ["job_role", "job_description", "role_group", "role_family", "role_key"]:
                    catalog[col] = catalog[col].fillna("").astype(str).str.strip()
                catalog = catalog[catalog["job_role"].ne("") & catalog["job_description"].ne("")].copy()
                catalog["role_key"] = catalog["job_role"].apply(normalize_role_light)
                return catalog.reset_index(drop=True)

        feature_rows = [
            self.extractor.extract_job_features(row.job_role, row.job_description)
            for row in raw_catalog.itertuples(index=False)
        ]
        feature_df = pd.DataFrame(feature_rows)
        structured_catalog = pd.concat([raw_catalog.reset_index(drop=True), feature_df], axis=1)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        structured_catalog.to_csv(cache_path, index=False, encoding="utf-8")
        return structured_catalog.reset_index(drop=True)

    def _score_dataframe(self, resume_text: str, jobs: pd.DataFrame) -> pd.DataFrame:
        jobs = jobs.copy().reset_index(drop=True)
        if jobs.empty:
            return jobs

        raw_features = prepare_raw_inference_frame(resume_text, jobs, self.extractor)

        probabilities = None
        if self.model_loaded:
            probabilities = self.model.predict(raw_features, batch_size=settings.model_batch_size)

        scored = add_core_scores(jobs, raw_features, probabilities=probabilities)
        return scored.sort_values("ranking_score", ascending=False).reset_index(drop=True)

    def _score_rows(self, resume_text: str, jobs: pd.DataFrame) -> list[dict]:
        scored = self._score_dataframe(resume_text, jobs)
        return scored_dataframe_to_records(scored)

    def score_cv_for_role(self, resume_text: str, selected_role: str, top_k: int = 5) -> dict:
        selected_key = normalize_role_light(selected_role)
        jobs = self.catalog[self.catalog["role_key"].astype(str) == selected_key].copy()

        if jobs.empty:
            raise ValueError(
                f"Role '{selected_role}' tidak ditemukan secara exact. "
                "Gunakan role yang sama dengan job_role di catalog."
            )

        jobs = jobs.drop_duplicates(subset=["job_role", "job_description"]).head(settings.max_role_descriptions).reset_index(drop=True)
        scored_df = self._score_dataframe(resume_text, jobs)
        summary = build_user_score_summary(scored_df, top_k=top_k)
        top_df = scored_df.head(top_k)

        return {
            "selected_role": selected_role,
            "total_job_descriptions_used": int(len(jobs)),
            **summary,
            "top_matches": scored_dataframe_to_records(top_df),
        }

    def _fast_prefilter_jobs_for_cv(
        self,
        resume_text: str,
        prefilter_k: int,
        role_hint: Optional[str] = None,
        source_filter: Optional[Union[str, list[str]]] = None,
    ) -> pd.DataFrame:
        candidates = self.catalog.copy()

        if source_filter is not None:
            if isinstance(source_filter, str):
                source_filter = [source_filter]
            candidates = candidates[candidates["source_dataset"].isin(source_filter)].copy()

        if role_hint is not None and str(role_hint).strip():
            role_hint_clean = clean_text(role_hint)
            hinted = candidates[
                candidates["role_key"].astype(str).str.contains(role_hint_clean, case=False, na=False)
                | candidates["job_role"].astype(str).str.contains(str(role_hint), case=False, na=False)
            ].copy()
            if not hinted.empty:
                candidates = hinted

        cv_profile = self.extractor.extract_cv_profile(resume_text)
        cv_terms = (
            cv_profile["cv_skills"]
            | cv_profile["cv_tools_technologies"]
            | cv_profile["cv_domain_keywords"]
            | set(clean_text(cv_profile.get("cv_target_role", "")).split())
        )

        def quick_score(row):
            job_terms = (
                pipe_to_set(row.get("job_required_skills", ""))
                | pipe_to_set(row.get("job_required_tools", ""))
                | pipe_to_set(row.get("job_domain", ""))
                | set(clean_text(row.get("job_role", "")).split())
            )
            if not job_terms:
                return 0.0
            return len(cv_terms & job_terms) / len(job_terms)

        candidates = candidates.copy()
        candidates["quick_filter_score"] = candidates.apply(quick_score, axis=1)
        return candidates.sort_values("quick_filter_score", ascending=False).head(prefilter_k).reset_index(drop=True)

    def recommend_jobs(
        self,
        resume_text: str,
        top_n: int = 10,
        role_hint: Optional[str] = None,
        source_filter: Optional[Union[str, list[str]]] = None,
    ) -> dict:
        candidate_jobs = self._fast_prefilter_jobs_for_cv(
            resume_text=resume_text,
            prefilter_k=settings.recommendation_prefilter_k,
            role_hint=role_hint,
            source_filter=source_filter,
        )

        scored_df = self._score_dataframe(resume_text, candidate_jobs)
        scored_df = scored_df.drop_duplicates(subset=["job_role", "job_description"]).reset_index(drop=True)
        summary = build_user_score_summary(scored_df, top_k=top_n)
        top_df = scored_df.head(top_n)

        return {
            "total_candidates_scored": int(len(scored_df)),
            "top_n": top_n,
            **summary,
            "recommendations": scored_dataframe_to_records(top_df),
        }


service = JobMatchingService()
