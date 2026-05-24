from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_name: str = "Evalify Job Matching AI API"
    artifacts_dir: Path = Path("artifacts")
    data_dir: Path = Path("data")

    taxonomy_path: Path = Path("artifacts/taxonomy.json")
    metadata_path: Path = Path("artifacts/metadata.json")
    feature_config_path: Path = Path("artifacts/feature_config_v5_structured_features.joblib")

    # TensorFlow model V5, sama seperti notebook inference.
    model_path: Path = Path("artifacts/evalify_custom_transformer_job_matching_v5_structured_features.keras")

    # Catalog utama dan cache structured features.
    job_catalog_path: Path = Path("data/unique_job_role_descriptions_v5.csv")
    fallback_job_catalog_path: Path = Path("data/sample_job_catalog.csv")
    structured_job_catalog_cache_path: Path = Path("data/unique_job_role_descriptions_v5_structured_cache.csv")

    use_model_if_available: bool = True

    # Sama dengan notebook: selected role memakai max_descriptions=300,
    # recommendation memakai prefilter_k=1000.
    max_role_descriptions: int = 300
    recommendation_prefilter_k: int = 1000
    model_batch_size: int = 128

    class Config:
        env_file = ".env"


settings = Settings()
