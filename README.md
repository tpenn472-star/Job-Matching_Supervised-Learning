# FastAPI V5 Inference Patch

Patch ini mengganti scoring FastAPI supaya mengikuti alur inference notebook V5:

1. Catalog dibaca dari `data/unique_job_role_descriptions_v5.csv`.
2. Structured job features dibuat/dibaca dari `data/unique_job_role_descriptions_v5_structured_cache.csv`.
3. Pair features dihitung dengan formula V5.
4. Jika artifact model tersedia, `fit_score = match_probability * 100`.
5. `structured_score = structured_match_score * 100`.
6. `ranking_score = 0.75 * fit_score + 0.25 * structured_score`.

## Cara pakai

Salin isi folder ini ke root project kamu, overwrite file berikut:

```text
main.py
app/config.py
app/extractor.py
app/model_adapter.py
app/pdf_utils.py
app/schemas.py
app/scoring.py
app/service.py
```

Pastikan file ini ada di project:

```text
artifacts/evalify_custom_transformer_job_matching_v5_structured_features.keras
artifacts/feature_config_v5_structured_features.joblib
artifacts/taxonomy.json
data/unique_job_role_descriptions_v5.csv
```

Jika `data/unique_job_role_descriptions_v5_structured_cache.csv` belum ada, API akan membuatnya saat startup.

Cek `/health`. Jika `model_loaded` bernilai `true`, scoring sudah memakai model TensorFlow seperti notebook. Jika `false`, API tetap jalan tetapi `fit_score` memakai fallback structured score.
