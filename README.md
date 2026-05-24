# Job Matching Supervised Learning API

A FastAPI-based job matching service for scoring candidate CVs against selected job roles and recommending the most suitable job roles from a structured job catalog.

This project combines supervised learning, structured feature extraction, candidate profile enrichment, and API-based inference to support an AI-powered job matching workflow.

## Overview

This API compares a candidate CV or resume with job descriptions from a structured job catalog.

The system supports two main workflows:

1. Score a candidate CV against a selected job role.
2. Recommend the top job roles based on the uploaded CV.

The API accepts both plain text CV input and PDF resume upload. It extracts resume text, identifies structured candidate features, scores the candidate against job descriptions, and returns explainable matching evidence.

The project uses:

- FastAPI for API serving
- TensorFlow/Keras for supervised model inference
- Taxonomy-based feature extraction
- Candidate profile enrichment using a trained sequence model
- Structured matching evidence
- Optional GenAI-based explanation for selected role scoring

## Main Features

### 1. CV to Selected Role Scoring

Score a candidate CV against a specific selected job role.

Example use case:

> A candidate uploads a CV and selects "Software Engineer".  
> The API returns the top matching Software Engineer job descriptions, fit scores, structured evidence, and an optional AI-generated explanation.

### 2. Top 10 Job Recommendation

Recommend the most suitable job roles based on a candidate CV.

Example use case:

> A candidate uploads a CV without selecting a role.  
> The API returns the top 10 recommended job roles from the catalog.

Each recommended job includes:

- Job role
- Job description
- User-friendly match score
- AI fit score
- Structured evidence score
- Profile quality score
- Matched and missing skills/tools/domains

### 3. PDF Resume Upload

The API supports PDF resumes and extracts text automatically before scoring.

### 4. Taxonomy-Based Feature Extraction

The system extracts structured information from CVs and job descriptions, including:

- Skills
- Tools and technologies
- Domain keywords
- Education level
- Seniority level
- Years of experience
- Project experience terms
- Certifications

### 5. Candidate Profile Enrichment

The API includes an internal candidate profile enrichment layer.

This layer uses a trained sequence model to detect resume entities such as skills, education, experience, designation, company, and other profile signals. The extracted entities are validated using the existing taxonomy before being used in scoring.

This helps improve structured matching while keeping the scoring stable and explainable.

### 6. TensorFlow/Keras Model Inference

The main job matching model predicts the fit probability between a candidate resume and a job description.

The model artifact is stored in the `artifacts/` directory.

### 7. User-Friendly Scoring

The API returns both technical scores and user-friendly scores.

For selected role scoring, the main user-facing score combines:

```text
final_user_score =
  0.70 * fit_score_average_top_k
+ 0.25 * structured_score_best
+ 0.05 * profile_quality_score
```

For top 10 job recommendation, each job receives a per-job user-friendly match score:

```text
user_match_score =
  0.70 * fit_score
+ 0.25 * structured_score
+ 0.05 * profile_quality_score
```

### 8. Optional GenAI Explanation

For selected role scoring, the API can generate an explanation using GenAI.

The explanation may include:

- Summary
- Strengths
- Gaps
- Recommendations
- Feedback

GenAI is used only as a post-processing explanation layer. It does not determine the main score.

## Project Structure

```text
Job-Matching_Supervised-Learning/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── explanation.py
│   ├── extractor.py
│   ├── model_adapter.py
│   ├── pdf_utils.py
│   ├── profile_enrichment.py
│   ├── schemas.py
│   ├── scoring.py
│   └── service.py
│
├── artifacts/
│   ├── evalify_custom_transformer_job_matching_v5_structured_features.keras
│   ├── feature_config_v5_structured_features.joblib
│   ├── metadata.json
│   ├── ner_bilstm_attention_model.keras
│   ├── ner_vocab.json
│   └── taxonomy.json
│
├── data/
│   ├── unique_job_role_descriptions_v5.csv
│   ├── unique_job_role_descriptions_v5_structured_cache.csv
│   └── example_cv_web_developer.pdf
│
├── notebooks/
│   └── evalify_job_matching_custom_transformer_v5_structured_features_training.ipynb
│
├── tests/
│   └── test_payloads.json
│
├── .env.example
├── .gitattributes
├── .gitignore
├── example_client.py
├── main.py
├── run_api.py
├── requirements.txt
└── README.md
```

## Important Files

### `main.py`

Defines the FastAPI application and API routes.

Available routes:

```text
GET  /
GET  /health
POST /v1/cv-score
POST /v1/cv-score/pdf
POST /v1/recommend-jobs
POST /v1/recommend-jobs/pdf
```

### `app/config.py`

Stores configuration for model paths, taxonomy paths, catalog paths, profile enrichment artifacts, and optional GenAI settings.

Important paths include:

```text
artifacts/taxonomy.json
artifacts/metadata.json
artifacts/feature_config_v5_structured_features.joblib
artifacts/evalify_custom_transformer_job_matching_v5_structured_features.keras
artifacts/ner_bilstm_attention_model.keras
artifacts/ner_vocab.json
data/unique_job_role_descriptions_v5.csv
data/unique_job_role_descriptions_v5_structured_cache.csv
```

### `app/extractor.py`

Handles text cleaning and taxonomy-based extraction.

It extracts:

- Skills
- Tools and technologies
- Domains
- Education level
- Seniority level
- Years of experience
- Target role signals

### `app/profile_enrichment.py`

Loads the candidate profile enrichment model and extracts profile entities from resume text.

The enrichment output is validated against the taxonomy before being used in scoring. This helps prevent noisy raw entities from directly affecting the match score.

### `app/scoring.py`

Contains the feature engineering and scoring logic.

It calculates:

- `fit_score`
- `structured_score`
- `ranking_score`
- `user_match_score`
- `final_user_score`
- `user_friendly_score`
- Matching evidence such as matched and missing skills/tools/domains

### `app/model_adapter.py`

Loads the TensorFlow/Keras matching model and applies the same numeric feature normalization used during training.

The model receives:

- Cleaned resume text
- Cleaned job text
- Numeric structured features

### `app/explanation.py`

Handles optional GenAI-based explanation generation.

This file is used to generate natural-language explanations for selected role scoring. It does not modify the actual scores.

### `app/service.py`

Contains the main orchestration logic:

- Load job catalog
- Load structured catalog cache
- Load model artifacts
- Load profile enrichment model
- Score CV against selected role
- Recommend top jobs
- Generate optional explanation
- Return final API response

## Requirements

Recommended environment:

```text
Python 3.11
FastAPI
Uvicorn
TensorFlow
Pandas
NumPy
Joblib
Pydantic
pypdf
google-generativeai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If TensorFlow fails to load because of protobuf compatibility, make sure `protobuf` is updated:

```bash
pip install --upgrade protobuf
```

Recommended protobuf requirement:

```text
protobuf>=6.31.1
```

## Environment Variables

Create a `.env` file for local configuration.

Example:

```env
GENAI_API_KEY=
GENAI_MODEL_NAME=gemini-1.5-flash
USE_GENAI_EXPLANATION_IF_AVAILABLE=true
```

Do not commit `.env` to GitHub. Use `.env.example` for documentation.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Andro114/Job-Matching_Supervised-Learning.git
cd Job-Matching_Supervised-Learning
```

### 2. Pull Git LFS Files

This project uses Git LFS for large model and dataset artifacts.

```bash
git lfs install
git lfs pull
```

### 3. Create Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the API

Run the API with:

```bash
python run_api.py
```

Or run directly with Uvicorn:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

After the server starts, open:

```text
http://127.0.0.1:8000/docs
```

or:

```text
http://localhost:8000/docs
```

Do not open:

```text
http://0.0.0.0:8000
```

`0.0.0.0` is used by the server to listen on all network interfaces, but it is not the browser address.

## Health Check

Endpoint:

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "catalog_rows": 1000,
  "message": "Evalify AI API is running. Open /docs for Swagger UI.",
  "model_load_error": null
}
```

## API Endpoints

### 1. CV Score by Text

```http
POST /v1/cv-score
```

Scores a text-based CV against a selected job role.

Example request:

```json
{
  "resume_text": "Experienced software engineer with Java, SQL, Oracle Database, PL/SQL, Linux, and backend development experience.",
  "selected_role": "Software Engineer"
}
```

Example response structure:

```json
{
  "selected_role": "Software Engineer",
  "total_job_descriptions_used": 109,
  "fit_score_best": 87.64,
  "fit_score_average_top_k": 78.97,
  "structured_score_best": 53.0,
  "ranking_score_best": 76.83,
  "final_user_score": 72.53,
  "user_friendly_score": {
    "overall_score": 72.53,
    "fit_score": {
      "score": 78.97,
      "weight": 0.7,
      "source": "average_top_k"
    },
    "structured_score": {
      "score": 53.0,
      "weight": 0.25,
      "source": "best_evidence"
    },
    "profile_quality_score": {
      "score": 80.0,
      "weight": 0.05,
      "source": "profile_enrichment"
    }
  },
  "ai_explanation": {
    "enabled": true,
    "generated": true,
    "error": null,
    "summary": "...",
    "strengths": [],
    "gaps": [],
    "recommendations": [],
    "feedback": "..."
  },
  "top_matches": []
}
```

### 2. CV Score by PDF Upload

```http
POST /v1/cv-score/pdf
```

Form data:

```text
file: resume PDF
selected_role: target role
```

This endpoint extracts text from the uploaded PDF and scores it against the selected role.

### 3. Job Recommendation by Text

```http
POST /v1/recommend-jobs
```

Recommends the top job roles based on a text-based CV.

Example request:

```json
{
  "resume_text": "Software engineer experienced in Java, SQL, Oracle Database, PL/SQL, Linux, backend development, and database systems.",
  "role_hint": "software"
}
```

`role_hint` is optional and can be used to narrow recommendations to a specific role keyword.

### 4. Job Recommendation by PDF Upload

```http
POST /v1/recommend-jobs/pdf
```

Form data:

```text
file: resume PDF
role_hint: optional role keyword
```

This endpoint extracts text from the uploaded PDF and returns the top 10 job recommendations.

Each recommendation includes a per-job user-friendly score.

Example response structure:

```json
{
  "total_candidates_scored": 1000,
  "top_n": 10,
  "recommendations": [
    {
      "job_role": "Software Engineer",
      "job_description": "...",
      "user_match_score": 77.22,
      "user_friendly_score": {
        "overall_score": 77.22,
        "fit_score": {
          "score": 87.64,
          "weight": 0.7,
          "source": "job_model_fit"
        },
        "structured_score": {
          "score": 47.5,
          "weight": 0.25,
          "source": "job_evidence"
        },
        "profile_quality_score": {
          "score": 80.0,
          "weight": 0.05,
          "source": "profile_enrichment"
        }
      },
      "fit_score": 87.64,
      "structured_score": 47.5,
      "ranking_score": 76.83,
      "evidence": {
        "matched_skills": ["oracle database", "pl/sql", "sql"],
        "missing_skills": ["application development", "technical support"],
        "matched_tools": ["oracle database", "pl/sql", "sql"],
        "missing_tools": [],
        "matched_domains": [],
        "missing_domains": []
      }
    }
  ]
}
```

## Scoring Logic

### `fit_score`

Represents the supervised model prediction score.

```text
fit_score = match_probability * 100
```

This score comes from the trained TensorFlow/Keras model.

### `structured_score`

Represents rule-based evidence matching.

It is calculated from structured features such as:

- Skill match
- Tool match
- Domain match
- Experience match
- Education match
- Project/responsibility match
- Seniority match
- Certification match

### `profile_quality_score`

Represents a candidate profile completeness signal extracted from the CV.

This score is produced by the internal profile enrichment layer and is used with a small weight to avoid dominating the final score.

### `ranking_score`

An internal score used to sort job matches.

```text
ranking_score =
  (0.75 * fit_score + 0.25 * structured_score)
  * profile_multiplier
```

The profile multiplier is conservative and prevents profile enrichment from dominating the ranking.

### Selected Role User Score

For selected role scoring, the main user-facing score is:

```text
final_user_score =
  0.70 * fit_score_average_top_k
+ 0.25 * structured_score_best
+ 0.05 * profile_quality_score
```

### Top 10 Recommendation User Score

For top 10 job recommendations, each job has its own user-facing score:

```text
user_match_score =
  0.70 * fit_score
+ 0.25 * structured_score
+ 0.05 * profile_quality_score
```

## Response Evidence

Each scored job includes evidence:

```json
{
  "evidence": {
    "matched_skills": ["sql", "oracle database", "pl/sql"],
    "missing_skills": ["application development", "technical support"],
    "matched_tools": ["sql", "oracle database", "pl/sql"],
    "missing_tools": [],
    "matched_domains": [],
    "missing_domains": []
  }
}
```

This makes the score more explainable because users can see which parts of their CV matched the job description and which requirements are missing.

## Data Files

### `data/unique_job_role_descriptions_v5.csv`

Main job catalog used for scoring and recommendation.

Expected columns include:

```text
job_role
job_description
```

or compatible columns such as:

```text
Job Roles
Job Description
```

### `data/unique_job_role_descriptions_v5_structured_cache.csv`

Structured cache generated from the job catalog.

This file stores precomputed structured job features to make API startup and inference faster.

If the taxonomy or job catalog changes, regenerate this cache by deleting the file and restarting the API.

## Artifact Files

### `artifacts/evalify_custom_transformer_job_matching_v5_structured_features.keras`

Trained TensorFlow/Keras job matching model.

### `artifacts/feature_config_v5_structured_features.joblib`

Feature configuration used for numeric feature normalization.

### `artifacts/taxonomy.json`

Taxonomy used for structured feature extraction.

### `artifacts/ner_bilstm_attention_model.keras`

Candidate profile enrichment model.

### `artifacts/ner_vocab.json`

Vocabulary and tag mapping used by the profile enrichment model.

### `artifacts/metadata.json`

Additional metadata for the model or project.

## Training Notebook

The model training workflow is available in:

```text
notebooks/evalify_job_matching_custom_transformer_v5_structured_features_training.ipynb
```

The notebook includes the supervised learning pipeline for training the custom transformer-based job matching model with structured features.

The trained model artifact is exported to:

```text
artifacts/evalify_custom_transformer_job_matching_v5_structured_features.keras
```

The feature normalization configuration is exported to:

```text
artifacts/feature_config_v5_structured_features.joblib
```

## Example Usage with `curl`

### CV Score

Windows CMD:

```bash
curl -X POST "http://127.0.0.1:8000/v1/cv-score" ^
  -H "Content-Type: application/json" ^
  -d "{\"resume_text\":\"Experienced software engineer with Java, SQL, Oracle Database, PL/SQL, Linux, and backend development experience.\",\"selected_role\":\"Software Engineer\"}"
```

### Job Recommendation

Windows CMD:

```bash
curl -X POST "http://127.0.0.1:8000/v1/recommend-jobs" ^
  -H "Content-Type: application/json" ^
  -d "{\"resume_text\":\"Software engineer with Java, SQL, Oracle Database, PL/SQL, Linux, and backend development experience.\",\"role_hint\":\"software\"}"
```

### PDF CV Score

Windows CMD:

```bash
curl -X POST "http://127.0.0.1:8000/v1/cv-score/pdf" ^
  -F "file=@data/example_cv_web_developer.pdf" ^
  -F "selected_role=Software Engineer"
```

### PDF Job Recommendation

Windows CMD:

```bash
curl -X POST "http://127.0.0.1:8000/v1/recommend-jobs/pdf" ^
  -F "file=@data/example_cv_web_developer.pdf"
```

## Common Issues

### `ERR_ADDRESS_INVALID`

Do not open:

```text
http://0.0.0.0:8000
```

Use:

```text
http://127.0.0.1:8000/docs
```

or:

```text
http://localhost:8000/docs
```

### Slow Startup

The first startup may take time because TensorFlow loads the model artifacts.

Wait until the terminal shows:

```text
Application startup complete
```

### PDF Upload Returns 422

This usually happens because required form fields are missing.

For `/v1/cv-score/pdf`, make sure the request includes:

```text
file
selected_role
```

For `/v1/recommend-jobs/pdf`, only `file` is required.

When using Swagger UI, select the PDF file again before each new request.

### GenAI Explanation Not Generated

If `ai_explanation.generated` is `false`, check:

```text
GENAI_API_KEY
GENAI_MODEL_NAME
USE_GENAI_EXPLANATION_IF_AVAILABLE
```

GenAI explanation is optional. Scoring still works without it.

### TensorFlow Protobuf Error

If TensorFlow fails with protobuf compatibility errors, update protobuf:

```bash
pip install --upgrade protobuf
```

Recommended requirement:

```text
protobuf>=6.31.1
```

### Model File Not Downloaded After Clone

Run:

```bash
git lfs pull
```

## License

This project is created for educational and capstone project purposes.

## Author

Created by Andro.

Repository:

```text
https://github.com/Andro114/Job-Matching_Supervised-Learning
```
