# Job Matching Supervised Learning API

A FastAPI-based job matching service for scoring candidate CVs against selected job roles and recommending the most suitable job roles from a structured job catalog.

This project combines:
- CV/resume text extraction
- PDF resume upload support
- Taxonomy-based skill, tool, domain, education, and experience extraction
- Structured matching scores
- TensorFlow/Keras model-based inference
- Job role recommendation API
- Git LFS support for large machine learning artifacts

---

## Overview

This API is designed to support an AI-powered job matching workflow.

Given a candidate CV, the system can:

1. Score the CV against a selected job role.
2. Return the top matching job descriptions for that role.
3. Recommend the top job roles from the job catalog.
4. Explain matching evidence such as matched skills, missing skills, matched tools, and missing tools.
5. Accept both plain text CV input and PDF CV upload.

The API is built with FastAPI and can be tested directly through Swagger UI.

---

## Main Features

### 1. CV to Selected Role Scoring

Score a candidate CV against a specific target role.

Example use case:

> A candidate uploads a CV and selects "Data Analyst".  
> The API returns the best matching Data Analyst job descriptions and fit scores.

### 2. Top Job Recommendation

Recommend the most suitable job roles for a candidate based on their CV.

Example use case:

> A candidate uploads a CV without choosing a role.  
> The API ranks available jobs and returns the top recommended roles.

### 3. PDF Resume Upload

The API supports PDF resumes and extracts text automatically before scoring.

### 4. Skill and Tool Matching Evidence

Each result includes structured evidence:

- Matched skills
- Missing skills
- Matched tools
- Missing tools
- Matched domains
- Missing domains

### 5. TensorFlow/Keras Model Inference

The project supports inference using a trained TensorFlow/Keras model stored in the `artifacts/` directory.

---

## Project Structure

```text
Job-Matching_Supervised-Learning/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── extractor.py
│   ├── model_adapter.py
│   ├── pdf_utils.py
│   ├── schemas.py
│   ├── scoring.py
│   └── service.py
│
├── artifacts/
│   ├── evalify_custom_transformer_job_matching_v5_structured_features.keras
│   ├── feature_config_v5_structured_features.joblib
│   ├── metadata.json
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
├── example_client.py
├── main.py
├── run_api.py
├── requirements.txt
└── README.md

