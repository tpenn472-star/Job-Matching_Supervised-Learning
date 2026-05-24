import json
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"

resume = """
Fresh graduate in Computer Science with experience in web development, backend API,
data preprocessing, and machine learning prototypes. Skilled in Python, FastAPI,
REST API, SQL, Git, JavaScript, HTML, CSS, React, Node.js, Pandas, NumPy,
TensorFlow, and model inference. Built AI interview simulation backend services.
Bachelor of Computer Science. Seeking Web Developer or Backend Developer role.
"""

print("Health:")
print(requests.get(f"{BASE_URL}/health").json())

print("\nCV Score from plain text:")
print(json.dumps(requests.post(f"{BASE_URL}/v1/cv-score", json={
    "resume_text": resume,
    "selected_role": "Web Developer",
    "top_k": 5
}).json(), indent=2))

print("\nRecommendations from plain text:")
print(json.dumps(requests.post(f"{BASE_URL}/v1/recommend-jobs", json={
    "resume_text": resume,
    "top_n": 10
}).json(), indent=2))

pdf_path = Path("data/example_cv_web_developer.pdf")

print("\nCV Score from PDF:")
with pdf_path.open("rb") as f:
    response = requests.post(
        f"{BASE_URL}/v1/cv-score/pdf",
        files={"file": ("example_cv_web_developer.pdf", f, "application/pdf")},
        data={"selected_role": "Web Developer", "top_k": 5},
    )
print(json.dumps(response.json(), indent=2))

print("\nRecommendations from PDF:")
with pdf_path.open("rb") as f:
    response = requests.post(
        f"{BASE_URL}/v1/recommend-jobs/pdf",
        files={"file": ("example_cv_web_developer.pdf", f, "application/pdf")},
        data={"top_n": 10},
    )
print(json.dumps(response.json(), indent=2))
