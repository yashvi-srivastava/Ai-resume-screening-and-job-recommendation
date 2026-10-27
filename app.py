"""Streamlit interface for resume parsing and job recommendation."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("resume_app")

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_job_dataset  # noqa: E402
from recommender import recommend_jobs  # noqa: E402
from resume_parser import parse_resume  # noqa: E402
from skill_extractor import extract_profile  # noqa: E402


st.set_page_config(page_title="Resume Screening and Job Recommendation", layout="wide")
st.title("AI-Based Resume Screening and Job Recommendation")
st.write("Upload a resume to compare it with the Hugging Face jobs and local supplemental roles.")


@st.cache_data(show_spinner="Loading job dataset...")
def get_jobs():
    return load_job_dataset()


def parse_uploaded_resume(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()  # bytes of the upload (works on Streamlit Cloud)
    # Log only the type and size - never the file name or content.
    logger.info("Upload received: type=%s, size=%d bytes", suffix, len(file_bytes))
    if not file_bytes:
        raise ValueError("The uploaded file is empty (0 bytes). Please upload the resume again.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        temporary_file.write(file_bytes)
        temporary_path = temporary_file.name
    try:
        # preserve_layout keeps line breaks so Projects/Certifications can be found.
        return parse_resume(temporary_path, preserve_layout=True)
    finally:
        Path(temporary_path).unlink(missing_ok=True)


try:
    jobs_df = get_jobs()
except Exception as error:
    st.error(f"The job dataset could not be loaded: {error}")
    st.stop()

st.sidebar.header("Recommendation settings")
top_n = st.sidebar.slider("Number of recommendations", 1, min(20, len(jobs_df)), 5)
selected_categories = st.sidebar.multiselect(
    "Filter by category",
    options=sorted(jobs_df["category"].unique()),
    default=sorted(jobs_df["category"].unique()),
)

uploaded_file = st.file_uploader(
    "Upload a resume",
    type=["pdf", "docx", "txt"],
    help="Text-based PDF files are supported. Scanned image PDFs may not contain extractable text.",
)

if uploaded_file is None:
    st.info("Upload a PDF, DOCX, or TXT resume to begin.")
    st.stop()

try:
    resume_text = parse_uploaded_resume(uploaded_file)
except (OSError, ValueError) as error:
    st.error(f"Resume processing failed: {error}")
    st.stop()
except Exception as error:  # unexpected parser failure: show a clean message, log the type only
    logger.error("Unexpected resume parsing error (%s)", type(error).__name__)
    st.error("The resume could not be processed. Please try another file or export it again as PDF/DOCX.")
    st.stop()

profile = extract_profile(resume_text)
logger.info(
    "Profile extracted: characters=%d, lines=%d, skills=%d, projects_chars=%d, certifications_chars=%d",
    len(resume_text), resume_text.count("\n") + 1, len(profile["skills"]),
    len(profile["projects"]), len(profile["certifications"]),
)

st.subheader("Extracted Resume Information")
profile_left, profile_right = st.columns(2)
with profile_left:
    st.write("**Skills**")
    st.write(", ".join(sorted(profile["skills"])) or "No supported skills detected")
    st.write("**Education level**")
    st.write(profile["education_level"] or "Not detected")
    st.write("**Experience years**")
    st.write(profile["experience_years"] or "Not detected")
with profile_right:
    st.write("**Projects**")
    st.write(profile["projects"] or "Not detected")
    st.write("**Certifications**")
    st.write(profile["certifications"] or "Not detected")

with st.expander("View extracted resume text"):
    st.text_area("Resume text", resume_text, height=220, label_visibility="collapsed")

with st.expander("Parsing diagnostics (no resume content)"):
    st.write(f"Characters extracted: {len(resume_text)}")
    st.write(f"Lines extracted: {resume_text.count(chr(10)) + 1}")
    st.write(f"Projects section found: {'yes' if profile['projects'] else 'no'}")
    st.write(f"Certifications section found: {'yes' if profile['certifications'] else 'no'}")
    if not profile["projects"] and not profile["certifications"]:
        st.caption(
            "No Projects or Certifications heading was recognised. Recognised headings include "
            "Projects, Academic Projects, Certifications, Certificates and Courses."
        )

filtered_jobs = jobs_df[jobs_df["category"].isin(selected_categories)]
if filtered_jobs.empty:
    st.warning("No jobs are available for the selected categories.")
    st.stop()

results = recommend_jobs(resume_text, filtered_jobs, top_n=top_n)
st.subheader("Recommended Jobs")

for _, result in results.iterrows():
    title = f"{result['job_title']} | {result['category']} | {result['match_score']}% match"
    with st.expander(title, expanded=False):
        score_left, score_right = st.columns(2)
        score_left.metric("Text similarity", f"{result['text_similarity']}%")
        score_right.metric("Skill overlap", f"{result['skill_overlap']}%")

        st.write("**Matched skills**")
        st.write(", ".join(result["matched_skills"]) or "None detected")
        st.write("**Missing listed job skills**")
        st.write(", ".join(result["missing_skills"]) or "None")
        st.write(f"**Education requirement:** {result['education_status']}")
        st.write(f"**Experience requirement:** {result['experience_status']}")
        st.write(f"**Job ID:** {result['job_id']}")
        st.write("**Job description**")
        st.write(filtered_jobs.loc[filtered_jobs["job_id"] == result["job_id"], "job_description"].iloc[0])
