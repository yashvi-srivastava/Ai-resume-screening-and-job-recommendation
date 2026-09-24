"""
skill_extractor.py
-------------------
Extracts a normalized set of skills, education level, and years of experience
from raw resume text using a curated skills vocabulary + regex heuristics.

This is a lightweight, dependency-free alternative to a full spaCy NER
pipeline, so the project runs anywhere without heavy model downloads.
You can swap this for spaCy's PhraseMatcher / a fine-tuned NER model later
(see README "Upgrading the NLP layer").
"""

import re

# ---------------------------------------------------------------------------
# 1. Skill vocabulary
#    In a real project, load this from a DB / O*NET / ESCO taxonomy instead
#    of hardcoding it. Keep it lowercase; matching is case-insensitive.
# ---------------------------------------------------------------------------
SKILL_VOCAB = {
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "r",
    "html", "css", "react", "angular", "vue", "node.js", "django", "flask",
    "fastapi", "spring boot", "machine learning", "deep learning",
    "natural language processing", "nlp", "computer vision", "pytorch",
    "tensorflow", "keras", "scikit-learn", "pandas", "numpy", "opencv",
    "data analysis", "data visualization", "power bi", "tableau", "excel",
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "git", "github",
    "rest api", "graphql", "microservices", "mongodb", "postgresql",
    "mysql", "redis", "kafka", "spark", "hadoop", "airflow", "linux",
    "agile", "scrum", "project management", "communication", "leadership",
    "artificial intelligence", "model deployment", "statistics", "a/b testing",
    "data warehousing", "etl", "terraform", "networking", "security",
    "monitoring", "mlops", "testing", "database design", "incident response",
    "risk management", "performance tuning", "natural language processing",
}

EDUCATION_LEVELS = {
    "phd": 5, "doctorate": 5,
    "master": 4, "m.tech": 4, "mtech": 4, "msc": 4, "mba": 4, "m.s.": 4,
    "bachelor": 3, "b.tech": 3, "btech": 3, "bsc": 3, "b.e.": 3, "be ": 3,
    "diploma": 2,
    "high school": 1,
}

_EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*years?\s+of\s+experience",
    r"experience\s*[:\-]?\s*(\d+)\+?\s*years?",
    r"(\d+)\+?\s*yrs?\s+experience",
]

_SECTION_NAMES = {
    "projects": ("projects", "project experience", "academic projects"),
    "certifications": ("certifications", "certificates", "licenses"),
}


def extract_skills(text: str) -> set:
    """Return the set of vocabulary skills found in `text`."""
    text_lower = text.lower()
    found = set()
    for skill in SKILL_VOCAB:
        # word-boundary-safe match; handles multi-word skills too
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def extract_education_level(text: str) -> int:
    """Return the highest education level score found (0 if none)."""
    text_lower = text.lower()
    level = 0
    for keyword, score in EDUCATION_LEVELS.items():
        if keyword in text_lower:
            level = max(level, score)
    return level


def extract_experience_years(text: str) -> float:
    """Best-effort extraction of total years of experience mentioned."""
    text_lower = text.lower()
    years_found = []
    for pattern in _EXPERIENCE_PATTERNS:
        for match in re.findall(pattern, text_lower):
            years_found.append(float(match))
    return max(years_found) if years_found else 0.0


def extract_sections(text: str) -> dict[str, str]:
    """Return text under recognizable projects and certifications headings."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections = {name: "" for name in _SECTION_NAMES}
    active_section = None

    for line in lines:
        heading = line.casefold().rstrip(":")
        matched_section = next(
            (
                section
                for section, names in _SECTION_NAMES.items()
                if heading in names
            ),
            None,
        )
        if matched_section:
            active_section = matched_section
            continue

        if active_section:
            sections[active_section] += f" {line}"

    return {name: value.strip() for name, value in sections.items()}


def extract_profile(text: str) -> dict:
    """Convenience wrapper returning a full structured profile."""
    profile = {
        "skills": extract_skills(text),
        "education_level": extract_education_level(text),
        "experience_years": extract_experience_years(text),
    }
    profile.update(extract_sections(text))
    return profile


if __name__ == "__main__":
    sample = """
    Data Scientist with 4 years of experience in Python, SQL, Machine Learning
    and NLP. B.Tech in Computer Science. Worked with TensorFlow and Pandas.
    """
    print(extract_profile(sample))
