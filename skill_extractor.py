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

# Headings that START a section we capture. Keys are the profile fields.
# Entries are stored in normalized form (see _normalize_heading): lowercase,
# "&" written as "and", no bullets or trailing colons.
_SECTION_NAMES = {
    "projects": {
        "projects", "project", "academic projects", "academic project",
        "personal projects", "personal project", "project experience",
        "project work", "key projects", "technical projects", "major projects",
        "minor projects", "mini projects", "selected projects",
        "projects undertaken", "projects and research",
    },
    "certifications": {
        "certifications", "certification", "certificates", "certificate",
        "licenses", "licences", "licenses and certifications",
        "certifications and licenses", "courses", "course",
        "courses and certifications", "certifications and courses",
        "online courses", "professional certifications",
        "training and certifications", "courses and training",
        "certifications and training",
    },
}

# Other common resume headings. They are not captured; they only END the
# section that is currently open.
_OTHER_HEADINGS = {
    "education", "educational qualifications", "academic qualifications",
    "academics", "qualifications", "education and training",
    "experience", "work experience", "professional experience",
    "employment history", "work history", "internships", "internship",
    "internship experience", "skills", "technical skills", "key skills",
    "core competencies", "skills and tools", "tools and technologies",
    "technologies", "summary", "professional summary", "career summary",
    "career objective", "objective", "profile", "professional profile",
    "about me", "achievements", "accomplishments", "awards",
    "awards and achievements", "honors and awards", "publications",
    "research", "research experience", "extracurricular activities",
    "extra curricular activities", "co curricular activities", "activities",
    "positions of responsibility", "volunteer experience", "volunteering",
    "interests", "hobbies", "hobbies and interests", "languages",
    "strengths", "references", "declaration", "personal details",
    "personal information", "contact", "contact information", "workshops",
}

_MAX_HEADING_WORDS = 6
_MAX_HEADING_CHARS = 60
_ALL_HEADINGS = sorted(
    set().union(*_SECTION_NAMES.values(), _OTHER_HEADINGS), key=len, reverse=True
)
# Used only when the text has no line breaks at all: "Projects:" / "Skills:"
# (with a colon) are treated as headings wherever they appear.
_INLINE_HEADING_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(h).replace(r"\ ", r"\s+") for h in _ALL_HEADINGS) + r")\s*:",
    re.IGNORECASE,
)


def extract_skills(text: str) -> set:
    """Return the set of vocabulary skills found in `text`."""
    # Collapse line breaks so a phrase wrapped over two lines
    # ("machine\nlearning") still matches.
    text_lower = re.sub(r"\s+", " ", text.lower())
    found = set()
    for skill in SKILL_VOCAB:
        # word-boundary-safe match; handles multi-word skills too
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def extract_education_level(text: str) -> int:
    """Return the highest education level score found (0 if none)."""
    text_lower = re.sub(r"\s+", " ", text.lower())
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


def _normalize_heading(text: str) -> str:
    """Lowercase, '&' -> 'and', drop bullets / trailing colons, collapse spaces."""
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"^[^a-z0-9]*\d+\s*[.)]\s+", "", text)  # "3. Education" -> "Education"
    text = re.sub(r"^[^a-z0-9]+", "", text)       # leading bullets, symbols
    text = re.sub(r"[^a-z0-9]+$", "", text)       # trailing colon, dash, dots
    text = re.sub(r"[^a-z0-9]+", " ", text)       # inner punctuation -> space
    return re.sub(r"\s+", " ", text).strip()


def _classify_line(line: str):
    """Return (kind, remainder) if `line` is a heading, otherwise (None, line).

    kind is a profile field ('projects', 'certifications') or 'other'.
    remainder is any text after "Heading:" on the same line.
    """
    candidates = [(line, "")]
    if ":" in line:
        head, _, rest = line.partition(":")
        candidates.insert(0, (head, rest.strip()))
    for head, rest in candidates:
        if len(head) > _MAX_HEADING_CHARS:
            continue
        key = _normalize_heading(head)
        if not key or len(key.split()) > _MAX_HEADING_WORDS:
            continue
        for section, names in _SECTION_NAMES.items():
            if key in names:
                return section, rest
        if key in _OTHER_HEADINGS:
            return "other", rest
    return None, line


def extract_sections(text: str) -> dict[str, str]:
    """Return text under recognizable projects and certifications headings.

    A section runs from its heading until the next recognised heading
    (including headings such as Education or Skills that are not captured).
    """
    if "\n" not in text:
        # No line breaks at all (e.g. text flattened upstream): put each
        # "Heading:" on its own line so the normal logic can work.
        text = _INLINE_HEADING_RE.sub(lambda m: "\n" + m.group(0), text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections = {name: "" for name in _SECTION_NAMES}
    active_section = None

    for line in lines:
        kind, remainder = _classify_line(line)
        if kind is not None:
            active_section = kind if kind in sections else None
            if active_section and remainder:
                sections[active_section] += f" {remainder}"
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
