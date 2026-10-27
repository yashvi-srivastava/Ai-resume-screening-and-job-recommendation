import pytest

from src.resume_parser import parse_resume
from src.skill_extractor import extract_profile, extract_sections


RESUME = """Riya Verma
Data Analyst with 2 years of experience in Python and SQL.
EDUCATION
B.Tech in Computer Science
PROJECTS
- Sales dashboard using Power BI
CERTIFICATIONS
- AWS Cloud Practitioner
SKILLS
Python, SQL
"""


def test_sections_end_at_next_recognised_heading():
    sections = extract_sections(RESUME)

    assert sections["projects"] == "- Sales dashboard using Power BI"
    assert sections["certifications"] == "- AWS Cloud Practitioner"


@pytest.mark.parametrize(
    "heading",
    ["Projects", "Academic Projects", "Personal Projects", "PROJECT EXPERIENCE", "Projects:"],
)
def test_project_heading_variants(heading):
    assert extract_sections(f"{heading}\nChat app in Flask\nSkills\nPython")["projects"] == "Chat app in Flask"


@pytest.mark.parametrize(
    "heading",
    ["Certifications", "Certificates", "Courses", "Courses & Certifications", "Licenses and Certifications"],
)
def test_certification_heading_variants(heading):
    assert extract_sections(f"{heading}\nAWS CCP\nEducation\nB.Tech")["certifications"] == "AWS CCP"


def test_parse_resume_can_preserve_line_breaks(tmp_path):
    path = tmp_path / "resume.txt"
    path.write_text(RESUME, encoding="utf-8")

    layout_text = parse_resume(str(path), preserve_layout=True)
    profile = extract_profile(layout_text)

    assert "\n" in layout_text
    assert "Sales dashboard" in profile["projects"]
    assert "AWS Cloud Practitioner" in profile["certifications"]
    # Default behaviour is unchanged: single-line text.
    assert "\n" not in parse_resume(str(path))


def test_heading_word_inside_a_sentence_is_not_a_heading():
    sections = extract_sections("I led several projects at work\nPython")

    assert sections == {"projects": "", "certifications": ""}


def test_skill_phrase_wrapped_over_two_lines_is_still_found():
    assert "machine learning" in extract_profile("Experienced in machine\nlearning")["skills"]
