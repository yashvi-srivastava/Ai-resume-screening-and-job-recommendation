from src.skill_extractor import extract_profile


def test_extract_profile_reads_available_resume_information():
    resume_text = """
    Data Analyst with 2 years of experience in Python and SQL.
    B.Tech in Computer Science.
    Projects:
    Sales dashboard using Power BI.
    Certifications:
    AWS Cloud Practitioner.
    """

    profile = extract_profile(resume_text)

    assert {"python", "sql"}.issubset(profile["skills"])
    assert profile["education_level"] == 3
    assert profile["experience_years"] == 2.0
    assert "Sales dashboard" in profile["projects"]
    assert "AWS Cloud Practitioner" in profile["certifications"]


def test_extract_profile_leaves_missing_sections_empty():
    profile = extract_profile("Python developer with 1 year of experience.")

    assert profile["projects"] == ""
    assert profile["certifications"] == ""