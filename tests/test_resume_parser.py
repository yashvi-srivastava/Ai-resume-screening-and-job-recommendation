import pytest

from src.resume_parser import parse_resume


def test_parse_txt_cleans_extracted_text(tmp_path):
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("  Python\n\n  SQL  ", encoding="utf-8")

    assert parse_resume(str(resume_path)) == "Python SQL"


def test_parse_resume_rejects_empty_file(tmp_path):
    resume_path = tmp_path / "empty.txt"
    resume_path.write_text("\n  ", encoding="utf-8")

    with pytest.raises(ValueError, match="no extractable text"):
        parse_resume(str(resume_path))


def test_parse_resume_rejects_unsupported_file(tmp_path):
    resume_path = tmp_path / "resume.csv"
    resume_path.write_text("name", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_resume(str(resume_path))