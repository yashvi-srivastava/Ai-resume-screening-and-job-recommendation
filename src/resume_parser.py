"""
resume_parser.py
-----------------
Extracts raw text from resumes in PDF, DOCX, or TXT format.
"""

import os
import re


def clean_resume_text(text: str) -> str:
    """Normalize extracted resume text while preserving its words."""
    return re.sub(r"\s+", " ", text).strip()


def parse_pdf(filepath: str) -> str:
    import pdfplumber
    text_chunks = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks)


def parse_docx(filepath: str) -> str:
    import docx
    doc = docx.Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs)


def parse_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_resume(filepath: str) -> str:
    """Dispatch to the right parser based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        parsed_text = parse_pdf(filepath)
    elif ext == ".docx":
        parsed_text = parse_docx(filepath)
    elif ext == ".txt":
        parsed_text = parse_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    cleaned_text = clean_resume_text(parsed_text)
    if not cleaned_text:
        raise ValueError("The resume file contains no extractable text")
    return cleaned_text


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(parse_resume(sys.argv[1])[:500])
