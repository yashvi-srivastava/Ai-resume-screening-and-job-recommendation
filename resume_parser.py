"""
resume_parser.py
-----------------
Extracts raw text from resumes in PDF, DOCX, or TXT format.

Diagnostics are written with the standard `logging` module. Only counts and
exception type names are logged - never resume text or file names.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)


def clean_resume_text(text: str) -> str:
    """Normalize extracted resume text into ONE line (line breaks removed)."""
    return re.sub(r"\s+", " ", text).strip()


def clean_resume_layout(text: str) -> str:
    """Normalize whitespace inside each line but KEEP the line breaks.

    Section detection (Projects, Certifications, ...) needs line boundaries,
    so the app uses this instead of clean_resume_text().
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    lines = [re.sub(r"[ \t\u00a0\f\v]+", " ", line).strip() for line in text.split("\n")]
    # Drop blank lines; keep one line break between non-empty lines.
    return "\n".join(line for line in lines if line)


def parse_pdf(filepath: str) -> str:
    import pdfplumber

    text_chunks = []
    empty_pages = 0
    try:
        with pdfplumber.open(filepath) as pdf:
            page_count = len(pdf.pages)
            for number, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception as error:  # one bad page must not lose the rest
                    logger.warning(
                        "PDF page %d/%d could not be read (%s)",
                        number, page_count, type(error).__name__,
                    )
                    page_text = ""
                if not page_text.strip():
                    empty_pages += 1
                text_chunks.append(page_text)
    except Exception as error:
        logger.error("PDF could not be opened (%s)", type(error).__name__)
        raise ValueError(
            "The PDF could not be read. It may be corrupted or password-protected."
        ) from error

    logger.info(
        "PDF parsed: pages=%d, pages_without_text=%d, characters=%d",
        page_count, empty_pages, sum(len(chunk) for chunk in text_chunks),
    )
    return "\n".join(text_chunks)


def parse_docx(filepath: str) -> str:
    import docx
    doc = docx.Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs)


def parse_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_resume(filepath: str, preserve_layout: bool = False) -> str:
    """Dispatch to the right parser based on file extension.

    preserve_layout=False (default) returns single-line text, exactly as before.
    preserve_layout=True keeps line breaks so sections can be detected.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        parsed_text = parse_pdf(filepath)
    elif ext == ".docx":
        parsed_text = parse_docx(filepath)
    elif ext == ".txt":
        parsed_text = parse_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    cleaned_text = (
        clean_resume_layout(parsed_text) if preserve_layout else clean_resume_text(parsed_text)
    )
    if not cleaned_text:
        logger.warning("No extractable text found (file type %s)", ext)
        hint = " Scanned image PDFs need OCR and are not supported." if ext == ".pdf" else ""
        raise ValueError("The resume file contains no extractable text." + hint)
    return cleaned_text


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(parse_resume(sys.argv[1])[:500])
