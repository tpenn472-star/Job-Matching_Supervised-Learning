from io import BytesIO
from pypdf import PdfReader


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        raise ValueError("Uploaded PDF is empty.")

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:
        raise ValueError("Invalid or unreadable PDF file.") from exc

    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages_text.append(text)

    extracted_text = "\n\n".join(pages_text).strip()

    if not extracted_text:
        raise ValueError(
            "No text could be extracted from the PDF. "
            "If this is a scanned PDF, OCR is required."
        )

    return extracted_text
