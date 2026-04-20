from __future__ import annotations

from pathlib import Path


def extract_text_with_ocr(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    raise NotImplementedError(
        "OCR is not configured. Implement open_match_lca.data.ocr_adapter.extract_text_with_ocr "
        f"for {path} and rerun extract_aux_documents.py with --enable-ocr."
    )
