"""
Single-Pass VLM Visual Evidence Extractor — Stage 1b of the financial
evidence-extraction pipeline.
Location: lib/python/legal_platform/financial_llm_extraction.py

Produces one whole-page visual reading of the rendered page image at a
single fixed DPI. Paired with financial_structural_extraction.py's PyMuPDF
structural reading (Stage 1a), this gives the Stage 2 text-LLM
reconstruction (financial_reconciliation.py) two independent evidence
sources instead of noisy repeats of the same VLM pass at different DPIs.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import fitz  # PyMuPDF
import dataiku

from .config import (
    FINANCIAL_PAGE_RENDER_DPI,
    FINANCIAL_REQUEST_MAX_ATTEMPTS,
    FINANCIAL_RETRY_DELAY_SECONDS,
    FINANCIAL_VLM_PRIMARY_ID,
)
from .llm import parse_json_object, strip_think

FINANCIAL_PAGE_VISUAL_READING_PROMPT = r"""
You are an expert forensic accountant analyzing documentary evidence in Saudi banking dispute cases.

The document page may contain BOTH genuine case transactions AND static institutional headers, legal text, or metadata footers.

Your task has two parts, in order:

PART 1 - TRIAGE: classify each piece of information on the page and discard institutional headers/metadata (rules below). Only genuine transaction rows move on to Part 2.

PART 2 - COMPLETE REWRITE: for every remaining genuine transaction row visible on this page, rewrite it as ONE complete line item with every column filled in.
   - Go row by row, top to bottom, exactly as printed. Output EVERY row - do not summarize, sample, skip, or truncate the table.
   - Fill EVERY field: row_index, date_exact_text, description, amount_exact_text, currency_exact_text, debit_or_credit, reference_number, party_source, running_balance_exact_text.
   - Leave field as "" ONLY if genuinely missing from that row.

TRIAGE CLASSIFICATION RULES:
1. DISCARD AS INSTITUTIONAL HEADERS / METADATA:
   - Bank Paid-Up Capital, Commercial Registration (C.R. No), Unified Number, VAT Number, P.O. Box, Phone.
   - Page header/footer barcodes, timestamps, system tracking IDs.
   - General Legal citation decrees and Committee Articles.

2. EXTRACT AS CASE FINANCIAL MOVEMENTS:
   - Wire transfers / Hawala receipts.
   - Binance P2P orders and crypto transfers.
   - POS purchases, mada debits, account credits.
   - Specific bank freeze or unfreeze holds for the customer.

RETURN JSON SCHEMA ONLY:
{
  "page_has_case_transactions": true,
  "line_items": [
    {
      "row_index": 0,
      "date_exact_text": "YYYY/MM/DD",
      "transaction_type": "transfer|fee|pos|salary|compensation|other",
      "description": "Clean description of the transaction",
      "amount_exact_text": "4092.44",
      "currency_exact_text": "SAR",
      "debit_or_credit": "debit|credit|unclear",
      "party_source": "",
      "reference_number": "",
      "running_balance_exact_text": "",
      "source_quote": ""
    }
  ],
  "stated_totals_or_balances": [],
  "extraction_uncertainty": []
}
""".strip()


def _extract_text(response: Any) -> str:
    if getattr(response, "success", None) is False:
        raise RuntimeError(str(getattr(response, "error_message", "The LLM request failed.")))
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("The completion returned no usable text.")
    return strip_think(text)


def _call_vlm(image_bytes: bytes, mime_type: str, page_number: int) -> dict:
    project = dataiku.api_client().get_default_project()
    llm = project.get_llm(FINANCIAL_VLM_PRIMARY_ID)
    completion = llm.new_completion()
    try:
        completion.settings["temperature"] = 0.0
    except Exception:
        pass

    completion.with_message(FINANCIAL_PAGE_VISUAL_READING_PROMPT, role="system")

    message = completion.new_multipart_message(role="user")
    message.with_text(
        f"Analyze page {page_number}: filter out header/footer metadata, "
        "then rewrite the COMPLETE set of genuine case transactions."
    )
    message.with_inline_image(image_bytes, mime_type=mime_type or "image/png")
    message.add()

    response = completion.execute()
    parsed = parse_json_object(_extract_text(response))

    if not parsed.get("page_has_case_transactions", True):
        parsed["line_items"] = []

    return parsed


def render_pdf_page_to_dpi(pdf_bytes: bytes, page_number: int, dpi: int) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc.load_page(page_number - 1)
        pixmap = page.get_pixmap(dpi=int(dpi), alpha=False)
        return pixmap.tobytes("png")
    finally:
        doc.close()


def extract_page_visual_evidence(
    image_bytes: Optional[bytes],
    mime_type: str,
    page_number: int,
    pdf_bytes: Optional[bytes] = None,
) -> tuple[dict, list[str]]:
    """One deterministic VLM pass over the page, rendered at a single fixed
    DPI when the original PDF is available (falls back to whatever page
    image is already stored otherwise). Retries only cover transient
    request failures — there is no multi-pass voting to feed anymore.
    """
    failures: list[str] = []

    page_bytes = image_bytes
    page_mime = mime_type or "image/png"
    if pdf_bytes:
        try:
            page_bytes = render_pdf_page_to_dpi(pdf_bytes, page_number, dpi=FINANCIAL_PAGE_RENDER_DPI)
            page_mime = "image/png"
        except Exception as error:
            failures.append(f"Page render at {FINANCIAL_PAGE_RENDER_DPI} DPI failed: {error!r}")

    if not page_bytes:
        failures.append("No page image available for visual extraction.")
        return {"line_items": []}, failures

    result: dict = {"line_items": []}
    for attempt in range(1, int(FINANCIAL_REQUEST_MAX_ATTEMPTS) + 1):
        try:
            result = _call_vlm(page_bytes, page_mime, page_number)
            break
        except Exception as error:
            failures.append(f"VLM visual pass attempt {attempt}: {error!r}")
            if attempt < int(FINANCIAL_REQUEST_MAX_ATTEMPTS):
                time.sleep(float(FINANCIAL_RETRY_DELAY_SECONDS))

    return result, failures
