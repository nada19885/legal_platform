
"""
Pure VLM Multi-DPI Pipeline Orchestrator.
Location: lib/python/legal_platform/financial_extraction_pipeline.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from typing import Callable, Optional
from .storage import case_rows 

from .config import FINANCIAL_PAGE_MAX_WORKERS, FINANCIAL_LINE_ITEMS_DATASET  
from .financial_llm_extraction import extract_page_line_items_multi
from .financial_page_sources import (
    FinancialPageSource,
    load_document_pdf_bytes,
    load_financial_pages,
    load_page_image_bytes,
)
from .financial_reconciliation import persist_reconciled_rows, reconcile_page

ProgressCallback = Callable[[int, int, str], None]


def _already_extracted_page_ids(case_id: str) -> set[str]:              
    df = case_rows(FINANCIAL_LINE_ITEMS_DATASET, case_id)
    if df.empty or "page_id" not in df.columns:
        return set()
    return set(df["page_id"].astype(str).tolist())


def sanitize_extracted_line_items(raw_items: list[dict]) -> list[dict]:
    """Filters out empty or invalid rows before writing to the database."""
    clean_items = []
    for item in raw_items:
        try:
            fields = json.loads(item.get("fields_json", "{}") or "{}")
            raw_amt = str(fields.get("amount", {}).get("value", "")).strip()
        except Exception:
            raw_amt = ""

        if not raw_amt:
            continue

        numeric_amt = re.sub(r"[^\d.]", "", raw_amt)
        if not numeric_amt:
            continue

        try:
            val = float(numeric_amt)
            if val <= 0:
                continue
            clean_items.append(item)
        except (ValueError, TypeError):
            continue

    return clean_items


def _process_page(
    page: FinancialPageSource,
    pdf_bytes_by_document: dict[str, Optional[bytes]],
) -> tuple[list[dict], list[str]]:
    failures: list[str] = []
    pdf_bytes = pdf_bytes_by_document.get(page.case_document_id)

    image_bytes = None
    try:
        image_bytes = load_page_image_bytes(page)
        # 3 independent VLM passes (150, 200, 300 DPI)
        llm_passes, vlm_failures = extract_page_line_items_multi(
            image_bytes,
            page.page_image_mime_type,
            page.page_number,
            pdf_bytes=pdf_bytes,
            pass_count=3,
        )
        failures.extend(vlm_failures)
    except Exception as error:
        llm_passes = []
        failures.append(f"VLM Multi-DPI extraction failed: {error!r}")

    # Reconcile: 2 agrees = verified. 3 differ = VLM Arbitrator. If arbitrator not sure = user review.
    rows = reconcile_page(
        page.page_id,
        page.case_document_id,
        page.page_number,
        llm_passes,
        image_bytes=image_bytes,
        mime_type=page.page_image_mime_type,
    )

    cleaned_rows = sanitize_extracted_line_items(rows)
    return cleaned_rows, failures


def run_financial_extraction(
    case_id: str,
    page_ids: Optional[list[str]] = None,  # <--- Now accepts page_ids
    actor: str = "",
    progress_callback: Optional[ProgressCallback] = None,
    force_rerun: bool = False,
) -> dict:
    # Load all pages, then strictly filter to ONLY the requested financial/mixed pages
    all_pages = load_financial_pages(case_id, None)
    if page_ids is not None:
        requested = set(str(pid) for pid in page_ids)
        pages = [p for p in all_pages if str(p.page_id) in requested]
    else:
        pages = all_pages

    if not pages:
        raise ValueError("No extracted financial pages are available for this case.")
        
    if not force_rerun:
        already_done = _already_extracted_page_ids(case_id)
        pages = [p for p in pages if str(p.page_id) not in already_done]

    if not pages:
        existing = case_rows(FINANCIAL_LINE_ITEMS_DATASET, case_id).to_dict(orient="records")
        return {
            "case_id": case_id,
            "pages_processed": 0,
            "rows_persisted": 0,
            "verified": sum(1 for r in existing if r.get("row_status") == "verified"),
            "needs_review": sum(1 for r in existing if r.get("row_status") == "needs_review"),
            "single_source_low_confidence": sum(1 for r in existing if r.get("row_status") == "single_source_low_confidence"),
            "page_failures": [],
            "new_rows_appended": 0,
        }
        

    pdf_bytes_by_document: dict[str, Optional[bytes]] = {
        document_id: load_document_pdf_bytes(case_id, document_id)
        for document_id in {page.case_document_id for page in pages}
    }

    all_rows: list[dict] = []
    page_failures: list[dict] = []
    completed = 0
    worker_count = max(1, min(int(FINANCIAL_PAGE_MAX_WORKERS), len(pages)))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_page = {
            executor.submit(_process_page, page, pdf_bytes_by_document): page
            for page in pages
        }
        for future in as_completed(future_to_page):
            page = future_to_page[future]
            try:
                rows, failures = future.result()
            except Exception as error:
                rows, failures = [], [f"Page worker failed: {error!r}"]

            all_rows.extend(rows)
            if failures:
                page_failures.append({
                    "page_id": page.page_id,
                    "page_number": page.page_number,
                    "failures": failures,
                })

            completed += 1
            if progress_callback:
                progress_callback(completed, len(pages), page.page_id)

    persisted_count = persist_reconciled_rows(case_id, all_rows)
    verified_count = sum(1 for r in all_rows if r.get("row_status") == "verified")
    needs_review_count = sum(1 for r in all_rows if r.get("row_status") == "needs_review")
    low_conf_count = sum(1 for r in all_rows if r.get("row_status") == "single_source_low_confidence")

    return {
        "case_id": case_id,
        "pages_processed": len(pages),
        "rows_persisted": persisted_count if isinstance(persisted_count, int) else len(all_rows),
        "verified": verified_count,
        "needs_review": needs_review_count,
        "single_source_low_confidence": low_conf_count,
        "page_failures": page_failures,
        "new_rows_appended": len(all_rows),
    }


