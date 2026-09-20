"""
Text-LLM Reconstruction Engine — Stage 2 of the financial evidence-
extraction pipeline.
Location: lib/python/legal_platform/financial_reconciliation.py

Takes Stage 1a's PyMuPDF structural evidence and Stage 1b's single VLM
visual evidence for one page and asks a text LLM to reconcile them into
clean transaction rows, with a per-field certainty flag. This replaces the
old 2-of-3 majority-vote + VLM-arbitrator design: there is no longer a set
of noisy repeats of the same reading to vote on, only two independent
sources to cross-check against each other.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional

from .config import (
    FINANCIAL_EXTRACTION_LLM_ID,
    FINANCIAL_LINE_ITEMS_DATASET,
    FINANCIAL_REQUEST_MAX_ATTEMPTS,
    FINANCIAL_RETRY_DELAY_SECONDS,
)
from .ids import stable_id
from .llm import complete_json
from .storage import append_rows

FIELD_NAMES = (
    "date",
    "amount",
    "currency",
    "debit_or_credit",
    "description",
    "reference_number",
    "party_source",
    "running_balance",
)

RECONSTRUCTION_SYSTEM_PROMPT = r"""
You are a senior forensic accountant reconstructing one page of financial
evidence in a Saudi banking dispute case involving Banque Saudi Fransi (BSF).

You receive TWO independent readings of the same page:
- "structural_evidence": machine-extracted native PDF text, tables and
  headings (PyMuPDF). Exact and reliable wherever it is non-empty, but
  EMPTY on a scanned/image-only page — an empty value here just means "no
  structural evidence available", not a contradiction.
- "visual_evidence": one VLM's visual reading of the rendered page image
  (numbers, Arabic and English text, table layout, dates).

YOUR TASK:
1. Reconcile Arabic/English direction and correspondence between the two
   readings.
2. Reconstruct every genuine financial transaction row on the page, in
   printed order, with every column filled in.
3. For every field, decide whether you are CERTAIN of the value: certain
   only when the two sources agree, or when a single available source is
   crisp and unambiguous. Mark uncertain whenever the sources conflict, or
   the only available reading is blurry/ambiguous/cut off.
4. Ignore institutional headers/footers (bank paid-up capital, C.R. number,
   VAT number, P.O. Box, phone, barcodes, legal citations) — only genuine
   case transactions are rows.

RULES:
- Never invent a value neither source supports.
- "sources.pymupdf" / "sources.vlm" must be the literal text each source
  reported for that field, or null if that source had nothing for it.
- If both sources are silent on a field, its value is "" and certain is false.

RETURN JSON ONLY:
{
  "transactions": [
    {
      "row_index": 0,
      "date": {"value": "", "certain": true, "reason": "", "sources": {"pymupdf": null, "vlm": ""}},
      "description": {"value": "", "certain": true, "reason": "", "sources": {"pymupdf": null, "vlm": ""}},
      "amount": {"value": "", "certain": true, "reason": "", "sources": {"pymupdf": null, "vlm": ""}},
      "currency": {"value": "", "certain": true, "reason": "", "sources": {"pymupdf": null, "vlm": ""}},
      "debit_or_credit": {"value": "", "certain": true, "reason": "", "sources": {"pymupdf": null, "vlm": ""}},
      "reference_number": {"value": "", "certain": true, "reason": "", "sources": {"pymupdf": null, "vlm": ""}},
      "party_source": {"value": "", "certain": true, "reason": "", "sources": {"pymupdf": null, "vlm": ""}},
      "running_balance": {"value": "", "certain": true, "reason": "", "sources": {"pymupdf": null, "vlm": ""}}
    }
  ]
}
""".strip()


def reconstruct_page_transactions(
    structural_evidence: dict,
    visual_evidence: dict,
    page_number: int,
) -> dict:
    """Stage 2: one text-LLM call reconciling both evidence sources for a
    single page into structured transactions with a per-field certainty
    flag. Retries cover only transient request failures.
    """
    payload = {
        "page_number": page_number,
        "structural_evidence": structural_evidence,
        "visual_evidence": visual_evidence,
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, int(FINANCIAL_REQUEST_MAX_ATTEMPTS) + 1):
        try:
            return complete_json(
                system_prompt=RECONSTRUCTION_SYSTEM_PROMPT,
                user_payload=payload,
                llm_id=FINANCIAL_EXTRACTION_LLM_ID,
                temperature=0.0,
            )
        except Exception as error:
            last_error = error
            if attempt < int(FINANCIAL_REQUEST_MAX_ATTEMPTS):
                time.sleep(float(FINANCIAL_RETRY_DELAY_SECONDS))
    raise RuntimeError(f"Stage 2 reconstruction failed: {last_error!r}")


def build_rows_from_reconstruction(
    page_id: str,
    case_document_id: str,
    page_number: int,
    reconstruction: dict,
) -> list[dict]:
    """Converts Stage 2's transactions into the same row shape the rest of
    the pipeline (financial_normalizer, financial_corrections, financial_
    forensics) already expects: fields_json = {field: {value, status,
    candidates}}, row_status in {verified, needs_review}.
    """
    rows: list[dict] = []
    for idx, item in enumerate(reconstruction.get("transactions", []) or []):
        resolved_fields = {}
        for field_name in FIELD_NAMES:
            info = item.get(field_name) or {}
            value = str(info.get("value", "") or "").strip()
            certain = bool(info.get("certain", False))
            sources = info.get("sources") or {}
            candidates = [
                {"value": str(sources[key]), "source": key}
                for key in ("pymupdf", "vlm")
                if sources.get(key)
            ]

            if not value:
                status = "missing"
            elif certain:
                status = "verified"
            else:
                status = "conflict"

            resolved_fields[field_name] = {
                "value": value,
                "status": status,
                "candidates": candidates,
                "reason": "" if certain else str(info.get("reason", "") or ""),
            }

        field_statuses = {
            resolved_fields[f]["status"]
            for f in FIELD_NAMES
            if resolved_fields[f]["status"] != "missing"
        }
        row_status = "needs_review" if "conflict" in field_statuses else "verified"

        row_index = int(item.get("row_index", idx))
        cluster_key = f"ROW_{page_number}_{row_index}"
        row_id = stable_id("FLI", page_id, cluster_key)

        rows.append({
            "row_id": row_id,
            "page_id": page_id,
            "case_document_id": case_document_id,
            "page_number": page_number,
            "cluster_key": cluster_key,
            "fields_json": json.dumps(resolved_fields, ensure_ascii=False),
            "row_status": row_status,
            "has_conflict": row_status == "needs_review",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    return rows


def persist_reconciled_rows(case_id: str, rows: list[dict]) -> int:
    for row in rows:
        row["case_id"] = case_id
    return append_rows(FINANCIAL_LINE_ITEMS_DATASET, rows)
