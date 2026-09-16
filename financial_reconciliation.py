"""
Pure VLM Multi-DPI Reconciliation Engine with Autonomous VLM Arbitrator.
Location: lib/python/legal_platform/financial_reconciliation.py
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re
from typing import Optional

import dataiku

from .config import (
    FINANCIAL_LINE_ITEMS_DATASET,
    FINANCIAL_VLM_PRIMARY_ID,
)
from .ids import stable_id
from .llm import parse_json_object, strip_think
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

ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _to_ascii_digits(value: str) -> str:
    return str(value or "").translate(ARABIC_INDIC_DIGITS)


def _normalize_amount_for_comparison(exact_text: str) -> Optional[Decimal]:
    text = _to_ascii_digits(str(exact_text or "")).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"[^\d.,\-]", "", text)
    if not text:
        return None

    last_comma = text.rfind(",")
    last_dot = text.rfind(".")
    if last_comma > last_dot:
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", "")

    try:
        val = Decimal(text)
        return -abs(val) if negative or text.startswith("-") else val
    except (InvalidOperation, ValueError):
        return None


def _normalize_text_for_comparison(text: str) -> str:
    clean = _to_ascii_digits(str(text or "")).lower().strip()
    return re.sub(r"\s+", " ", clean)


ARBITRATOR_PROMPT = r"""
You are the Chief Forensic Auditor for Banque Saudi Fransi.
Three independent visual extraction passes of this document page produced conflicting candidate values for a specific transaction row.

CONFLICT DETAILS:
Row Index: {row_index}
Conflicting Field: "{field_name}"
Candidates extracted across the 3 DPI passes:
{candidates_json}

YOUR TASK:
Look carefully at the attached page image at this row's position and determine the true value printed on the document.

RETURN JSON ONLY:
{{
  "resolved_value": "exact value as clearly printed on the document",
  "is_sure": true,
  "confidence_reason": "Clear explanation of why this reading is definitively correct, or why it remains ambiguous/blurry"
}}

RULES:
- Set "is_sure": true ONLY if the characters/digits are crisp and unambiguously visible to you.
- Set "is_sure": false if the scan is smudged, cut off, or if you cannot be 100% certain between the candidates.
""".strip()


def _call_vlm_arbitrator(
    image_bytes: bytes,
    mime_type: str,
    row_index: int,
    field_name: str,
    candidates: list[dict],
) -> dict:
    """Sends the full page image and the 3 disagreeing reads to the VLM to arbitrate."""
    try:
        project = dataiku.api_client().get_default_project()
        llm = project.get_llm(FINANCIAL_VLM_PRIMARY_ID)
        completion = llm.new_completion()
        try:
            completion.settings["temperature"] = 0.0
        except Exception:
            pass

        prompt = ARBITRATOR_PROMPT.format(
            row_index=row_index,
            field_name=field_name,
            candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
        )
        completion.with_message(prompt, role="system")

        message = completion.new_multipart_message(role="user")
        message.with_text(f"Resolve conflict for row {row_index}, field '{field_name}'. Are you certain?")
        message.with_inline_image(image_bytes, mime_type=mime_type or "image/png")
        message.add()

        response = completion.execute()
        text = getattr(response, "text", "") or ""
        return parse_json_object(strip_think(text))
    except Exception as err:
        return {"resolved_value": "", "is_sure": False, "confidence_reason": f"Arbitrator call failed: {err!r}"}


@dataclass
class FieldObservation:
    value: str
    dpi: int
    pass_index: int


@dataclass
class ReconciledRow:
    page_id: str
    case_document_id: str
    page_number: int
    row_index: int
    fields: dict[str, list[FieldObservation]]


def _resolve_field_with_arbitrator(
    field_name: str,
    observations: list[FieldObservation],
    image_bytes: Optional[bytes],
    mime_type: str,
    row_index: int,
) -> dict:
    if not observations:
        return {"value": "", "status": "missing", "candidates": []}

    candidates = [{"value": o.value, "source": f"VLM_{o.dpi}DPI"} for o in observations]

    if len(observations) == 1:
        return {
            "value": observations[0].value,
            "status": "single_source_low_confidence",
            "candidates": candidates,
        }

    # Step 1: Try 2-out-of-3 majority vote first
    if field_name == "amount":
        norm_values = [_normalize_amount_for_comparison(o.value) for o in observations]
        counts = Counter(v for v in norm_values if v is not None)
        for val, count in counts.items():
            if count >= 2:
                idx = norm_values.index(val)
                return {"value": observations[idx].value, "status": "verified", "candidates": candidates}
    else:
        norm_texts = [_normalize_text_for_comparison(o.value) for o in observations]
        text_counts = Counter(t for t in norm_texts if t)
        for txt, count in text_counts.items():
            if count >= 2:
                idx = norm_texts.index(txt)
                return {"value": observations[idx].value, "status": "verified", "candidates": candidates}

    # Step 2: All 3 differ -> Call Arbitrator VLM with page image + candidate readings
    if image_bytes:
        arb_res = _call_vlm_arbitrator(image_bytes, mime_type, row_index, field_name, candidates)
        if arb_res.get("is_sure") is True and arb_res.get("resolved_value"):
            candidates.append({"value": str(arb_res["resolved_value"]), "source": "VLM_Arbitrator_Confirmed"})
            return {
                "value": str(arb_res["resolved_value"]),
                "status": "verified",
                "candidates": candidates,
                "arbitrator_note": arb_res.get("confidence_reason", ""),
            }

    # Step 3: Arbitrator is NOT sure -> route to user for manual selection
    return {
        "value": observations[0].value,
        "status": "conflict",
        "candidates": candidates,
    }


def reconcile_page(
    page_id: str,
    case_document_id: str,
    page_number: int,
    llm_passes: list[dict],
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/png",
) -> list[dict]:
    row_clusters: dict[int, ReconciledRow] = {}

    for pass_data in llm_passes:
        dpi = int(pass_data.get("_pass_dpi", 200))
        pass_idx = int(pass_data.get("_pass_index", 1))
        items = pass_data.get("line_items", []) or []

        for idx, item in enumerate(items):
            r_idx = int(item.get("row_index", idx))
            if r_idx not in row_clusters:
                row_clusters[r_idx] = ReconciledRow(
                    page_id=page_id,
                    case_document_id=case_document_id,
                    page_number=page_number,
                    row_index=r_idx,
                    fields={name: [] for name in FIELD_NAMES},
                )

            row = row_clusters[r_idx]
            for f in FIELD_NAMES:
                val = str(item.get(f"{f}_exact_text", "") or item.get(f, "") or "").strip()
                if val:
                    row.fields[f].append(FieldObservation(value=val, dpi=dpi, pass_index=pass_idx))

    reconciled_rows: list[dict] = []
    for r_idx in sorted(row_clusters.keys()):
        row = row_clusters[r_idx]
        resolved_fields = {
            f: _resolve_field_with_arbitrator(
                f, row.fields[f], image_bytes, mime_type, r_idx
            )
            for f in FIELD_NAMES
        }

        field_statuses = {
            resolved_fields[f]["status"]
            for f in FIELD_NAMES
            if resolved_fields[f]["status"] != "missing"
        }

        if "conflict" in field_statuses:
            row_status = "needs_review"
        elif field_statuses == {"verified"} or "verified" in field_statuses:
            row_status = "verified"
        else:
            row_status = "single_source_low_confidence"

        cluster_key = f"ROW_{page_number}_{r_idx}"
        row_id = stable_id("FLI", page_id, cluster_key)

        reconciled_rows.append({
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

    return reconciled_rows


def persist_reconciled_rows(case_id: str, rows: list[dict]) -> int:
    for row in rows:
        row["case_id"] = case_id
    return append_rows(FINANCIAL_LINE_ITEMS_DATASET, rows)




