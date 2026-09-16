

import io
import re
import json
import math
import mimetypes
import uuid
import base64
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from flask import request, jsonify, send_file, Response

# The Flask `app` object is provided by the Dataiku Standard WebApp runtime.
from dataiku.customwebapp import *  # noqa: F401,F403  -> exposes `app`

### Added by Youssif for Monitoring Purposes ###
import os
import sys
sys.path.insert(0, "/dataiku/design/plugins/dev/usage-analytics-lib/python-lib")
from usageanalyticslib import write_usage_event
 
PROJECT_KEY = os.environ.get("DKU_CURRENT_PROJECT_KEY", "unknown_project")
 
def increment_usage(session_id, field, amount=1):
    if session_id:
        write_usage_event(PROJECT_KEY, session_id, field, amount)
### END ###



# ------------------------------------------------------------
# legal_platform integrations — identical imports to legal_ui.py.
# The webapp backend runs in the same Dataiku code environment, so the
# package imports resolve exactly as before.
# ------------------------------------------------------------
from legal_platform.ids import random_id
from legal_platform.intake import create_case, add_message
from legal_platform.storage import list_cases, latest_case, case_rows
from legal_platform.files import save_upload
from legal_platform.extraction import extract_pdf_page_by_page
from legal_platform.case_mapping import build_case_map
from legal_platform.case_map_storage import persist_case_map
from legal_platform.facts import add_fact_candidate
from legal_platform.chat_orchestrator import assess_next_step, run_legal_analysis, run_defence_plan
from legal_platform.interview_state import persist_interview_state
from legal_platform.case_summary import generate_case_summary, approve_case_summary
from legal_platform.attorney_workbench import generate_bilingual_memo, answer_case_question, revise_bilingual_pleading
from legal_platform.audit import audit
from legal_platform.agreement_workbench import (
    AGREEMENT_TYPES, RELATIONSHIP_TYPES, classify_agreement, extract_clause_map,
    run_agreement_review, save_agreement_state, load_agreement_states,
    discuss_agreement,
)

# ------------------------------------------------------------
# Accounting & forensic financial analysis — identical imports to the
# Streamlit "Accounting analysis" tab (legal_ui.py). Same package, same
# functions; only the calling convention (Flask route + job) differs.
# ------------------------------------------------------------
from legal_platform.config import (
    FINANCIAL_DISCREPANCIES_DATASET,
    FINANCIAL_DOCUMENT_CLASSIFICATION_DATASET,
    FINANCIAL_FINDINGS_DATASET,
    FINANCIAL_LINE_ITEM_CORRECTIONS_DATASET,
    FINANCIAL_LINE_ITEMS_DATASET,
    FINANCIAL_TIMELINE_DATASET,
)
from legal_platform.financial_classification import classify_case_documents
from legal_platform.financial_corrections import (
    list_rows_needing_review,
    load_latest_corrections,
    submit_correction,
)
from legal_platform.financial_extraction_pipeline import run_financial_extraction
from legal_platform.financial_forensics import (
    build_and_save_financial_timeline,
    load_saved_forensic_results,
    run_discrepancy_and_findings_analysis,
)
from legal_platform.financial_normalizer import normalize_row_for_ledger

APP_ASSETS_FOLDER_ID = "qx2RWzgX"
APP_LOGO_PATH = "logo.png"

ARABIC_RE = re.compile(r"[\u0600-\u06ff]")


# ============================================================
# JSON-safe serialisation of pandas frames / numpy scalars
# ============================================================
def _clean_scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


def _json_safe(value):
    """Recursively coerce a payload into strictly valid JSON.

    Python's json module emits bare NaN / Infinity tokens, which are legal
    Python but illegal JSON, so the browser rejects the whole response;
    numpy scalars fail serialisation outright. Both arrive through
    pandas-derived records and through workflow payloads that were stored
    with those values already in them. _clean_scalar() covers only the
    top level of the case row, so nested structures need this.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if hasattr(value, "item"):
        # numpy scalars -> their Python equivalents
        try:
            return _json_safe(value.item())
        except (AttributeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _ok(payload):
    """Success responses go out through the sanitiser."""
    return jsonify(_json_safe(payload))


def frame_to_records(frame):
    """Serialise a DataFrame to a list of plain dicts (NaN -> "")."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    return json.loads(frame.fillna("").to_json(orient="records", force_ascii=False))


def _row_dict(frame_row):
    return {k: _clean_scalar(v) for k, v in frame_row.items()}


# ============================================================
# Accounting & forensic analysis helpers — same calls as the Streamlit
# accounting tab, reshaped for a JSON response instead of st.dataframe /
# st.expander widgets.
# ============================================================
def _build_normalized_ledger(case_id, line_items_df):
    """Normalize every extracted line item against the latest attorney
    corrections. Identical sequence to legal_ui.py's ledger build."""
    if line_items_df is None or line_items_df.empty:
        return []
    corrections = load_latest_corrections(case_id)
    ledger = []
    for _, row in line_items_df.iterrows():
        item = normalize_row_for_ledger(row.to_dict(), corrections)
        if item is not None:
            ledger.append(item)
    return ledger


def _serialise_conflicts(case_id):
    """Rows still needing manual verification, with each conflicted
    field's candidate readings unpacked from fields_json so the frontend
    never has to parse JSON-in-JSON."""
    rows = list_rows_needing_review(case_id) or []
    out = []
    for row in rows:
        row_id = str(row.get("row_id", ""))
        try:
            fields = json.loads(row.get("fields_json", "{}") or "{}")
        except (TypeError, json.JSONDecodeError):
            fields = {}
        conflicted_fields = []
        if isinstance(fields, dict):
            for field_name, field_data in fields.items():
                if isinstance(field_data, dict) and field_data.get("status") == "conflict":
                    candidates = [
                        {"value": c.get("value"), "source": c.get("source")}
                        for c in (field_data.get("candidates") or [])
                    ]
                    conflicted_fields.append({"field": field_name, "candidates": candidates})
        if conflicted_fields:
            out.append({
                "row_id": row_id,
                "page_number": row.get("page_number"),
                "fields": conflicted_fields,
            })
    return out


# ============================================================
# Case data loading — identical query set to legal_ui.load_case_data()
# ============================================================
def load_case_data(case_id):
    data = {
        "case": latest_case(case_id) or {"case_id": case_id},
        "messages": case_rows("case_messages", case_id),
        "documents": case_rows("case_documents", case_id),
        "pages": case_rows("case_document_pages", case_id),
        "facts": case_rows("case_facts", case_id),
        "fact_candidates": case_rows("case_fact_candidates", case_id),
        "parties": case_rows("case_parties", case_id),
        "events": case_rows("case_events", case_id),
        "issues": case_rows("case_issues", case_id),
        "issue_candidates": case_rows("case_issue_candidates", case_id),
        "evidence": case_rows("case_evidence", case_id),
        "approvals": case_rows("case_approvals", case_id),
        "audit_events": case_rows("audit_events", case_id),
        "financial_line_items": case_rows(FINANCIAL_LINE_ITEMS_DATASET, case_id),
    }
    forensic = load_saved_forensic_results(case_id)
    data["financial_timeline"] = forensic.get("timeline", [])
    data["forensic_findings"] = forensic.get("accounting_findings", {})
    data["discrepancies"] = forensic.get("discrepancies", [])
    data["cross_check_summary"] = forensic.get("cross_check_summary", {})
    return data


def best(data, approved_key, candidate_key):
    return data[approved_key] if not data[approved_key].empty else data[candidate_key]


def _frame_ids(frame, candidates):
    if frame is None or frame.empty:
        return []
    for column in candidates:
        if column in frame.columns:
            return sorted(frame[column].fillna("").astype(str).tolist())
    return [str(len(frame))]


def case_fingerprint(data):
    """Unchanged from legal_ui.py — used to detect record changes after a
    completed workflow state so earlier work is invalidated correctly."""
    return {
        "documents": _frame_ids(data.get("documents"), ["case_document_id", "document_id"]),
        "pages": _frame_ids(data.get("pages"), ["case_document_page_id", "page_id"]),
        "evidence": _frame_ids(data.get("evidence"), ["case_evidence_id", "evidence_id"]),
        "facts_count": len(best(data, "facts", "fact_candidates")),
        "issues_count": len(best(data, "issues", "issue_candidates")),
    }


# ============================================================
# Server-side workflow state
#
# The Streamlit app kept these keys in st.session_state and flushed them
# to the audit log via persist_workflow_state(). Here we reconstruct the
# same dict from the audit log on every request (restore_workflow_state)
# and write it back the same way (persist_workflow_state). This is the
# "reuse existing server persistence" strategy — no new store.
# ============================================================
WORKFLOW_KEYS = (
    "attorney_summary", "summary_approved", "summary_approved_by", "research",
    "analysis", "strategy", "memo", "pleading_versions", "pleading_status",
    "pleading_finalised_by", "case_dirty", "dirty_reason",
)


def blank_workflow_state():
    return {
        "attorney_summary": None,
        "summary_approved": False,
        "summary_approved_by": "",
        "research": None,
        "analysis": None,
        "strategy": None,
        "memo": None,
        "pleading_versions": [],
        "pleading_status": "draft",
        "pleading_finalised_by": "",
        "case_dirty": False,
        "dirty_reason": "",
    }


def restore_workflow_state(data):
    """Port of legal_ui.restore_workflow_state — reads the latest
    case_workflow audit payload, then applies the fingerprint dirty check.
    Also restores an approved summary from the case_summary audit /
    approvals tables, mirroring the original main-body restoration."""
    state = blank_workflow_state()

    audits = data.get("audit_events")
    if isinstance(audits, pd.DataFrame) and not audits.empty and "entity_type" in audits.columns:
        states = audits[audits["entity_type"].astype(str) == "case_workflow"].copy()
        if not states.empty:
            if "event_at" in states.columns:
                states = states.sort_values("event_at")
            row = states.iloc[-1].to_dict()
            try:
                payload = json.loads(str(row.get("new_value_json", "{}") or "{}"))
            except (TypeError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, dict):
                for key in WORKFLOW_KEYS:
                    if key in payload:
                        state[key] = payload[key]
                saved_fingerprint = payload.get("fingerprint") or {}
                current_fingerprint = case_fingerprint(data)
                if saved_fingerprint and saved_fingerprint != current_fingerprint:
                    state["case_dirty"] = True
                    state["dirty_reason"] = "The case record changed after the last completed workflow state."
                    state["summary_approved"] = False

    # Restore latest approved summary (main-body logic in legal_ui.py).
    if state["attorney_summary"] is None and isinstance(audits, pd.DataFrame) and not audits.empty:
        summary_audits = audits.copy()
        if "entity_type" in summary_audits.columns and "action" in summary_audits.columns:
            summary_audits = summary_audits[
                (summary_audits["entity_type"].astype(str) == "case_summary")
                & (summary_audits["action"].astype(str) == "approved")
            ]
            if not summary_audits.empty:
                latest_audit = summary_audits.iloc[-1].to_dict()
                try:
                    restored = json.loads(str(latest_audit.get("new_value_json", "{}") or "{}"))
                    if isinstance(restored, dict) and restored:
                        state["attorney_summary"] = restored
                        state["summary_approved"] = True
                        state["summary_approved_by"] = str(latest_audit.get("actor", "") or "")
                except (TypeError, json.JSONDecodeError):
                    pass

    approvals = data.get("approvals")
    if isinstance(approvals, pd.DataFrame) and not approvals.empty and "approval_type" in approvals.columns:
        summary_approvals = approvals[approvals["approval_type"].astype(str) == "case_summary"]
        if not summary_approvals.empty:
            latest_approval = summary_approvals.iloc[-1].to_dict()
            state["summary_approved"] = str(latest_approval.get("decision", "")).lower() == "approved"
            state["summary_approved_by"] = str(latest_approval.get("approved_by", "") or "")

    return state


def workflow_payload(state, data):
    payload = {key: state.get(key) for key in WORKFLOW_KEYS}
    payload["summary_approved"] = bool(state.get("summary_approved"))
    payload["case_dirty"] = bool(state.get("case_dirty"))
    payload["fingerprint"] = case_fingerprint(data)
    return payload


def persist_workflow_state(state, data, case_id, action="saved", actor="system", reason=""):
    """Identical to legal_ui.persist_workflow_state — one audit write."""
    audit(
        case_id,
        "case_workflow",
        case_id,
        action,
        actor=actor,
        new_value=workflow_payload(state, data),
        reason=reason,
    )


def invalidate_for_new_material(state, data, case_id, reason):
    """Port of legal_ui.invalidate_for_new_material."""
    state["case_dirty"] = True
    state["dirty_reason"] = reason
    state["attorney_summary"] = None
    state["summary_approved"] = False
    state["summary_approved_by"] = ""
    state["research"] = None
    state["analysis"] = None
    state["strategy"] = None
    state["memo"] = None
    state["pleading_versions"] = []
    state["pleading_status"] = "draft"
    state["pleading_finalised_by"] = ""
    persist_workflow_state(state, data, case_id, action="invalidated", reason=reason)


# ============================================================
# Workflow steps + next action (ported from legal_ui.py, without the
# translation calls — the frontend supplies the localised labels from
# the step "key"). The state machine itself is unchanged.
# ============================================================
def workflow_steps(state, data):
    has_documents = not data["documents"].empty and not data["pages"].empty
    has_summary = state["attorney_summary"] is not None
    approved = bool(state["summary_approved"]) and not state["case_dirty"]
    has_analysis = state["analysis"] is not None and state["research"] is not None
    has_pleading = state["memo"] is not None
    is_final = has_pleading and state["pleading_status"] == "final"

    states = [
        {"key": "documents", "done": has_documents},
        {"key": "summary", "done": approved, "started": has_summary},
        {"key": "analysis", "done": has_analysis},
        {"key": "pleading", "done": has_pleading},
        {"key": "final", "done": is_final},
    ]
    current_found = False
    for step in states:
        if step["done"]:
            step["state"] = "complete"
        elif not current_found:
            step["state"] = "current"
            current_found = True
        else:
            step["state"] = "upcoming"
    if is_final:
        for step in states:
            step["state"] = "complete"
    return states


def next_action_key(state, data):
    steps = workflow_steps(state, data)
    current = next((item for item in steps if item["state"] == "current"), steps[-1])
    return current["key"]


# ============================================================
# Ordering + page-reference helpers (ported verbatim from legal_ui.py)
# ============================================================
def _party_rank(role):
    text = str(role or "").lower()
    priorities = [
        (0, ("claimant", "plaintiff", "applicant", "مدعي", "طالب")),
        (1, ("defendant", "respondent", "مدعى عليه", "مدعى عليها")),
        (2, ("appellant", "مستأنف")),
        (3, ("bank", "بنك", "مصرف")),
        (4, ("witness", "شاهد")),
        (5, ("expert", "خبير")),
        (6, ("authority", "جهة", "نيابة", "شرطة")),
    ]
    for rank, words in priorities:
        if any(word in text for word in words):
            return rank
    return 99


def _ordered_parties(items):
    return sorted(
        list(items or []),
        key=lambda item: (_party_rank(item.get("role")), str(item.get("name", "")).lower()),
    )


def _chronology_key(item):
    raw = str(item.get("date", "") or "").strip()
    if not raw or str(item.get("date_precision", "")).lower() == "unknown":
        return (1, "9999-99-99", raw)
    normal = raw.replace("/", "-")
    return (0, normal, raw)


def _ordered_chronology(items):
    return sorted(list(items or []), key=_chronology_key)


def _page_reference_map(data):
    documents = {}
    doc_frame = data.get("documents")
    if isinstance(doc_frame, pd.DataFrame) and not doc_frame.empty:
        for _, row in doc_frame.fillna("").iterrows():
            doc_id = str(row.get("case_document_id", "") or row.get("document_id", "") or "")
            filename = str(row.get("original_filename", "") or row.get("file_name", "") or row.get("filename", "") or "Document")
            if doc_id:
                documents[doc_id] = filename

    mapping = {}
    page_frame = data.get("pages")
    if isinstance(page_frame, pd.DataFrame) and not page_frame.empty:
        for _, row in page_frame.fillna("").iterrows():
            page_id = str(row.get("case_document_page_id", "") or row.get("page_id", "") or "")
            doc_id = str(row.get("case_document_id", "") or row.get("document_id", "") or "")
            page_number = row.get("page_number", "")
            filename = documents.get(doc_id, str(row.get("original_filename", "") or "Document"))
            label = f"{filename} — page {page_number}" if page_number != "" else filename
            if page_id:
                mapping[page_id] = label
    return mapping


def _page_labels(source_ids, data):
    mapping = _page_reference_map(data)
    labels = []
    for source_id in source_ids or []:
        value = str(source_id or "").strip()
        if not value:
            continue
        label = mapping.get(value, "Original page reference unavailable")
        if label not in labels:
            labels.append(label)
    return labels


def short_case_reference(case_id):
    raw = re.sub(r"[^A-Za-z0-9]", "", str(case_id or ""))
    suffix = raw[-6:].upper() if raw else "------"
    return f"CASE-{suffix}"


def first_case_value(row, *columns, default="—"):
    for column in columns:
        value = row.get(column, "")
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip()
    return default


def case_search_frame(cases, query):
    if cases.empty or not query.strip():
        return cases
    needle = query.strip().casefold()
    searchable = [
        "case_name", "client_name", "customer_name", "matter_type",
        "responsible_attorney", "created_by", "case_id", "case_status",
    ]
    mask = pd.Series(False, index=cases.index)
    for column in searchable:
        if column in cases.columns:
            mask = mask | cases[column].fillna("").astype(str).str.casefold().str.contains(needle, regex=False)
    if "case_id" in cases.columns:
        references = cases["case_id"].fillna("").astype(str).map(short_case_reference).str.casefold()
        mask = mask | references.str.contains(needle, regex=False)
    return cases[mask]


def filter_cases_by_workflow(cases, workflow_type):
    if cases.empty:
        return cases
    frame = cases.copy()
    if "workflow_type" not in frame.columns:
        return frame if workflow_type == "litigation" else frame.iloc[0:0]
    values = frame["workflow_type"].fillna("").astype(str).str.strip().str.lower()
    if workflow_type == "litigation":
        return frame[(values == "") | (values == "litigation")]
    return frame[values == "agreement_review"]


def serialise_case_card(item, workflow_type):
    case_id = str(item.get("case_id", ""))
    activity = first_case_value(item, "updated_at", "created_at", default="—")
    if "T" in activity:
        activity = activity.replace("T", " ")[:16]
    status = first_case_value(item, "workflow_status", "case_status", default="")
    return {
        "case_id": case_id,
        "reference": short_case_reference(case_id),
        "title": first_case_value(item, "case_name", "client_name", "customer_name", default=""),
        "matter": first_case_value(item, "matter_type", default=""),
        "attorney": first_case_value(item, "responsible_attorney", "created_by", default=""),
        "activity": activity,
        "status": status,
        "workflow_type": workflow_type,
    }


# ============================================================
# Pleading export — memo_to_markdown + build_pleading_docx_bytes
# ported verbatim so the .md / .docx output is byte-identical.
# ============================================================
def memo_to_markdown(memo, language):
    section = memo.get("pleading_ar" if language == "ar" else "pleading_en", {})
    title = memo.get("title_ar" if language == "ar" else "title_en", "")
    filing_type = memo.get("filing_type_ar" if language == "ar" else "filing_type_en", "")
    is_ar = language == "ar"
    lines = []

    if is_ar and section.get("basmala"):
        lines.append(section.get("basmala"))
    if filing_type:
        lines.append(f"# {filing_type}")
    if title:
        lines.append(f"## {title}")

    for key in ("court_heading", "case_details", "party_heading", "subject", "formal_salutation", "opening"):
        value = section.get(key, "")
        if value:
            lines.append(str(value))

    facts = section.get("facts", [])
    if facts:
        lines.append("## أولاً: الوقائع" if is_ar else "## I. Statement of Facts")
        if isinstance(facts, str):
            lines.append(facts)
        else:
            ordered = sorted(facts, key=lambda x: int(x.get("sequence", 999999) or 999999))
            for item in ordered:
                source = ""
                if item.get("source_document"):
                    source = str(item.get("source_document"))
                    if item.get("source_page"):
                        source += f" — {'الصفحة' if is_ar else 'page'} {item.get('source_page')}"
                fact_text = item.get("fact", "")
                date = item.get("date", "")
                prefix = f"**{date}:** " if date else ""
                lines.append(f"{item.get('sequence', '')}. {prefix}{fact_text}".strip())
                if source:
                    lines.append(f"*{'المصدر' if is_ar else 'Source'}: {source}*")

    sections = [
        ("ثانياً: الدفوع الشكلية والإجرائية" if is_ar else "II. Procedural Defences", "procedural_defences"),
        ("ثالثاً: الدفوع الموضوعية" if is_ar else "III. Substantive Defences", "substantive_defences"),
    ]
    for heading, key in sections:
        items = section.get(key, [])
        if items:
            lines.append(f"## {heading}")
            for idx, item in enumerate(items, 1):
                lines.append(f"### {idx}. {item.get('heading', '')}")
                if item.get("supported_fact"):
                    lines.append(f"**{'الواقعة المستند إليها' if is_ar else 'Supported fact'}:** {item.get('supported_fact')}")
                if item.get("authority"):
                    lines.append(f"**{'السند النظامي' if is_ar else 'Authority'}:** {item.get('authority')}")
                if item.get("application"):
                    lines.append(f"**{'التطبيق' if is_ar else 'Application'}:** {item.get('application')}")
                if item.get("requested_consequence"):
                    lines.append(f"**{'الأثر المطلوب' if is_ar else 'Requested consequence'}:** {item.get('requested_consequence')}")

    rebuttals = section.get("response_to_opponent", [])
    if rebuttals:
        lines.append("## رابعاً: الرد على ادعاءات الخصم" if is_ar else "## IV. Response to the Opposing Party")
        for idx, item in enumerate(rebuttals, 1):
            lines.append(f"### {idx}. {item.get('allegation', '')}")
            lines.append(item.get("response", ""))

    requests = section.get("requests", [])
    if requests:
        lines.append("## خامساً: الطلبات" if is_ar else "## V. Relief Requested")
        for index, item in enumerate(requests, 1):
            if isinstance(item, dict):
                text = item.get("text", "")
                basis = item.get("basis", "")
                lines.append(f"{index}. {text}")
                if basis:
                    lines.append(f"   *{'الأساس' if is_ar else 'Basis'}: {basis}*")
            else:
                lines.append(f"{index}. {item}")

    reservations = section.get("evidence_reservations", []) or section.get("reservations", [])
    if reservations:
        lines.append("## التحفظات المتعلقة بالأدلة" if is_ar else "## Evidence Reservations")
        lines.extend(f"- {item}" for item in reservations)

    closing = section.get("closing", "")
    if closing:
        lines.extend(["## الختام" if is_ar else "## Closing", closing])
    signature = section.get("signature_block", "")
    if signature:
        lines.append(signature)
    return "\n\n".join(str(item) for item in lines if item)


def build_pleading_docx_bytes(memo, case_reference=""):
    from io import BytesIO
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement

    def _set_rtl(paragraph):
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_pr = paragraph._p.get_or_add_pPr()
        bidi = OxmlElement("w:bidi")
        p_pr.append(bidi)

    def _add_markdown_text(document, markdown_text, rtl=False):
        for raw_line in str(markdown_text or "").split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("### "):
                paragraph = document.add_heading(line[4:], level=3)
            elif line.startswith("## "):
                paragraph = document.add_heading(line[3:], level=2)
            elif line.startswith("# "):
                paragraph = document.add_heading(line[2:], level=1)
            else:
                paragraph = document.add_paragraph(line)
            if rtl:
                _set_rtl(paragraph)

    document = Document()
    if case_reference:
        title_paragraph = document.add_paragraph(case_reference)
        title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _add_markdown_text(document, memo_to_markdown(memo, "ar"), rtl=True)
    document.add_page_break()
    _add_markdown_text(document, memo_to_markdown(memo, "en"), rtl=False)

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# ============================================================
# Managed-folder asset reads (logo + rendered page images).
# Identical to legal_ui._load_app_logo_data_uri / _load_page_image_bytes.
# ============================================================
def _load_app_logo_data_uri():
    try:
        import dataiku
        folder = dataiku.Folder(APP_ASSETS_FOLDER_ID)
        with folder.get_download_stream(APP_LOGO_PATH) as stream:
            raw = stream.read()
        return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    except Exception:
        return ""


def _load_page_image_bytes(page_image_path):
    if not page_image_path:
        return None
    try:
        import dataiku
        from legal_platform.config import CASE_DOCUMENT_FOLDER_ID
        folder = dataiku.Folder(CASE_DOCUMENT_FOLDER_ID)
        with folder.get_download_stream(page_image_path) as stream:
            return stream.read()
    except Exception:
        return None


def _page_row(data, page_id):
    pages = data.get("pages")
    if not isinstance(pages, pd.DataFrame) or pages.empty:
        return None
    id_column = "case_document_page_id" if "case_document_page_id" in pages.columns else "page_id"
    if id_column not in pages.columns:
        return None
    match = pages[pages[id_column].astype(str) == str(page_id)]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


# ============================================================
# Facts register serialisation (documents tab) — reproduces the
# page-id resolution logic from legal_ui.py so click-to-source works.
# ============================================================
def serialise_facts_register(data):
    facts_register = best(data, "facts", "fact_candidates")
    rows = []
    if not isinstance(facts_register, pd.DataFrame) or facts_register.empty:
        return rows
    pages_frame = data.get("pages")
    for _, row in facts_register.fillna("").iterrows():
        fact_text = str(row.get("fact_text", ""))
        if not fact_text:
            continue
        fact_id = str(row.get("fact_id", "") or row.get("fact_candidate_id", ""))
        page_ids = []
        if str(row.get("page_number", "")).strip() and isinstance(pages_frame, pd.DataFrame) and not pages_frame.empty:
            doc_col = "case_document_id" if "case_document_id" in pages_frame.columns else None
            if doc_col and str(row.get("case_document_id", "")).strip():
                match = pages_frame[
                    (pages_frame[doc_col].astype(str) == str(row.get("case_document_id", "")))
                    & (pages_frame["page_number"].astype(str) == str(row.get("page_number", "")))
                ]
                id_col = "case_document_page_id" if "case_document_page_id" in pages_frame.columns else "page_id"
                if not match.empty and id_col in match.columns:
                    page_ids = [str(match.iloc[0][id_col])]
        rows.append({
            "fact_id": fact_id,
            "fact_text": fact_text,
            "confidence": _clean_scalar(row.get("confidence")),
            "source_type": str(row.get("source_type", "") or ""),
            "verification_status": str(row.get("verification_status", "") or ""),
            "page_ids": page_ids,
            "page_labels": _page_labels(page_ids, data),
        })
    return rows


def serialise_chat_messages(data):
    messages = data.get("messages")
    out = []
    if isinstance(messages, pd.DataFrame) and not messages.empty:
        frame = messages.copy()
        if "created_at" in frame.columns:
            frame = frame.sort_values("created_at")
        for _, row in frame.iterrows():
            role = str(row.get("role", ""))
            if role in {"user", "assistant"}:
                out.append({"role": role, "content": str(row.get("message_text", ""))})
    return out


def serialise_summary_support(summary, data):
    """Attach page-label strings for chronology/evidence so the frontend
    can show the same 'Related evidence pages' text and click-to-source
    without re-implementing the page-reference map."""
    if not isinstance(summary, dict):
        return {}
    chronology = _ordered_chronology(summary.get("chronology", []))
    parties = _ordered_parties(summary.get("parties", []))
    evidence = []
    for item in summary.get("available_evidence", []) or []:
        if not isinstance(item, dict):
            continue
        page_ids = item.get("source_page_ids", []) or item.get("page_ids", []) or []
        evidence.append({"item": item, "page_ids": page_ids, "page_labels": _page_labels(page_ids, data)})
    return {
        "ordered_parties": parties,
        "ordered_chronology": [
            {"item": item, "page_labels": _page_labels(item.get("source_page_ids", []), data)}
            for item in chronology
        ],
        "ordered_evidence": evidence,
    }


# ============================================================
# Background job infrastructure (async + progress polling)
#
# Streamlit blocked the run and drew a live progress bar. Standard
# WebApps cannot block a request for a long LLM/extraction call without
# risking a proxy timeout, so we run the SAME call in a worker thread and
# expose its progress. The legal_platform calls, their arguments, and the
# order of persistence writes are identical to the Streamlit handlers.
# ============================================================
# ------------------------------------------------------------
# Execution mode
#
# RUN_JOBS_INLINE=True (the old default) runs every job synchronously
# inside the Flask request thread, exactly like the original Streamlit
# app. That defeats /job_status entirely: the HTTP request blocks for
# the full duration of the LLM/extraction pipeline (which can be
# minutes), so the webapp proxy can time it out mid-run.
#
# Backgrounding was previously abandoned because a raw daemon
# threading.Thread was suspected of losing Dataiku's auth/connection
# context. This codebase never impersonates the browsing user (no
# get_auth_info_from_browser_headers / internal_ticket calls anywhere),
# so per Dataiku's webapp-security model the backend authenticates as
# one fixed service identity for the whole process — not something tied
# to the Flask request/thread — and a background thread should carry it
# fine. Rather than trust that theory blind a second time,
# _dataiku_write_canary() below makes every backgrounded job prove it
# can write to and read back from this case's own dataset BEFORE it
# spends minutes on an LLM pipeline, so a real regression fails loudly
# up front instead of silently dropping the final write.
#
# True  -> runs inline (emergency fallback only — flip this back if the
#          canary starts failing in this Dataiku environment).
# False -> background execution (default).
# ------------------------------------------------------------
RUN_JOBS_INLINE = False

JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="legal-platform-job")


def _dataiku_write_canary(case_id):
    """Prove this worker thread can write to and read back from Dataiku
    before running an expensive pipeline on it. Raises loudly instead of
    letting a real result be computed and then silently lost."""
    marker_id = audit(case_id, "job_canary", case_id, "canary_write", actor="system")
    rows = case_rows("audit_events", case_id)
    ids = set(rows["audit_event_id"].astype(str)) if not rows.empty and "audit_event_id" in rows.columns else set()
    if marker_id not in ids:
        raise RuntimeError(
            "Background job could not read back its own write to Dataiku from this "
            "worker thread. Set RUN_JOBS_INLINE=True in main.py as an immediate fallback."
        )


def _new_job():
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "progress": {"stage": "", "current": 0, "total": 0, "detail": ""},
            "result": None,
            "error": None,
        }
    return job_id


def _set_progress(job_id, **kwargs):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job["progress"].update(kwargs)


def _finish_job(job_id, result=None, error=None):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        if error is not None:
            job["status"] = "error"
            job["error"] = error
        else:
            job["status"] = "done"
            job["result"] = result

### Youssif added session_id here ###
def _run_job(job_id, target, session_id=None):
    """Run a unit of work and record its outcome against job_id.

    The full traceback always goes to the webapp backend log (Logs tab)
    and the exception type + message go to the browser, so a failure is
    never silent at either end.
    """
    try:
        result = target()
        _finish_job(job_id, result=result)
    except Exception as error:  # noqa: BLE001 -> mirror Streamlit's broad except with a user-facing message
        traceback.print_exc()
        ### Added by Youssif for Monitoring Purposes ###
        if session_id:
            increment_usage(session_id, "error_count",1)
        ### END ###
        _finish_job(job_id, error=f"{type(error).__name__}: {error}")

### Youssif added session_id here ###
def _run_async(job_id, target, case_id=None, session_id=None):
    """Background by default; RUN_JOBS_INLINE=True is the emergency
    fallback to the old synchronous behaviour."""
    if RUN_JOBS_INLINE:
        ### Youssif added session_id here ###
        _run_job(job_id, target, session_id=session_id)
        return

    def guarded():
        if case_id:
            try:
                _dataiku_write_canary(case_id)
            except Exception as error:
                traceback.print_exc()
                _finish_job(job_id, error=f"{type(error).__name__}: {error}")
                return
        ### Youssif added session_id here ###
        _run_job(job_id, target, session_id=session_id)

    JOB_EXECUTOR.submit(guarded)


@app.route("/job_status")
def job_status():
    job_id = request.args.get("job_id", "")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"status": "unknown"}), 404
        payload = {
            "status": job["status"],
            "progress": dict(job["progress"]),
            "error": job["error"],
        }
        if job["status"] == "done":
            payload["result"] = job["result"]
        # Clear finished jobs after they are read once to avoid growth.
        if job["status"] in {"done", "error"}:
            JOBS.pop(job_id, None)
    return _ok(payload)


@app.route("/diagnostics")
def diagnostics():
    """Read-only: how many rows each case table actually holds.

    Answers "did the extraction read anything?" directly. Open it in a
    browser tab: <backend-url>/diagnostics?case_id=CASE_...
    Nothing is written and no business function is called.
    """
    case_id = request.args.get("case_id", "")
    if not case_id:
        return jsonify({"error": "case_id is required"}), 400
    data = load_case_data(case_id)
    counts = {}
    for key, frame in data.items():
        if isinstance(frame, pd.DataFrame):
            counts[key] = int(len(frame))
        elif isinstance(frame, dict):
            counts[key] = len(frame)
    state = restore_workflow_state(data)
    return _ok({
        "case_id": case_id,
        "row_counts": counts,
        "extraction_reached_facts": bool(counts.get("facts", 0) or counts.get("fact_candidates", 0)),
        "workflow": {
            "has_attorney_summary": bool(state.get("attorney_summary")),
            "summary_approved": bool(state.get("summary_approved")),
            "has_analysis": bool(state.get("analysis")),
            "has_memo": bool(state.get("memo")),
            "case_dirty": bool(state.get("case_dirty")),
            "dirty_reason": state.get("dirty_reason", ""),
        },
        "run_jobs_inline": RUN_JOBS_INLINE,
    })


# ============================================================
# Bootstrap + case library endpoints
# ============================================================
@app.route("/bootstrap")
def bootstrap():
    return jsonify({
        "logo": _load_app_logo_data_uri(),
        "agreement_types": list(AGREEMENT_TYPES),
        "relationship_types": list(RELATIONSHIP_TYPES),
    })


@app.route("/cases")
def cases_endpoint():
    workflow = request.args.get("workflow", "litigation")
    query = request.args.get("query", "")
    cases = list_cases()
    filtered = filter_cases_by_workflow(cases, workflow)
    filtered = case_search_frame(filtered, query)
    if "updated_at" in filtered.columns:
        filtered = filtered.sort_values("updated_at", ascending=False)
    cards = [serialise_case_card(row, workflow) for row in frame_to_records(filtered)]
    return _ok({"cases": cards})


@app.route("/create_case", methods=["POST"])
def create_case_endpoint():
    body = request.get_json(force=True)
    case_name = str(body.get("case_name", "")).strip()
    language = body.get("language", "en")
    workflow_type = body.get("workflow_type", "litigation")
    if not case_name:
        return jsonify({"error": "case_name is required"}), 400
    case_id = create_case(
        case_name=case_name,
        intake_mode="mixed",
        language=language,
        workflow_type=workflow_type,
    )
    return jsonify({"case_id": case_id})


# ============================================================
# Case snapshot — replaces the whole st.session_state read for a case.
# Returns the reconstructed workflow state plus everything the frontend
# needs to render Home / Documents / Attorney review.
# ============================================================
@app.route("/case")
def case_endpoint():
    case_id = request.args.get("case_id", "")
    if not case_id:
        return jsonify({"error": "case_id is required"}), 400
    data = load_case_data(case_id)
    case = data["case"]
    workflow_type = str(case.get("workflow_type", "") or "").strip() or "litigation"

    response = {
        "case_id": case_id,
        "reference": short_case_reference(case_id),
        "workflow_type": workflow_type,
        "case": {k: _clean_scalar(v) for k, v in case.items()},
        "display_name": first_case_value(case, "client_name", "customer_name", "case_name", default=""),
        "counts": {
            "documents": int(len(data["documents"])),
            "pages": int(len(data["pages"])),
            "facts": int(len(best(data, "facts", "fact_candidates"))),
            "parties": int(len(data["parties"])),
            "issues": int(len(best(data, "issues", "issue_candidates"))),
        },
    }

    if workflow_type == "agreement_review":
        stored = load_agreement_states(case_id)
        response["agreement_state"] = {
            "classification": stored.get("classification"),
            "profile": stored.get("confirmed_profile"),
            "clause_map": stored.get("clause_map"),
            "authorities": stored.get("authorities"),
            "review": stored.get("review"),
        }
        return _ok(response)

    state = restore_workflow_state(data)
    response["workflow_state"] = state
    response["steps"] = workflow_steps(state, data)
    response["next_action_key"] = next_action_key(state, data)
    response["chat_messages"] = serialise_chat_messages(data)
    response["facts_register"] = serialise_facts_register(data)
    if state.get("attorney_summary"):
        response["summary_support"] = serialise_summary_support(state["attorney_summary"], data)

    response["accounting"] = {
        "documents": [
            {
                "case_document_id": str(row.get("case_document_id", "")),
                "original_filename": row.get("original_filename") or row.get("case_document_id", ""),
            }
            for row in frame_to_records(data["documents"])
        ],
        "has_line_items": not data["financial_line_items"].empty,
        "pending_conflicts": _serialise_conflicts(case_id),
        "normalized_ledger": _build_normalized_ledger(case_id, data["financial_line_items"]),
        "cross_check_summary": data.get("cross_check_summary") or {},
        "discrepancies": data.get("discrepancies") or [],
        "findings": data.get("forensic_findings") or {},
    }
    return _ok(response)


def _load_state(case_id):
    data = load_case_data(case_id)
    return data, restore_workflow_state(data)


# ============================================================
# Litigation actions
# ============================================================
@app.route("/documents/process", methods=["POST"])
def documents_process():
    """Async: save_upload -> extract_pdf_page_by_page -> build_case_map ->
    persist_case_map for each PDF, then invalidate_for_new_material. Same
    calls/order as the Streamlit intake handler; progress reported live."""
    case_id = request.form.get("case_id", "")
    file_purpose = request.form.get("file_purpose", "full_case")
    ### Added by Youssif for Monitoring Purposes ###
    session_id = request.form.get("session_id")
    ### END ###
    
    render_zoom = 1.6
    uploaded = request.files.getlist("files")
    # Read the bytes and the MIME type now (the request context ends when
    # the worker thread runs).
    payloads = [(f.filename, f.read(), f.mimetype) for f in uploaded]
    job_id = _new_job()

    def task():
        total_pages = 0
        usable_pages = 0
        totals = {"facts": 0, "issues": 0, "parties": 0, "events": 0, "evidence_requests": 0}
        status_counts = {}
        failure_notes = []
        ### Added by Youssif for Monitoring Purposes ###
        llm_call_count = 0
        ### END ###
        for name, raw, content_type in payloads:
            document_row, pdf_bytes = save_upload(
                case_id, _MemoryUpload(name, raw, content_type), document_type=file_purpose
            )

            def update_progress(current, total, page_number, state, _name=name):
                _set_progress(job_id, stage="extract", current=current, total=total,
                              detail=f"{_name}: page {page_number}/{total} — {state}")

            pages = extract_pdf_page_by_page(
                case_id=case_id,
                case_document_id=document_row["case_document_id"],
                pdf_bytes=pdf_bytes,
                zoom=render_zoom,
                progress_callback=update_progress,
            )
            ### Added by Youssif for Monitoring Purposes ###
            llm_call_count += len(pages)
            ### END ###
            completed_pages = [p for p in pages if p.get("processing_status") in {"completed", "completed_review_required"}]
            for page in pages:
                page_status = str(page.get("processing_status", "") or "unknown")
                status_counts[page_status] = status_counts.get(page_status, 0) + 1
                if page_status in {"completed", "completed_review_required"}:
                    continue
                for note_key in ("processing_error", "error", "processing_note", "note", "failure_reason"):
                    if page.get(note_key):
                        note = str(page[note_key])[:300]
                        if note not in failure_notes:
                            failure_notes.append(note)
                        break
            total_pages += len(pages)
            usable_pages += len(completed_pages)
            if completed_pages:
                case_map = build_case_map(completed_pages)
                ### Added by Youssif for Monitoring Purposes ###
                llm_call_count += 1
                ### END ###
                counts = persist_case_map(case_id, case_map)
                for key in totals:
                    totals[key] += int(counts.get(key, 0) or 0)
        ### Added by Youssif for Monitoring Purposes ###
        if llm_call_count:
            increment_usage(session_id, "llm_request_count", llm_call_count)        
        ### END ###
        data = load_case_data(case_id)
        state = restore_workflow_state(data)
        invalidate_for_new_material(state, data, case_id, "New document or evidence was uploaded.")
        ### Added by Youssif for Monitoring Purposes ###
        if payloads:
            increment_usage(session_id, "attachment_count", len(payloads))
        ### END ###
        
        return {
            "files": len(payloads),
            "usable": usable_pages,
            "total": total_pages,
            # Diagnostics: what each page came back as, and the first few
            # reasons pages were rejected. Empty on a healthy run.
            "page_status": status_counts,
            "page_errors": failure_notes[:3],
            **totals,
        }
    ### Youssif added session_id here ##
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


class _MemoryUpload:
    """Adapter so save_upload receives the same interface Streamlit's
    UploadedFile provided.

    Streamlit's UploadedFile exposes .name, .type, .size and .id on top of
    the stream methods, and save_upload reads .type to record the MIME
    type. Only .name and the stream methods were reproduced originally,
    which is why processing raised AttributeError on the first upload.
    """

    def __init__(self, name, raw, content_type=""):
        self.name = name
        self.type = content_type or mimetypes.guess_type(name or "")[0] or "application/octet-stream"
        self.size = len(raw or b"")
        # A local stand-in for Streamlit's file id. It is never persisted,
        # so it deliberately does not use legal_platform's random_id().
        self.id = uuid.uuid4().hex
        self.file_id = self.id
        self._raw = raw
        self._buffer = io.BytesIO(raw)

    def read(self, *args, **kwargs):
        return self._buffer.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._buffer.seek(*args, **kwargs)

    def tell(self):
        return self._buffer.tell()

    def getvalue(self):
        return self._raw

    def getbuffer(self):
        return self._buffer.getbuffer()

    def close(self):
        self._buffer.close()


@app.route("/documents/review_completeness", methods=["POST"])
def documents_review_completeness():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    ### Added by Youssif for Monitoring Purposes ###
    session_id = body.get("session_id")
    ### END ###
    data, state = _load_state(case_id)
    interview_state = body.get("interview_state")
    result = assess_next_step(
        case_record=data["case"], messages=data["messages"], documents=data["documents"], pages=data["pages"],
        facts=data["facts"], fact_candidates=data["fact_candidates"], parties=data["parties"],
        issues=data["issues"], issue_candidates=data["issue_candidates"], evidence=data["evidence"],
        legal_research=((state.get("research") or {}).get("authority_nodes", [])),
        previous_state=interview_state,
    )
    ### Added by Youssif for Monitoring Purposes ###
    increment_usage(session_id, "llm_request_count",1)
    increment_usage(session_id, "message_count",1)
    ### END ###
    persist_interview_state(case_id, result["decision"].payload)
    return _ok({"reply": result["reply"], "interview_state": result["decision"].payload})


# ============================================================
# Accounting & forensic dispute analysis
#
# Same pipeline as the Streamlit "Accounting analysis" tab:
#   1. classify_case_documents        (Stage 2 — optional, manual trigger)
#   2. run_financial_extraction       (Stage 3 — dual-resolution OCR + reconcile)
#   3. submit_correction              (manual conflict resolution)
#   4. normalize_row_for_ledger       (per-row, done on every /case read)
#   5. build_and_save_financial_timeline + run_discrepancy_and_findings_analysis
# The normalized ledger, conflicts, timeline and findings are all
# returned as part of /case (see case_endpoint) so the tab renders from
# the same snapshot as everything else; these routes only perform writes.
# ============================================================
@app.route("/accounting/classify_documents", methods=["POST"])
def accounting_classify_documents():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    ### Added by Youssif for Monitoring Purposes ###
    session_id = body.get("session_id")
    ### END ###
    job_id = _new_job()

    def task():
        data = load_case_data(case_id)
        doc_ids = (
            data["documents"]["case_document_id"].dropna().astype(str).tolist()
            if not data["documents"].empty else []
        )
        _set_progress(job_id, stage="classify", detail="Classifying financial documents…")
        classify_case_documents(case_id, doc_ids)
        ### Added by Youssif for Monitoring Purposes ###
        increment_usage(session_id, "llm_request_count", 1)
        ### END ###
        return {"classified_documents": len(doc_ids)}
    ### Youssif added session_id here ##
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


@app.route("/accounting/clear", methods=["POST"])
def accounting_clear():
    """Fresh-start reset: same four datasets the Streamlit reset button
    wipes, filtered down to the current case only."""
    import dataiku

    body = request.get_json(force=True)
    case_id = str(body.get("case_id", ""))
    if not case_id:
        return jsonify({"error": "case_id is required"}), 400
    for dataset_name in [
        FINANCIAL_LINE_ITEMS_DATASET,
        FINANCIAL_LINE_ITEM_CORRECTIONS_DATASET,
        FINANCIAL_TIMELINE_DATASET,
        FINANCIAL_DISCREPANCIES_DATASET,
        FINANCIAL_FINDINGS_DATASET,
    ]:
        try:
            dataset = dataiku.Dataset(dataset_name)
            frame = dataset.get_dataframe()
            if not frame.empty and "case_id" in frame.columns:
                dataset.write_with_schema(frame[frame["case_id"].astype(str) != case_id])
        except Exception:
            traceback.print_exc()
    return _ok({"cleared": True})


@app.route("/accounting/extract", methods=["POST"])
def accounting_extract():
    """Async: dual-resolution (150 vs 250 DPI) visual extraction and
    cross-source reconciliation over the selected documents."""
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    document_ids = [str(d) for d in (body.get("document_ids") or [])]
    ### Added by Youssif for Monitoring Purposes ###
    session_id = body.get("session_id")
    ### END ###
    if not document_ids:
        return jsonify({"error": "Select at least one document."}), 400
    job_id = _new_job()

    def task():
        def _on_progress(current, total, page_id):
            _set_progress(
                job_id, stage="extraction", current=current, total=total,
                detail=f"Extracting page {current} of {total}…" if total else "Extracting…",
            )
        _set_progress(job_id, stage="extraction", detail="Starting dual-resolution extraction…")
        result = run_financial_extraction(case_id, document_ids, progress_callback=_on_progress)
        ### Added by Youssif for Monitoring Purposes ###
        increment_usage(session_id, "llm_request_count", 1)
        ### END ###
        rows_persisted = result.get("rows_persisted", 0) if isinstance(result, dict) else 0
        return {"rows_persisted": rows_persisted}
    ### Youssif added session_id here ##
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


@app.route("/accounting/correction", methods=["POST"])
def accounting_correction():
    """Attorney resolves one conflicted field on one extracted row."""
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    row_id = str(body.get("row_id", ""))
    field_name = str(body.get("field_name", ""))
    value = body.get("value", "")
    corrected_by = str(body.get("corrected_by", "") or "attorney")
    if not (case_id and row_id and field_name):
        return jsonify({"error": "case_id, row_id and field_name are required."}), 400
    submit_correction(case_id, row_id, field_name, value, corrected_by=corrected_by)
    return _ok({"saved": True})


@app.route("/accounting/synthesize", methods=["POST"])
def accounting_synthesize():
    """Async: build the forensic timeline, then run the cross-check /
    discrepancy / findings analysis over the normalized ledger."""
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    ### Added by Youssif for Monitoring Purposes ###
    session_id = body.get("session_id")
    ### END ###
    job_id = _new_job()

    def task():
        data = load_case_data(case_id)
        line_items_df = data["financial_line_items"]
        if line_items_df.empty:
            raise ValueError("No extracted line items available. Run extraction first.")
        normalized_ledger = _build_normalized_ledger(case_id, line_items_df)
        _set_progress(job_id, stage="timeline", detail="Building forensic financial timeline…")
        build_and_save_financial_timeline(case_id, normalized_ledger)
        _set_progress(job_id, stage="findings", detail="Checking numbers agreement and categorizing variances…")
        run_discrepancy_and_findings_analysis(case_id, normalized_ledger)
        ### Added by Youssif for Monitoring Purposes ###
        increment_usage(session_id, "llm_request_count", 1)
        increment_usage(session_id, "message_count", 1)
        ### END ###
        refreshed = load_case_data(case_id)
        return {
            "cross_check_summary": refreshed.get("cross_check_summary") or {},
            "discrepancies": refreshed.get("discrepancies") or [],
            "findings": refreshed.get("forensic_findings") or {},
        }
    ### Youssif added session_id here ##
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


@app.route("/summary/prepare", methods=["POST"])
def summary_prepare():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    ### Added by Youssif for Monitoring Purposes ###
    session_id = body.get("session_id")
    ### END ###
    job_id = _new_job()

    def task():
        data = load_case_data(case_id)
        state = restore_workflow_state(data)
        _set_progress(job_id, stage="summary", detail="Preparing consolidated attorney review…")
        summary = generate_case_summary(
            case_record=data["case"], documents=data["documents"], document_pages=data["pages"],
            facts=data["facts"], fact_candidates=data["fact_candidates"], parties=data["parties"],
            events=data["events"], issues=data["issues"], issue_candidates=data["issue_candidates"],
            evidence=data["evidence"], legal_research=((state.get("research") or {}).get("authority_nodes", [])),
            messages=data["messages"],
        )
        ### Added by Youssif for Monitoring Purposes ###
        increment_usage(session_id, "llm_request_count",1)
        increment_usage(session_id, "message_count",1)
        ### END ###
        state["attorney_summary"] = summary
        state["summary_approved"] = False
        state["memo"] = None
        state["case_dirty"] = False
        state["dirty_reason"] = ""
        persist_workflow_state(state, load_case_data(case_id), case_id, action="summary_prepared")
        return {"attorney_summary": summary, "summary_support": serialise_summary_support(summary, data)}
    ### Youssif added session_id here ##
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


@app.route("/summary/approve", methods=["POST"])
def summary_approve():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    reviewer = str(body.get("reviewer", "")).strip()
    comments = body.get("comments", "")
    data, state = _load_state(case_id)
    summary = state.get("attorney_summary")
    if not summary or not reviewer:
        return jsonify({"error": "A prepared summary and reviewer name are required."}), 400
    approval_id = approve_case_summary(case_id, summary, reviewer, comments)
    state["summary_approved"] = True
    state["summary_approved_by"] = reviewer
    state["memo"] = None
    state["case_dirty"] = False
    state["dirty_reason"] = ""
    persist_workflow_state(state, load_case_data(case_id), case_id, action="summary_approved", actor=reviewer)
    return _ok({"approval_id": approval_id, "workflow_state": state})


@app.route("/analysis/run", methods=["POST"])
def analysis_run():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    ### Added by Youssif for Monitoring Purposes ###
    session_id = body.get("session_id")
    ### END ###
    job_id = _new_job()

    def task():
        data = load_case_data(case_id)
        state = restore_workflow_state(data)
        facts = best(data, "facts", "fact_candidates")
        issues = best(data, "issues", "issue_candidates")
        if facts.empty or issues.empty:
            raise ValueError("Extracted facts and issues are required.")
        _set_progress(job_id, stage="analysis", detail="Retrieving applicable-law authorities…")
        result = run_legal_analysis(data["case"], facts, issues, data["evidence"])
        ### Added by Youssif for Monitoring Purposes ###
        increment_usage(session_id, "llm_request_count",1)
        increment_usage(session_id, "message_count",1)
        ### END ###
        state["research"] = result["research"]
        state["analysis"] = result["analysis"]
        persist_workflow_state(state, load_case_data(case_id), case_id, action="analysis_prepared")
        return {"research": state["research"], "analysis": state["analysis"]}
    ### Youssif added session_id here ##
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


@app.route("/analysis/defence_plan", methods=["POST"])
def analysis_defence_plan():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    ### Added by Youssif for Monitoring Purposes ###
    session_id = body.get("session_id")
    ### END ###
    job_id = _new_job()

    def task():
        data = load_case_data(case_id)
        state = restore_workflow_state(data)
        if not state.get("summary_approved"):
            raise ValueError("Approve the consolidated attorney review first.")
        _set_progress(job_id, stage="defence", detail="Developing defence plan…")
        strategy = run_defence_plan(
            data["case"], state["analysis"], best(data, "facts", "fact_candidates"),
            data["evidence"], state["research"]["authority_nodes"],
        )
        ### Added by Youssif for Monitoring Purposes ###
        increment_usage(session_id, "llm_request_count",1)
        increment_usage(session_id, "message_count",1)
        ### END ###
        state["strategy"] = strategy
        persist_workflow_state(state, load_case_data(case_id), case_id, action="defence_plan_prepared")
        return {"strategy": strategy}
    ### Youssif added session_id here ##
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


@app.route("/pleading/generate", methods=["POST"])
def pleading_generate():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    memo_instructions = body.get("instructions", "")
    direct = bool(body.get("direct", False))
    ### Added by Youssif for Monitoring Purposes ###
    session_id = body.get("session_id")
    ### END ###
    job_id = _new_job()

    def task():
        data = load_case_data(case_id)
        state = restore_workflow_state(data)
        case = data["case"]
        _set_progress(job_id, stage="pleading", detail="Drafting written pleading…")
        instructions = (
            "Prepare a complete formal defence pleading exclusively on behalf of Banque Saudi Fransi (BSF)."
            if direct else memo_instructions
        )
        memo = generate_bilingual_memo(
            case, state["attorney_summary"], state["analysis"], state["strategy"],
            state["research"], instructions,
            case_data=data,
        )
        ### Added by Youssif for Monitoring Purposes ###
        increment_usage(session_id, "llm_request_count",1)
        increment_usage(session_id, "message_count",1)
        ### END ###
        state["memo"] = memo
        note = "Initial approved-summary draft" if direct else (memo_instructions or "Initial pleading draft")
        state["pleading_versions"] = [{"version": 1, "draft": memo, "note": note}]
        state["pleading_status"] = "draft"
        state["pleading_finalised_by"] = ""
        persist_workflow_state(state, load_case_data(case_id), case_id, action="pleading_prepared")
        return {"memo": memo, "pleading_versions": state["pleading_versions"], "pleading_status": "draft"}
    ### Youssif added session_id here ##
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


@app.route("/pleading/revise", methods=["POST"])
def pleading_revise():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    revision_request = body.get("revision_request", "")
    ### Added by Youssif for Monitoring Purposes ###
    session_id = body.get("session_id")
    ### END ###
    job_id = _new_job()

    def task():
        data = load_case_data(case_id)
        state = restore_workflow_state(data)
        revised = revise_bilingual_pleading(
            revision_request,
            state["memo"],
            data["case"],
            state["attorney_summary"],
            state["analysis"],
            state["strategy"],
            state["research"],
            case_data=data,
        )
        ### Added by Youssif for Monitoring Purposes ###
        increment_usage(session_id, "llm_request_count",1)
        increment_usage(session_id, "message_count",1)
        ### END ###
        version_no = len(state["pleading_versions"]) + 1
        state["memo"] = revised
        state["pleading_versions"].append({"version": version_no, "draft": revised, "note": revision_request})
        state["pleading_status"] = "draft"
        persist_workflow_state(state, load_case_data(case_id), case_id, action="pleading_revised", reason=revision_request)
        return {
            "memo": revised,
            "pleading_versions": state["pleading_versions"],
            "pleading_status": "draft",
            "version": version_no,
        }
    ### Youssif added session_id here ##
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


@app.route("/pleading/restore_version", methods=["POST"])
def pleading_restore_version():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    selected_version = body.get("version")
    data, state = _load_state(case_id)
    selected = next((v for v in state["pleading_versions"] if v["version"] == selected_version), None)
    if not selected:
        return jsonify({"error": "version not found"}), 404
    version_no = len(state["pleading_versions"]) + 1
    state["memo"] = selected["draft"]
    state["pleading_versions"].append({
        "version": version_no, "draft": selected["draft"],
        "note": f"Restored from version {selected_version}",
    })
    persist_workflow_state(state, load_case_data(case_id), case_id, action="pleading_version_restored")
    return _ok({"memo": state["memo"], "pleading_versions": state["pleading_versions"]})


@app.route("/pleading/mark_final", methods=["POST"])
def pleading_mark_final():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    final_reviewer = str(body.get("final_reviewer", "")).strip()
    if not final_reviewer:
        return jsonify({"error": "final_reviewer is required"}), 400
    data, state = _load_state(case_id)
    state["pleading_status"] = "final"
    state["pleading_finalised_by"] = final_reviewer
    persist_workflow_state(state, load_case_data(case_id), case_id, action="pleading_finalised", actor=final_reviewer)
    return jsonify({"pleading_status": "final", "pleading_finalised_by": final_reviewer})


@app.route("/pleading/reopen", methods=["POST"])
def pleading_reopen():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    data, state = _load_state(case_id)
    state["pleading_status"] = "draft"
    state["pleading_finalised_by"] = ""
    persist_workflow_state(state, load_case_data(case_id), case_id, action="pleading_reopened")
    return jsonify({"pleading_status": "draft"})


@app.route("/pleading/export")
def pleading_export():
    case_id = request.args.get("case_id", "")
    fmt = request.args.get("fmt", "md")
    language = request.args.get("lang", "en")
    data, state = _load_state(case_id)
    memo = state.get("memo")
    if not memo:
        return jsonify({"error": "no pleading available"}), 404
    reference = short_case_reference(case_id)
    if fmt == "docx":
        try:
            docx_bytes = build_pleading_docx_bytes(memo, reference)
        except ImportError:
            return jsonify({"error": "python-docx is not installed in this code environment."}), 501
        return send_file(
            io.BytesIO(docx_bytes),
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            as_attachment=True,
            download_name=f"pleading_{reference}.docx",
        )
    text = memo_to_markdown(memo, language)
    return Response(
        text,
        mimetype="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="pleading_{language}.md"'},
    )


@app.route("/discussion/ask", methods=["POST"])
def discussion_ask():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    conversation_id = body.get("conversation_id", "")
    question = body.get("question", "")
    add_to_record = bool(body.get("add_to_record", False))
    job_id = _new_job()

    def task():
        data = load_case_data(case_id)
        state = restore_workflow_state(data)
        message_id = add_message(case_id, conversation_id, "user", question)
        answer = answer_case_question(
            question, data["case"], state["attorney_summary"], state["analysis"],
            state["strategy"], state["research"], case_data=data,
        )
        # Persist the assistant reply text (bilingual payload returned to client).
        assistant_text = json.dumps({
            "answer_ar": answer.get("answer_ar", ""),
            "answer_en": answer.get("answer_en", ""),
            "uncertainties": answer.get("uncertainties") or [],
            "source_ids": answer.get("source_ids") or [],
        }, ensure_ascii=False)
        add_message(case_id, conversation_id, "assistant", assistant_text)
        if add_to_record:
            add_fact_candidate(
                case_id=case_id, fact_text=question, source_type="chat",
                source_id=message_id, fact_type="user_statement", quote=question, confidence=0.65,
            )
            invalidate_for_new_material(
                state, load_case_data(case_id), case_id,
                "A discussion message was explicitly added to the formal case record.",
            )
        return {"answer": answer, "invalidated": add_to_record}

    _run_async(job_id, task, case_id=case_id)
    return jsonify({"job_id": job_id})


# ============================================================
# Page image + extracted text (click-to-source)
# ============================================================
@app.route("/page_image")
def page_image():
    case_id = request.args.get("case_id", "")
    page_id = request.args.get("page_id", "")
    data = load_case_data(case_id)
    row = _page_row(data, page_id)
    if not row:
        return Response(status=404)
    image_bytes = _load_page_image_bytes(row.get("page_image_path", ""))
    if not image_bytes:
        return Response(status=404)
    return send_file(io.BytesIO(image_bytes), mimetype="image/png")


@app.route("/page_text")
def page_text():
    case_id = request.args.get("case_id", "")
    page_id = request.args.get("page_id", "")
    data = load_case_data(case_id)
    row = _page_row(data, page_id)
    if not row:
        return jsonify({"label": "", "text": ""})
    label = _page_reference_map(data).get(str(page_id), "Original page")
    return _ok({"label": label, "text": str(row.get("page_text", "") or "")[:4000]})


# ============================================================
# Fact / clause verify-flag-note controls (append-only audit)
# ============================================================
@app.route("/flag", methods=["POST"])
def flag_control():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    entity_type = body.get("entity_type", "")
    entity_id = body.get("entity_id", "")
    action = body.get("action", "")  # verified | flagged | corrected
    actor = body.get("actor", "") or "attorney"
    reason = body.get("reason", "")
    if action not in {"verified", "flagged", "corrected"}:
        return jsonify({"error": "invalid action"}), 400
    audit(case_id, entity_type, entity_id, action, actor=actor, reason=reason)
    return jsonify({"ok": True})


# ============================================================
# Agreement workflow endpoints
# ============================================================
@app.route("/agreement/process", methods=["POST"])
def agreement_process():
    case_id = request.form.get("case_id", "")
    ### Added by Youssif for Monitoring Purposes ###
    session_id = request.form.get("session_id")
    ### END ###
    try:
        zoom = float(request.form.get("zoom", "1.6"))
    except ValueError:
        zoom = 1.6
    uploaded = request.files.getlist("files")
    payloads = [(f.filename, f.read(), f.mimetype) for f in uploaded]
    job_id = _new_job()

    def task():
        processed = 0
        ### Added by Youssif for Monitoring Purposes ###
        llm_call_count = 0
        ### END ###
        for name, raw, content_type in payloads:
            document_row, pdf_bytes = save_upload(case_id, _MemoryUpload(name, raw, content_type), document_type="agreement")

            def on_page(current, total, page_number, state, _name=name):
                _set_progress(job_id, stage="extract", current=current, total=total,
                              detail=f"{_name}: page {page_number}/{total} — {state}")

            pages = extract_pdf_page_by_page(
                case_id=case_id,
                case_document_id=document_row["case_document_id"],
                pdf_bytes=pdf_bytes, zoom=zoom,
                progress_callback=on_page,
            )
             ### Added by Youssif for Monitoring Purposes ###
            llm_call_count += len(pages)
            ### END ###
            if pages:
                processed += 1
        ### Added by Youssif for Monitoring Purposes ###
        if llm_call_count:
            increment_usage(session_id, "llm_request_count", llm_call_count)
        if processed:
            increment_usage(session_id, "attachment_count", processed)
        ### END ###
        return {"processed": processed}
    ### Youssif added session_id here ##
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


@app.route("/agreement/classify", methods=["POST"])
def agreement_classify():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    ### Added by Youssif for Monitoring Purposes ###
    session_id = body.get("session_id")
    ### END ###
    job_id = _new_job()

    def task():
        data = load_case_data(case_id)
        _set_progress(job_id, stage="classify", detail="Detecting agreement type and relationship…")
        result = classify_agreement(data["pages"])
        ### Added by Youssif for Monitoring Purposes ###
        increment_usage(session_id, "llm_request_count",1)
        increment_usage(session_id, "message_count",1)
        ### END ###
        save_agreement_state(case_id, "classification", result)
        return {"classification": result}

    ### Youssif added session_id here ###
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


@app.route("/agreement/confirm_classification", methods=["POST"])
def agreement_confirm_classification():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    profile = {
        "agreement_type": body.get("agreement_type"),
        "relationship_type": body.get("relationship_type"),
        "represented_party": str(body.get("represented_party", "")).strip(),
        "counterparty": str(body.get("counterparty", "")).strip(),
        "review_objective": str(body.get("review_objective", "")).strip(),
        "confirmed_by_attorney": True,
    }
    if not profile["represented_party"]:
        return jsonify({"error": "represented_party is required"}), 400
    save_agreement_state(case_id, "confirmed_profile", profile)
    return _ok({"profile": profile})


@app.route("/agreement/extract_clauses", methods=["POST"])
def agreement_extract_clauses():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    ### Added by Youssif for Monitoring Purposes ###
    session_id = body.get("session_id")
    ### END ###
    job_id = _new_job()

    def task():
        data = load_case_data(case_id)
        stored = load_agreement_states(case_id)
        profile = stored.get("confirmed_profile")

        def on_page(current, total, page_id):
            _set_progress(job_id, stage="pages", current=current, total=total,
                          detail=f"Page structure {current}/{total} · {page_id}")

        def on_chunk(current, total, chunk_id):
            _set_progress(job_id, stage="chunks", current=current, total=total,
                          detail=f"Consolidation chunk {current}/{total} · {chunk_id}")

        clause_map = extract_clause_map(
            data["pages"], profile,
            page_progress_callback=on_page,
            chunk_progress_callback=on_chunk,
        )
        ### Added by Youssif for Monitoring Purposes ###
        increment_usage(session_id, "llm_request_count",1)
        increment_usage(session_id, "message_count",1)
        ### END ###
        save_agreement_state(case_id, "clause_map", clause_map)
        return {"clause_map": clause_map}
    ### Youssif added session_id here ###
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


@app.route("/agreement/run_review", methods=["POST"])
def agreement_run_review():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    instructions = body.get("instructions", "")
    ### Added by Youssif for Monitoring Purposes ###
    session_id = body.get("session_id")
    ### END ###
    job_id = _new_job()

    def task():
        stored = load_agreement_states(case_id)
        clause_map = stored.get("clause_map")
        profile = stored.get("confirmed_profile")

        def on_review(current, total, clause_id):
            _set_progress(job_id, stage="review", current=current, total=total,
                          detail=f"Clause review {current}/{total} · {clause_id}")

        review, authorities = run_agreement_review(
            clause_map, profile, instructions=instructions, progress_callback=on_review,
        )
        ### Added by Youssif for Monitoring Purposes ###
        increment_usage(session_id, "llm_request_count",1)
        increment_usage(session_id, "message_count",1)
        ### END ###
        authorities_payload = {"authority_nodes": authorities}
        save_agreement_state(case_id, "authorities", authorities_payload)
        save_agreement_state(case_id, "review", review)
        return {"review": review, "authorities": authorities_payload}
    ### Youssif added session_id here ###
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


@app.route("/agreement/discuss", methods=["POST"])
def agreement_discuss():
    body = request.get_json(force=True)
    case_id = body.get("case_id", "")
    question = body.get("question", "")
    ### Added by Youssif for Monitoring Purposes ###
    session_id = body.get("session_id")
    ### END ###
    job_id = _new_job()

    def task():
        stored = load_agreement_states(case_id)
        authority_nodes = (stored.get("authorities") or {}).get("authority_nodes", [])
        answer = discuss_agreement(
            question,
            stored.get("confirmed_profile") or {},
            stored.get("clause_map") or {},
            stored.get("review") or {},
            authority_nodes,
        )
        ### Added by Youssif for Monitoring Purposes ###
        increment_usage(session_id, "llm_request_count",1)
        increment_usage(session_id, "message_count",1)
        ### END ###
        return {"answer": answer}
    ### Youssif added session_id here ##
    _run_async(job_id, task, case_id=case_id, session_id=session_id)
    return jsonify({"job_id": job_id})


