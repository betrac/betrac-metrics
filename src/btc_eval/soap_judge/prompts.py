"""Prompt templates for the SOAP-note LLM judge.

Two ``str.format`` templates drive the pipeline:

* ``EXTRACT_CLAIMS_PROMPT`` -- stage 1, placeholder ``{soap_note}``.
* ``JUDGE_SOAP_NOTE_PROMPT`` -- stage 2, placeholders ``{transcript}``,
  ``{soap_note}``, ``{claims_json}``.

Both ask the model to return a single JSON object. Literal JSON braces inside
the templates are doubled (``{{`` / ``}}``) so ``str.format`` leaves them
intact -- preserve that if you edit a template.

The rubric *is* the metric: editing these templates changes what the score
means. To experiment without forking the package, drop overrides in a directory
and point ``--prompt-dir`` (or ``$BTC_SOAP_PROMPT_DIR``) at it -- see
:func:`load_prompt`.
"""

from __future__ import annotations

import os
from pathlib import Path

EXTRACT_CLAIMS_PROMPT = """SYSTEM:
You are an information extraction engine. Your job is to extract atomic clinical documentation claims from a SOAP note.
You must NOT add medical knowledge, interpretations, or assumptions. Extract ONLY what is explicitly written.
If something is vague, keep it vague. If something is uncertain, mark it uncertain. If something is denied, mark it denied.
Return valid JSON only that matches the schema.

DEFINITION (Atomic claim):
A single, minimal statement that can be verified against a transcript using evidence.
Each claim should contain exactly one main assertion (or one denial) about one fact/event.

USER:
Extract atomic claims from the SOAP note below.

OUTPUT REQUIREMENTS:
- Return ONLY JSON.
- JSON must follow this schema:

{{
  "doc_type": "soap_claims",
  "claims": [
    {{
      "claim_id": "C001",
      "soap_section": "S|O|A|P|AP|Unknown",
      "claim_type": "chief_complaint|symptom|negated_symptom|history|medication_current|medication_started|medication_stopped|allergy|diagnosis|differential|exam_finding|vital|lab_result|imaging_result|procedure|order|referral|counseling|followup|patient_goal|other",
      "subject": "patient|clinician|caregiver|unknown",
      "speaker_attribution": "patient_reported|clinician_observed|clinician_assessed|clinician_planned|unknown",
      "polarity": "asserted|denied|uncertain",
      "temporality": "current|past|ongoing|planned|unknown",
      "certainty": "high|medium|low|unknown",
      "text": "verbatim-ish short claim text",
      "normalized": {{
        "entities": [
          {{
            "kind": "symptom|condition|medication|dose|route|frequency|duration|test|imaging|body_part|laterality|time|value|unit|other",
            "value": "string"
          }}
        ],
        "numeric": [
          {{
            "name": "string",
            "value": "number or string",
            "unit": "string or null"
          }}
        ],
        "time_expressions": ["string", "..."]
      }},
      "source_spans": [
        {{
          "source": "soap_note",
          "snippet": "exact snippet from the SOAP note that supports this claim"
        }}
      ],
      "notes": "optional; use only to clarify ambiguity, not to add new facts"
    }}
  ]
}}

RULES:
1) Do NOT invent details (no implied negatives like “denies fever” unless explicitly written).
2) Split combined sentences into multiple claims when they contain multiple checkable facts.
   - Example: “Fever and cough x3 days” => two claims (fever present, cough present) plus duration if clearly attached.
3) Keep negations as first-class claims (polarity=denied, claim_type=negated_symptom or appropriate).
4) Keep uncertainty as uncertainty (e.g., “likely”, “possible”, “consider”) => polarity=uncertain and/or certainty low/medium.
5) For medication orders, extract dose/route/frequency/duration as separate entities where present.
6) Do not rewrite into guidelines. Just extract what the note says.
7) If the SOAP section isn’t clear, use "Unknown".
8) Use stable IDs C001, C002, ... in the order the claims appear in the note.
9) Keep "text" short (ideally <= 20 words) but specific.
10) If a claim comes from a combined "Assessment and Plan" area and cannot be cleanly separated, use soap_section="AP" (otherwise use A for assessment statements and P for plan/orders).

SOAP NOTE:
<<<
{soap_note}
>>>
"""


JUDGE_SOAP_NOTE_PROMPT = """SYSTEM:
You are a strict clinical documentation evaluator (LLM judge) for SOAP notes created from doctor–patient transcripts.
Your job is to score the NOTE against the TRANSCRIPT with NO external medical knowledge.
Be conservative: if you cannot find explicit supporting evidence in the transcript, mark it Not-in-transcript.
Do NOT reward plausible but unstated details.
Do NOT penalize the note for faithfully documenting a poor clinician; focus on fidelity and documentation quality.

You will output ONLY valid JSON following the provided schema.

EVIDENCE RULES:
- When labeling a claim as Supported or Contradicted, you MUST provide at least one direct quote span from the transcript with speaker and turn_id.
- Quote spans must be exact substrings from the transcript and each quote must be <= 25 words.
- If you cannot locate explicit evidence, label the claim Not-in-transcript and leave evidence empty.
- If retrieval candidates are provided, ONLY use those; if none contains evidence, label Not-in-transcript.

SCORING (1–5 each subscore; 5 is best):
1) Faithfulness / Grounding: based on Supported vs Contradicted vs Not-in-transcript rates.
2) Structure / Formatting: correct SOAP headers, correct section placement, forbidden content checks.
3) Coverage / Completeness: whether key transcript facts are captured (checklist-based).
4) Conciseness: redundancy and supported-but-low-value or irrelevant content rate.

IMPORTANT DISTINCTIONS:
- Faithfulness: Is what is written explicitly supported by the transcript?
- Coverage: Did the note capture the important things that WERE said?
- Conciseness: Is the note compact and non-redundant while still capturing key content?

WHO-SAID MISMATCH:
- If a claim is marked as patient_reported but the supporting evidence is clinician-only (or vice versa),
  set who_said_mismatch = true for that claim, even if it is Supported.

IMPORTANT MEDICALIZATION & SPECIFICITY GUIDANCE (CRITICAL):
- Translating lay terms into standard medical terminology is ALLOWED and GOOD.
- Medicalization itself is NOT an error.
- ONLY flag errors when the note introduces unjustified medical interpretation or specificity
  that is not explicitly supported by the transcript.
- Do NOT invent implicit negatives, default clinical assumptions, or textbook-style inferences.

IMPORTANT UNDER-MEDICALIZATION GUIDANCE:
- Use under_medicalization ONLY when:
  - The transcript explicitly uses a standard medical term or diagnosis, AND
  - The SOAP note replaces it with a more vague or lay expression
    that loses clinically relevant meaning.
- Do NOT use under_medicalization when:
  - The transcript itself uses lay language only.
  - The SOAP note reasonably paraphrases without loss of meaning.
  - The choice is purely stylistic or about wording preference.
- Under-medicalization is about loss of medically explicit content,
  NOT about whether the note "could have been more medical".
  
USER:
Evaluate the SOAP note using the transcript and extracted claims.

INPUTS:
A) TRANSCRIPT
<<<
{transcript}
>>>

B) SOAP NOTE (raw)
<<<
{soap_note}
>>>

C) EXTRACTED CLAIMS (from the SOAP note)
<<<
{claims_json}
>>>

TASKS:

TASK 1 — Claim-level support classification
For EACH claim in C:
- Determine label: Supported | Contradicted | Not-in-transcript
- Provide evidence quotes (speaker + turn_id + quote) for Supported or Contradicted claims.
- Flag who_said_mismatch if claim.speaker_attribution implies patient_reported but evidence is clinician-only (or the reverse).
- If the claim contains multiple independent facts and only some are supported,
  label it Partial and specify supported_parts and unsupported_parts.

ERROR TYPE DEFINITIONS (STRICT):
If a claim has ANY issue — regardless of whether it is Supported, Partial, Contradicted, or Not-in-transcript —
classify the error using one or more of the following types.
NOTE: over_medicalization, under_medicalization, over_specific, and wrong_section
can occur even on Supported claims. Always check for these independently of the label.

1) missing_support:
   - The claim is plausible but not mentioned in the transcript.
   - Includes invented vitals, labs, imaging results, diagnoses, and ROS negatives not present in the transcript.
2) contradiction:
   - The claim directly conflicts with what is stated in the transcript.
3) over_specific:
   - The claim adds unjustified specificity beyond the transcript
     (e.g., numbers, exact duration, frequency, laterality, severity, explicit negatives).
4) over_medicalization:
   - The claim introduces a specific diagnosis, pathology, etiology, or medical interpretation
     not explicitly stated or confirmed in the transcript.
5) wrong_section:
   - The claim content belongs in a different SOAP section than indicated.
6) other:
   - Any other notable issue not covered above.
7) under_medicalization:
   - The claim weakens or omits a standard medical concept
     that was explicitly stated in the transcript,
     by replacing it with overly vague or lay wording,
     resulting in loss of clinical specificity.
   - Use this ONLY when the transcript itself contains an explicit medical term or diagnosis
     that the note fails to preserve.
   - This error can occur even when label = Supported.

TASK 2 — Structure / formatting check
Check for:
- Presence of all required top-level SOAP sections: Subjective, Objective, Assessment, Plan.
  NOTE: If a combined "Assessment & Plan" or "A&P" section exists, diagnostic
    reasoning AND orders/instructions may both appear there without constituting
    a placement error.
- Presence of common subheaders (e.g., CC, HPI, ROS) in Subjective when relevant.
- Correct section placement:
  - Subjective content in S,
  - Exam/labs in O,
  - Assessment reasoning or diagnoses in A,
  - Orders, referrals, instructions, or follow-up in P.

Return pass/fail flags and a list of errors or warnings.

TASK 3 — Coverage / completeness checklist (transcript-grounded)
Create a checklist of key items explicitly stated in the transcript that a good SOAP note SHOULD capture.
Each checklist item must be grounded in transcript evidence.

Minimum checklist categories (if present in transcript):
- Chief complaint / primary concern
- Key HPI elements (onset, duration, modifiers, severity, functional impact)
- Explicit negatives (only those explicitly asked and answered)
- Exam findings explicitly stated
- Clinician-stated concerns, uncertainty, or differential (only if stated)
- Plan items: orders, referrals, instructions, follow-up
- Patient adherence, refusal, or preferences

CROSS-CHECK (apply after checklist):
- HPI and ROS overlap consistency: flag as a warning if ROS includes denials or items semantically overlapping with any positive HPI/CC symptom 
(e.g., pain/ache/discomfort; SOB/dyspnea; dizziness/lightheadedness), unless the transcript explicitly disambiguates context/time/location.
- Report any overlap warnings in structure_findings.warnings.

For each checklist item:
- Indicate whether it is documented in the SOAP note (yes/no).
- If yes, list the claim_id(s) that cover it.
- If no, assign omission_severity: critical | moderate | minor.

Compute coverage_rate and critical_omissions_count.

TASK 4 — Conciseness
Assess:
- Redundancy: repeated or duplicated claims across sections.
- Supported-but-low-value content that adds little clinical value.
- Treat purely administrative/charting/meta content (forms, billing, documentation requirements) as low-value even if supported, unless it is explicitly tied to a clinical action or instruction.

Return:
- redundancy_count
- low_value_supported_count
- conciseness_score (1–5) with a brief rationale.

OUTPUT JSON SCHEMA (return EXACTLY this structure):
{{
  "doc_type": "soap_judgment",
  "subscores_1_to_5": {{
    "faithfulness_grounding": 1,
    "structure_formatting": 1,
    "coverage_completeness": 1,
    "conciseness": 1
  }},
  "metrics": {{
    "claim_counts": {{
      "total": 0,
      "supported": 0,
      "contradicted": 0,
      "not_in_transcript": 0,
      "partial": 0,
      "who_said_mismatch": 0
    }},
    "rates": {{
      "unsupported_rate": 0.0,
      "contradiction_rate": 0.0,
      "evidence_coverage_rate": 0.0,
      "who_said_mismatch_rate": 0.0
    }},
    "structure": {{
      "has_all_soap_sections": true,
      "section_order_ok": true,
      "placement_errors_count": 0,
      "forbidden_content_flags": []
    }},
    "coverage": {{
      "checklist_total": 0,
      "checklist_yes": 0,
      "coverage_rate": 0.0,
      "critical_omissions_count": 0
    }},
    "conciseness": {{
      "redundancy_count": 0,
      "low_value_supported_count": 0
    }}
  }},
  "claim_judgments": [
    {{
      "claim_id": "C001",
      "label": "Supported|Contradicted|Not-in-transcript|Partial",
      "who_said_mismatch": false,
      "evidence": [
        {{ "speaker": "DOCTOR|PATIENT", "turn_id": 0, "quote": "..." }}
      ],
      "supported_parts": ["..."],
      "unsupported_parts": ["..."],
      "error_types": [
        "missing_support|contradiction|over_specific|over_medicalization|under_medicalization|wrong_section|other"
      ]
    }}
  ],
  "structure_findings": {{
    "errors": ["..."],
    "warnings": ["..."]
  }},
  "coverage_checklist": [
    {{
      "item_id": "K01",
      "category": "CC|HPI|ROS|Exam|Assessment|Plan|Adherence",
      "item_text": "...",
      "evidence": [
        {{ "speaker": "DOCTOR|PATIENT", "turn_id": 0, "quote": "..." }}
      ],
      "documented_in_note": true,
      "covered_by_claim_ids": ["C001", "C005"],
      "omission_severity": "critical|moderate|minor|na"
    }}
  ],
  "conciseness_findings": {{
    "redundant_examples": ["..."],
    "low_value_supported_examples": ["..."],
    "notes": "brief rationale"
  }},
  "top_issues": [
    {{ "type": "contradiction|hallucination|missing_critical|structure", "detail": "..." }}
  ]
}}

SUBSCORE GUIDANCE:
- faithfulness_grounding:
  5 = no contradictions and very low unsupported rate
  3 = some unsupported claims, few contradictions
  1 = many unsupported claims or contradictions
- structure_formatting:
  5 = correct SOAP structure and placement, no forbidden content
  3 = minor placement or formatting issues
  1 = missing sections or major errors
- coverage_completeness:
  5 = captures nearly all key items, no critical omissions
  3 = some omissions, few critical
  1 = many omissions, multiple critical
- conciseness:
  5 = minimal redundancy and no fluff
  3 = some redundancy or low-value content
  1 = bloated or repetitive

Return JSON only.
"""


# Logical prompt name -> packaged default. Override files use these names
# (e.g. ``extract_claims.txt``, ``judge_soap_note.txt``).
_DEFAULTS: dict[str, str] = {
    "extract_claims": EXTRACT_CLAIMS_PROMPT,
    "judge_soap_note": JUDGE_SOAP_NOTE_PROMPT,
}

PROMPT_ENV_VAR = "BTC_SOAP_PROMPT_DIR"


def load_prompt(name: str, prompt_dir: str | os.PathLike[str] | None = None) -> str:
    """Return a prompt template by logical name, honoring overrides.

    Lookup precedence (first hit wins):
        1. ``<prompt_dir>/<name>.txt``        (the ``--prompt-dir`` flag)
        2. ``$BTC_SOAP_PROMPT_DIR/<name>.txt`` (environment override)
        3. the packaged default template

    Overrides are ``str.format`` templates and MUST preserve the placeholders
    (``{soap_note}`` etc.) and double every literal JSON brace (``{{`` / ``}}``).

    Raises:
        KeyError: if ``name`` is not a known prompt.
    """
    if name not in _DEFAULTS:
        raise KeyError(
            f"Unknown prompt {name!r}. Known prompts: {sorted(_DEFAULTS)}"
        )
    for base in (prompt_dir, os.environ.get(PROMPT_ENV_VAR)):
        if not base:
            continue
        candidate = Path(base) / f"{name}.txt"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return _DEFAULTS[name]
