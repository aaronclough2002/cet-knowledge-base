import re
from typing import Dict, List


FUTURE_TERMS = [
    r"\bwill\b",
    r"\bplans?\s+to\b",
    r"\bplanning\s+to\b",
    r"\bintends?\s+to\b",
    r"\bexpected\s+to\b",
    r"\banticipates?\b",
    r"\bforecast(?:ed|s|ing)?\b",
    r"\bproject(?:ed|ion|ions)?\b",
    r"\broadmap\b",
    r"\bstrategy\b",
    r"\bstrategic\s+initiative(s)?\b",
    r"\bnext\s+year\b",
    r"\bupcoming\b",
    r"\bfuture\b",
    r"\bover\s+the\s+next\s+\d+\s+(month|months|year|years)\b",
    r"\bby\s+20\d{2}\b",
    r"\bq[1-4]\s+20\d{2}\b",
    r"\bfy\s*20\d{2}\b",
    r"\b20(2[6-9]|3\d)\b",  # catches 2026+ as likely future planning language
]

SENSITIVE_PLAN_TERMS = [
    r"\bhiring\b",
    r"\bheadcount\b",
    r"\bworkforce\b",
    r"\blayoffs?\b",
    r"\breduction\s+in\s+force\b",
    r"\brestructuring\b",
    r"\breorganization\b",
    r"\bexpansion\b",
    r"\bexpand\b",
    r"\bmarket\s+entry\b",
    r"\bnew\s+office\b",
    r"\bnew\s+location\b",
    r"\bsite\s+opening\b",
    r"\bsite\s+closure\b",
    r"\bfacility\s+opening\b",
    r"\bfacility\s+closure\b",
    r"\bacquisition(s)?\b",
    r"\bmerger(s)?\b",
    r"\bdivestiture(s)?\b",
    r"\bbudget(s)?\b",
    r"\brevenue\b",
    r"\bprofit\b",
    r"\bmargin(s)?\b",
    r"\bebitda\b",
    r"\bcapital\s+investment\b",
    r"\bcapital\s+allocation\b",
    r"\bcost\s+reduction(s)?\b",
    r"\bcost\s+cutting\b",
    r"\bfinancial\s+plan(s)?\b",
    r"\boperating\s+plan(s)?\b",
    r"\btarget(s)?\b",
    r"\bforecast(s)?\b",
    r"\bprojection(s)?\b",
    r"\bpipeline\s+growth\b",
    r"\bmarket\s+expansion\b",
    r"\bstrategic\s+plan(s)?\b",
]


EXPLICIT_BANNED_PHRASES = [
    r"\bfuture\s+hiring\s+plan(s)?\b",
    r"\bplanned\s+layoffs?\b",
    r"\bheadcount\s+plan(s)?\b",
    r"\bexpansion\s+plan(s)?\b",
    r"\bnext\s+year('?s)?\s+budget\b",
    r"\bnext\s+year('?s)?\s+revenue\b",
    r"\brevenue\s+projection(s)?\b",
    r"\bprojected\s+revenue\b",
    r"\bforecasted\s+revenue\b",
    r"\bprofit\s+forecast(s)?\b",
    r"\bebitda\s+forecast(s)?\b",
    r"\bmargin\s+forecast(s)?\b",
    r"\bcapital\s+investment\s+roadmap\b",
    r"\bplanned\s+restructuring\b",
    r"\bplanned\s+acquisition(s)?\b",
    r"\bplanned\s+closure(s)?\b",
    r"\bplanned\s+opening(s)?\b",
    r"\bstrategic\s+roadmap\b",
    r"\bthree[- ]year\s+plan\b",
    r"\bfive[- ]year\s+plan\b",
    r"\bforward[- ]looking\s+statement(s)?\b",
]


def _find_matches(patterns: List[str], text: str) -> List[str]:
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append(match.group(0))
    return list(dict.fromkeys(matches))  # dedupe while preserving order


def _get_text_window(text: str, phrase: str, window: int = 120) -> str:
    lowered = text.lower()
    idx = lowered.find(phrase.lower())
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end = min(len(text), idx + len(phrase) + window)
    return text[start:end].replace("\n", " ")


def validate_document_text(text: str) -> Dict:
    """
    Rejects documents that appear to contain future-looking company plans.
    Returns:
    {
        "allowed": bool,
        "reason": str,
        "matches": List[str],
        "details": Dict
    }
    """
    if not text or not text.strip():
        return {
            "allowed": False,
            "reason": "No readable text could be extracted from this file.",
            "matches": [],
            "details": {"error": "empty_text"},
        }

    normalized_text = " ".join(text.split())

    explicit_matches = _find_matches(EXPLICIT_BANNED_PHRASES, normalized_text)
    future_matches = _find_matches(FUTURE_TERMS, normalized_text)
    sensitive_matches = _find_matches(SENSITIVE_PLAN_TERMS, normalized_text)

    # Strong reject: explicit banned phrases
    if explicit_matches:
        return {
            "allowed": False,
            "reason": "Upload rejected: document appears to contain future company plans.",
            "matches": explicit_matches[:10],
            "details": {
                "explicit_matches": explicit_matches[:20],
                "future_matches": future_matches[:20],
                "sensitive_matches": sensitive_matches[:20],
                "rule": "explicit_banned_phrase",
            },
        }

    # Strong reject: both future language and sensitive business-plan language
    if future_matches and sensitive_matches:
        examples = []
        for phrase in (future_matches[:3] + sensitive_matches[:3]):
            window = _get_text_window(normalized_text, phrase)
            if window:
                examples.append(window)

        return {
            "allowed": False,
            "reason": "Upload rejected: document appears to contain future-looking company strategy, staffing, operational, or financial plans.",
            "matches": (future_matches[:5] + sensitive_matches[:5])[:10],
            "details": {
                "explicit_matches": explicit_matches[:20],
                "future_matches": future_matches[:20],
                "sensitive_matches": sensitive_matches[:20],
                "examples": examples[:5],
                "rule": "future_plus_sensitive",
            },
        }

    return {
        "allowed": True,
        "reason": "",
        "matches": [],
        "details": {
            "explicit_matches": [],
            "future_matches": future_matches[:20],
            "sensitive_matches": sensitive_matches[:20],
            "rule": "allow",
        },
    }