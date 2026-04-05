"""Mock matcher for testing — no external dependencies."""

import re


# Small vocabulary for deterministic test scenarios
MOCK_CONCEPTS: dict[str, str] = {
    "chest pain": "C0008031",
    "dyspnea": "C0013404",
    "ecg": "C0013798",
    "troponin": "C0041199",
    "cardiac": "C0018787",
    "heart": "C0018787",
    "diabetes": "C0011849",
    "type 2 diabetes": "C0011860",
    "metformin": "C0025598",
    "glucose": "C0017725",
    "blood glucose": "C0005802",
    "hypertension": "C0020538",
    "blood pressure": "C0005823",
    "fever": "C0015967",
    "cough": "C0010200",
    "infection": "C0021311",
    "pneumonia": "C0032285",
    "prostate": "C0033572",
    "bph": "C0005001",
    "doxazosin": "C0058497",
    "medication": "C0013227",
    "treatment": "C0087111",
    "diagnosis": "C0011900",
    "symptoms": "C0683368",
    "patient": "C0030705",
}


class MockMatcher:
    """Deterministic matcher for tests — keyword matching against a small vocabulary."""

    def __init__(self, concepts: dict[str, str] | None = None):
        self.concepts = concepts or MOCK_CONCEPTS

    def match(self, text: str) -> list[list[dict]]:
        text_lower = text.lower()
        found = []
        for term, cui in self.concepts.items():
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, text_lower, re.IGNORECASE):
                found.append(
                    {
                        "cui": cui,
                        "term": term,
                        "similarity": 1.0,
                        "source": "mock",
                    }
                )
        if found:
            return [found]
        return []
