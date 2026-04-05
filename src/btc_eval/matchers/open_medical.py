"""Open Medical Concept Matcher — UMLS-free alternative.

Uses MeSH vocabulary keywords + optional scispaCy for medical concept extraction.
Drop-in replacement for QuickUMLS that requires no license.
"""

import hashlib
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Curated MeSH vocabulary — maps lowercase term to MeSH descriptor ID
MESH_VOCABULARY: dict[str, str] = {
    # Cardiovascular
    "chest pain": "D002637",
    "heart attack": "D009203",
    "myocardial infarction": "D009203",
    "hypertension": "D006973",
    "high blood pressure": "D006973",
    "ecg": "D004562",
    "electrocardiogram": "D004562",
    "troponin": "D020107",
    "atrial fibrillation": "D001281",
    "heart failure": "D006333",
    "angina": "D000787",
    "tachycardia": "D013610",
    "bradycardia": "D001919",
    "palpitations": "D006323",
    # Endocrine
    "diabetes": "D003920",
    "diabetes mellitus": "D003920",
    "type 2 diabetes": "D003924",
    "metformin": "D008687",
    "glucose": "D005947",
    "blood sugar": "D005947",
    "insulin": "D007328",
    "thyroid": "D013961",
    "hypothyroidism": "D007037",
    "hyperthyroidism": "D006980",
    # Respiratory
    "fever": "D005334",
    "cough": "D003371",
    "dyspnea": "D004417",
    "shortness of breath": "D004417",
    "pneumonia": "D011014",
    "asthma": "D001249",
    "bronchitis": "D001991",
    "copd": "D029424",
    "wheezing": "D012135",
    "sputum": "D013183",
    "oxygen saturation": "D010100",
    # Genitourinary
    "kidney": "D007668",
    "bladder": "D001743",
    "urination": "D014554",
    "prostate": "D011467",
    "bph": "D011470",
    "doxazosin": "D017292",
    "hematuria": "D006417",
    "urinary tract infection": "D014552",
    # Gastrointestinal
    "nausea": "D009325",
    "vomiting": "D014839",
    "diarrhea": "D003967",
    "abdominal pain": "D015746",
    "constipation": "D003248",
    # Neurological
    "headache": "D006261",
    "dizziness": "D004244",
    "seizure": "D012640",
    "numbness": "D006987",
    "stroke": "D020521",
    # Musculoskeletal
    "back pain": "D001416",
    "joint pain": "D018771",
    "arthritis": "D001168",
    "fracture": "D050723",
    # Dermatological
    "rash": "D005076",
    "itching": "D011537",
    "swelling": "D004487",
    # Mental health
    "anxiety": "D001007",
    "depression": "D003863",
    "insomnia": "D007319",
    # General
    "medication": "D004364",
    "treatment": "D013812",
    "diagnosis": "D003933",
    "patient": "D010361",
    "symptoms": "D013568",
    "allergy": "D006967",
    "pain": "D010146",
    "fatigue": "D005221",
    "weight loss": "D015431",
    "infection": "D007239",
    "antibiotic": "D000900",
    "blood pressure": "D001794",
    "heart rate": "D006339",
    "temperature": "D001831",
    "x-ray": "D014057",
    "ct scan": "D014057",
    "mri": "D008279",
    "ultrasound": "D014463",
    "biopsy": "D001706",
    "surgery": "D013514",
    "anesthesia": "D000758",
    "vaccination": "D014611",
    "immunization": "D014611",
    "referral": "D012017",
    "follow-up": "D005500",
}


class OpenMedicalMatcher:
    """Drop-in replacement for QuickUMLS using open medical vocabularies."""

    def __init__(self, use_mesh: bool = True, use_scispacy: bool = True):
        self.use_mesh = use_mesh
        self.use_scispacy = use_scispacy
        self.mesh_concepts: dict[str, str] = {}
        self.nlp = None

        if use_mesh:
            self._load_mesh_vocabulary()
        if use_scispacy:
            self._load_scispacy()

    @property
    def scispacy_active(self) -> bool:
        """Whether scispaCy NER is actually loaded and active."""
        return self.nlp is not None

    def _load_mesh_vocabulary(self):
        """Load built-in MeSH vocabulary."""
        self.mesh_concepts = MESH_VOCABULARY.copy()
        log.info("Loaded %d MeSH concepts", len(self.mesh_concepts))

    def _load_scispacy(self):
        """Load scispaCy biomedical NER model."""
        try:
            import spacy

            self.nlp = spacy.load("en_core_sci_md")
            log.info("Loaded scispaCy biomedical model")
        except ImportError:
            log.warning(
                "spaCy not installed — scispaCy disabled. Install with: pip install spacy scispacy"
            )
            self.nlp = None
        except OSError:
            log.warning(
                "scispaCy model not found — scispaCy disabled. "
                "Install with: pip install "
                "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                "releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz"
            )
            self.nlp = None
        except Exception as e:
            # Known issue: en_core_sci_md 0.5.4 config has 'True' (string) not
            # true (bool), which breaks with spaCy 3.8+ / newer confection.
            log.warning("scispaCy model loading failed (%s), attempting config fix...", e)
            try:
                import en_core_sci_md

                cfg_path = (
                    Path(en_core_sci_md.__file__).parent
                    / "en_core_sci_md-0.5.4"
                    / "config.cfg"
                )
                text = cfg_path.read_text()
                patched = re.sub(
                    r'(include_static_vectors\s*=\s*)["\']True["\']',
                    r"\1true",
                    text,
                )
                patched = re.sub(
                    r'(include_static_vectors\s*=\s*)["\']False["\']',
                    r"\1false",
                    patched,
                )
                if patched != text:
                    cfg_path.write_text(patched)
                    self.nlp = spacy.load("en_core_sci_md")
                    log.info(
                        "Loaded scispaCy after patching model config (fixed bool in config.cfg)"
                    )
                else:
                    raise
            except Exception as retry_err:
                raise RuntimeError(
                    "Failed to load scispaCy model 'en_core_sci_md'. "
                    "This is required for Open Medical concept matching. "
                    "Try reinstalling: uv pip install --force-reinstall "
                    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                    "releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz\n"
                    f"Original error: {retry_err}"
                ) from retry_err

    def match(self, text: str) -> list[list[dict]]:
        """Extract medical concepts from text.

        Returns QuickUMLS-compatible format: [[{concept_info}, ...]].
        """
        text_lower = text.lower().strip()
        all_concepts = []

        if self.use_mesh:
            all_concepts.extend(self._extract_mesh_concepts(text_lower))

        if self.use_scispacy and self.nlp:
            all_concepts.extend(self._extract_scispacy_concepts(text_lower))

        if all_concepts:
            return [all_concepts]
        return []

    def _extract_mesh_concepts(self, text: str) -> list[dict]:
        """Extract concepts using MeSH vocabulary keyword matching."""
        found = []
        for term, mesh_id in self.mesh_concepts.items():
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                found.append(
                    {
                        "cui": mesh_id,
                        "term": term,
                        "similarity": 1.0,
                        "source": "mesh",
                    }
                )
        return found

    def _extract_scispacy_concepts(self, text: str) -> list[dict]:
        """Extract concepts using scispaCy biomedical NER."""
        found = []
        doc = self.nlp(text)
        for ent in doc.ents:
            # en_core_sci_md labels all biomedical entities as 'ENTITY' —
            # accept any label the model produces
            if ent.label_:
                cui = "SPACY_" + hashlib.md5(ent.text.lower().encode()).hexdigest()[:8]
                found.append(
                    {
                        "cui": cui,
                        "term": ent.text.lower(),
                        "similarity": 0.85,
                        "source": "scispacy",
                        "entity_type": ent.label_,
                    }
                )
        return found


def config_open_medical_matcher(
    use_mesh: bool = True, use_scispacy: bool = True
) -> OpenMedicalMatcher:
    """Factory function — create an OpenMedicalMatcher instance."""
    return OpenMedicalMatcher(use_mesh=use_mesh, use_scispacy=use_scispacy)
