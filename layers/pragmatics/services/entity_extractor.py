"""
Entity Extractor - spaCy-based Named Entity Recognition

Extracts structured entities from text using spaCy's pre-trained NER models.
Replaces regex-based extraction throughout Jeeves.

Supported entities:
  - PERSON: Names of people
  - ORG: Organizations, companies
  - DATE: Dates and date expressions
  - EMAIL: Email addresses (custom pattern)
  - GPE: Geopolitical entities (countries, cities)
  - MONEY: Monetary values
  - TIME: Times of day
"""

import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import spacy
from spacy.language import Language


logger = logging.getLogger("pragmatics.entity_extractor")

# Singleton model instance
_nlp: Optional[Language] = None


def _get_nlp() -> Language:
    """Lazy-load spaCy model (en_core_web_sm is ~12MB)."""
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy model: en_core_web_sm")
        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded successfully")
    return _nlp


@dataclass
class ExtractedEntities:
    """Container for extracted entities."""
    names: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    money: List[str] = field(default_factory=list)
    times: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "names": self.names,
            "organizations": self.organizations,
            "dates": self.dates,
            "emails": self.emails,
            "locations": self.locations,
            "money": self.money,
            "times": self.times,
        }
    
    def has_any(self) -> bool:
        """Check if any entities were extracted."""
        return any([
            self.names, self.organizations, self.dates,
            self.emails, self.locations, self.money, self.times
        ])


# Email regex - spaCy doesn't extract emails natively
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)


def extract_entities(text: str) -> ExtractedEntities:
    """
    Extract named entities from text using spaCy NER.
    
    Args:
        text: Input text to analyze
        
    Returns:
        ExtractedEntities with categorized entity lists
    """
    nlp = _get_nlp()
    doc = nlp(text)
    
    entities = ExtractedEntities()
    
    # Process spaCy entities
    for ent in doc.ents:
        entity_text = ent.text.strip()
        
        if ent.label_ == "PERSON":
            # Deduplicate and clean names
            if entity_text and entity_text not in entities.names:
                entities.names.append(entity_text)
                
        elif ent.label_ == "ORG":
            if entity_text and entity_text not in entities.organizations:
                entities.organizations.append(entity_text)
                
        elif ent.label_ == "DATE":
            if entity_text and entity_text not in entities.dates:
                entities.dates.append(entity_text)
                
        elif ent.label_ in ("GPE", "LOC"):  # Geopolitical and Location
            if entity_text and entity_text not in entities.locations:
                entities.locations.append(entity_text)
                
        elif ent.label_ == "MONEY":
            if entity_text and entity_text not in entities.money:
                entities.money.append(entity_text)
                
        elif ent.label_ == "TIME":
            if entity_text and entity_text not in entities.times:
                entities.times.append(entity_text)
    
    # Extract emails (spaCy doesn't do this)
    email_matches = EMAIL_PATTERN.findall(text)
    entities.emails = list(set(email_matches))
    
    logger.debug(
        f"Extracted: {len(entities.names)} names, {len(entities.organizations)} orgs, "
        f"{len(entities.dates)} dates, {len(entities.emails)} emails"
    )
    
    return entities


def extract_entities_dict(text: str) -> Dict[str, List[str]]:
    """
    Extract entities and return as dictionary.
    
    Convenience wrapper for API responses.
    """
    return extract_entities(text).to_dict()


def extract_user_info(text: str) -> Dict[str, Optional[str]]:
    """
    Extract user-specific info commonly needed by Jeeves.
    
    Returns first found name/email, or None if not found.
    Useful for populating orchestrator workspace state.
    """
    entities = extract_entities(text)
    
    return {
        "name": entities.names[0] if entities.names else None,
        "email": entities.emails[0] if entities.emails else None,
        "organization": entities.organizations[0] if entities.organizations else None,
    }
