"""
Fact Extractor

Extracts structured facts from conversational text using spaCy NER.
Calls the pragmatics service for entity extraction instead of regex.

For structured documents (runbooks with tables/key-value pairs), also
parses markdown tables and key: value lines locally.
"""

import os
import re
import logging
from typing import List, Dict, Optional

import requests

logger = logging.getLogger("memory.fact_extractor")

# Pragmatics service URL for NER
PRAGMATICS_API_URL = os.getenv("PRAGMATICS_API_URL", "http://pragmatics_api:8001")


def extract_facts(text: str) -> List[Dict[str, str]]:
    """
    Extract named entities from text via pragmatics NER service.
    
    Returns list of {type, value} dicts for detected entities.
    Falls back gracefully if service is unavailable.
    """
    if not text or len(text.strip()) < 3:
        return []
    
    facts = []
    
    try:
        # Call pragmatics NER endpoint
        resp = requests.post(
            f"{PRAGMATICS_API_URL}/api/pragmatics/entities",
            json={"text": text},
            timeout=10,
        )
        
        if resp.status_code == 200:
            entities = resp.json()
            
            # Convert NER results to fact format
            for name in entities.get("names", []):
                facts.append({"type": "person_name", "value": name})
            
            for org in entities.get("organizations", []):
                facts.append({"type": "organization", "value": org})
            
            for email in entities.get("emails", []):
                facts.append({"type": "email", "value": email})
            
            for date in entities.get("dates", []):
                facts.append({"type": "date", "value": date})
            
            for loc in entities.get("locations", []):
                facts.append({"type": "location", "value": loc})
            
            for money in entities.get("money", []):
                facts.append({"type": "money", "value": money})
            
            for time in entities.get("times", []):
                facts.append({"type": "time", "value": time})
            
            logger.debug(f"NER extracted {len(facts)} facts from text")
        else:
            logger.warning(f"NER service returned {resp.status_code}")
    
    except requests.RequestException as e:
        logger.warning(f"NER service unavailable: {e}")
    except Exception as e:
        logger.error(f"NER extraction failed: {e}")
    
    return facts


def extract_facts_from_document(text: str) -> List[Dict[str, str]]:
    """
    For longer docs (runbooks, etc.)—gets NER facts plus tables and key:value lines.
    """
    facts = []
    
    # Get NER-based facts
    facts.extend(extract_facts(text))
    
    # Extract from markdown tables
    table_facts = _extract_from_tables(text)
    facts.extend(table_facts)
    
    # Extract key-value pairs (common in runbooks)
    kv_facts = _extract_key_value_pairs(text)
    facts.extend(kv_facts)
    
    # Dedupe
    seen = set()
    unique_facts = []
    for f in facts:
        key = (f["type"], f["value"].lower())
        if key not in seen:
            seen.add(key)
            unique_facts.append(f)
    
    return unique_facts


def _extract_from_tables(text: str) -> List[Dict[str, str]]:
    """Pull facts from markdown tables. First row = headers, rest = data."""
    facts = []
    
    # Find markdown tables (lines with |)
    lines = text.split('\n')
    in_table = False
    headers = []
    
    for line in lines:
        line = line.strip()
        if '|' in line and not line.startswith('```'):
            cells = [c.strip() for c in line.split('|') if c.strip()]
            
            # Skip separator lines
            if all(re.match(r'^[-:]+$', c) for c in cells):
                continue
            
            if not in_table:
                # First row = headers
                headers = [h.lower().replace(' ', '_') for h in cells]
                in_table = True
            else:
                # Data row
                for i, cell in enumerate(cells):
                    if i < len(headers) and cell and not cell.startswith('-'):
                        fact_type = f"table_{headers[i]}"
                        facts.append({"type": fact_type, "value": cell})
        else:
            in_table = False
            headers = []
    
    return facts


def _extract_key_value_pairs(text: str) -> List[Dict[str, str]]:
    """Grab Key: Value or Key = Value lines. Skips URLs and long values."""
    facts = []
    
    # Pattern for "Key: Value" or "Key = Value"
    kv_pattern = r'^([A-Za-z][A-Za-z0-9 _-]{2,30})[:=]\s*(.+)$'
    
    for line in text.split('\n'):
        line = line.strip()
        match = re.match(kv_pattern, line)
        if match:
            key = match.group(1).strip().lower().replace(' ', '_').replace('-', '_')
            value = match.group(2).strip()
            
            # Skip if value looks like a URL, code, or is too long
            if value and len(value) < 200 and not value.startswith('http'):
                facts.append({"type": f"kv_{key}", "value": value})
    
    return facts


def format_facts_for_storage(facts: List[Dict[str, str]]) -> str:
    """Turn facts into a readable string for storage (one per line, type: value)."""
    if not facts:
        return ""
    
    lines = []
    for f in facts:
        lines.append(f"{f['type']}: {f['value']}")
    
    return "\n".join(lines)


def facts_to_embedding_text(facts: List[Dict[str, str]], original_text: str = "") -> str:
    """Build text optimized for embedding—facts in readable form plus context snippet."""
    parts = []
    
    # Add formatted facts
    for f in facts:
        # Convert type to readable form
        readable_type = f['type'].replace('_', ' ').replace('kv ', '')
        parts.append(f"{readable_type}: {f['value']}")
    
    # If we have original text, extract key noun phrases
    if original_text and len(original_text) > 100:
        # Add first sentence or header as context
        first_line = original_text.split('\n')[0][:200]
        if first_line:
            parts.append(f"context: {first_line}")
    
    return " | ".join(parts)
