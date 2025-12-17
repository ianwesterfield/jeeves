"""
Fact Extractor

Pulls structured facts out of conversational text via regex. Things like
"My name is Ian" become {type: user_first_name, value: Ian}. Handles
personal info, family, work, contact, dates, preferences, and tech stuff
(VLANs, IPs, domains). Also parses markdown tables and key: value lines
for runbook-style docs.
"""

import re
from typing import List, Dict, Optional

# My fact patterns: (regex, fact_type, group_index_for_value)
FACT_PATTERNS = [
    # Names - MUST have possessive/intro to avoid matching things like "Name resolution"
    (r"(?:my )(first name|name) (?:is |'s |= )?([A-Z][a-z]+)", "user_first_name", 2),
    (r"(?:my )(last name|surname|family name) (?:is |'s |= )?([A-Z][a-z]+)", "user_last_name", 2),
    (r"(?:i'm |i am |my name is |call me )([A-Z][a-z]+(?: [A-Z][a-z]+)?)", "user_name", 1),
    
    # Nicknames/preferences for how to be addressed or refer to things
    (r"(?:call you |i'?ll call you |refer to you as |your name is )\"?([A-Za-z][A-Za-z0-9 ]*?)\"?(?:\.|,|$| from| in| because)", "assistant_nickname", 1),
    (r"(?:call me |refer to me as |address me as )\"?([A-Za-z][A-Za-z0-9 ]*?)\"?(?:\.|,|$| from| in| because)", "user_nickname", 1),
    
    # Family
    (r"(?:my )?(daughter|son|wife|husband|partner|child)(?:'s)? (?:name )?(?:is |'s )?([A-Z][a-z]+)", "family_{0}", 2),
    (r"(?:my )?(mother|father|mom|dad|brother|sister)(?:'s)? (?:name )?(?:is |'s )?([A-Z][a-z]+)", "family_{0}", 2),
    
    # Work/Professional
    (r"(?:i |I )(?:work at|work for|am employed at|am employed by) ([A-Z][A-Za-z0-9 &,.-]+?)(?:\.|,|$| which| and| as)", "employer", 1),
    (r"(?:i'm |i am |my job is |my role is |my title is |i work as )(?:a |an )?([A-Za-z]+(?: [A-Za-z]+){0,3}?)(?:\.|,|$| at| for)", "job_title", 1),
    
    # Contact
    (r"(?:my )?(?:email|e-mail)(?: address)? (?:is |'s |= )?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", "email", 1),
    (r"(?:my )?(?:phone|mobile|cell)(?: number)? (?:is |'s |= )?([+]?[\d\s()-]{10,})", "phone", 1),
    
    # Location
    (r"(?:i |I )(?:live in|am from|am located in|reside in) ([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+){0,3})", "location", 1),
    (r"(?:my )?(?:address|home) (?:is |'s |= )?(.+?)(?:\.|$)", "address", 1),
    
    # Dates
    (r"(?:my )?(?:birthday|birth date|DOB) (?:is |'s |= )?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2}(?:st|nd|rd|th)?,? \d{4})", "birthday", 1),
    (r"(?:i was |I was )?born (?:on |in )?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2}(?:st|nd|rd|th)?,? \d{4})", "birthday", 1),
    
    # Preferences
    (r"(?:my )?(?:favorite|favourite|preferred) (color|colour|food|movie|book|sport|team|music|song|artist) (?:is |'s |= )?([A-Za-z0-9 ]+?)(?:\.|,|$| and| but)", "favorite_{0}", 2),
    (r"(?:i |I )(?:like|love|enjoy|prefer) ([A-Za-z0-9 ]+?)(?:\.|,|$| and| but| because| when)", "preference", 1),
    
    # Technical/Infrastructure (for runbooks, etc.)
    (r"(?:VLAN|vlan) ?(\d+)(?: (?:is |for |= )?(?:the )?)?([A-Za-z]+(?:[ /][A-Za-z]+)?)?", "vlan_{0}", 2),
    (r"(?:IP|ip)(?: address)?(?: (?:is |= ))?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?)", "ip_address", 1),
    (r"(?:subnet|network) (?:is |= )?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2})", "subnet", 1),
    (r"([A-Za-z0-9.-]+\.(?:cloud|local|lan|home|internal)) ", "domain", 1),
]

# Headers that usually mean structured content is coming
SECTION_PATTERNS = [
    (r"^#+\s*(.+)$", "section"),  # Markdown headers
    (r"^\*\*(.+)\*\*:?\s*$", "section"),  # Bold headers
    (r"^([A-Z][A-Za-z ]+):\s*$", "section"),  # Title case headers
]


def extract_facts(text: str) -> List[Dict[str, str]]:
    """
    Run my regex patterns over the text and pull out facts.
    Returns list of {type, value} dicts. Dedupes automatically.
    """
    if not text:
        return []
    
    facts = []
    seen = set()  # Dedupe by (type, value)
    
    # Apply each pattern
    for pattern, fact_type, group_idx in FACT_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            try:
                # Handle dynamic fact types like "family_{0}" or "favorite_{0}"
                if "{0}" in fact_type:
                    qualifier = match.group(1).lower().replace(" ", "_")
                    resolved_type = fact_type.format(qualifier)
                else:
                    resolved_type = fact_type
                
                value = match.group(group_idx).strip()
                
                # Clean up value
                value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
                value = value.rstrip('.,;:')  # Remove trailing punctuation
                
                if value and len(value) > 1:
                    key = (resolved_type, value.lower())
                    if key not in seen:
                        seen.add(key)
                        facts.append({
                            "type": resolved_type,
                            "value": value
                        })
            except (IndexError, AttributeError):
                continue
    
    return facts


def extract_facts_from_document(text: str) -> List[Dict[str, str]]:
    """For longer docs (runbooks, etc.)—gets pattern-based facts plus tables and key:value lines."""
    facts = []
    
    # First get pattern-based facts
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
