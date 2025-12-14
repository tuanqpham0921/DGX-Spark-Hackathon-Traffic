"""Robust JSON parsing utilities for LLM responses."""

import json
import re
import logging

logger = logging.getLogger(__name__)


def extract_json_from_response(response_text: str) -> dict:
    """
    Extract and parse JSON from LLM response, handling various formats.
    
    Args:
        response_text: Raw text from LLM
        
    Returns:
        Parsed JSON dictionary, or a default structure if parsing fails
    """
    if not response_text or not response_text.strip():
        logger.warning("Empty response from LLM")
        return {}
    
    # Try direct JSON parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON block in markdown code fence
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON object
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # If all else fails, log and return empty dict
    logger.error(f"Could not parse JSON from response: {response_text[:200]}...")
    return {}


def safe_parse_json(response_text: str, default_structure: dict = None) -> dict:
    """
    Safely parse JSON with fallback to default structure.
    
    Args:
        response_text: Raw text from LLM
        default_structure: Default dict to return if parsing fails
        
    Returns:
        Parsed JSON or default structure
    """
    result = extract_json_from_response(response_text)
    
    if not result and default_structure:
        logger.warning("Using default structure due to JSON parsing failure")
        return default_structure
    
    return result

