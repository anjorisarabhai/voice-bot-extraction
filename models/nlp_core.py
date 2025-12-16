# models/nlp_core.py

import re
import time
from models.schema import VisitDetails
from typing import Tuple, Dict, Any, Optional

# --- Regex Definitions for Contact Info ---
# Simple email pattern (covers most common formats)
EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
# Simple phone pattern (looks for groups of digits, accommodating common separators)
PHONE_PATTERN = r'(\+\d{1,3})?\s*\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}'

# Define patterns to trigger fallback (currently empty, favoring NLP for all simple extraction)
COMPLEX_PATTERNS = []

# Define action/preposition markers to start the name extraction
START_MARKERS = ['with', 'for']
# Final STOP_MARKERS list: Used to cleanly stop name capture before prepositions, numbers, or punctuation
STOP_MARKERS = ['to', 'regarding', 'for', 'about', 'on', 'at', 'business', 
                'operation', 'discuss', 'review', 'close', 'account', 
                'structure', '1st', '2nd', '3rd', 'th', 'st', 'nd', 'rd', 
                'tomorrow', 'today', 'next', 'previous', 'yesterday', 'pm', 'am', 
                '.', ',', 'and', 'but', 'is', 'can', 'he', 'she', 'the', 'a']


def run_nlp_fast_path(transcript: str) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Runs the fast NLP path using aggressive string indexing. 
    It now includes logic for complex entity extraction (Email/Phone) and transcription cleanup.
    """
    start_time = time.time()
    
    # --- TRANSCRIPTION CLEANUP: Convert ASR artifacts to symbols ---
    # This is critical for robust email/phone extraction via regex
    cleaned_transcript = transcript.lower().replace(' at the rate ', '@')
    cleaned_transcript = cleaned_transcript.replace(' dot ', '.')
    
    nlp_output = {field: "N/A" for field in VisitDetails.model_fields.keys()}
    
    # 1. Extract Name
    words = cleaned_transcript.split() 
    name_candidate = "N/A"
    
    for marker in START_MARKERS:
        if marker in words:
            start_index = words.index(marker) + 1
            name_words = []
            
            # Capture words until a defined stop word or a number/date artifact is reached
            for word in words[start_index:]:
                if word in STOP_MARKERS:
                    break
                if word and word[0].isdigit():
                    break
                name_words.append(word)
            
            if name_words:
                # 1. Join and Capitalize
                raw_name = ' '.join(name_words).title()
                
                # 2. FINAL CLEANUP: Remove trailing punctuation/whitespace from the name
                name_candidate = re.sub(r'[.,\s]+$', '', raw_name)
                break

    # 2. Extract Contact Info (NEW LOGIC)
    
    # Email Extraction: Search the cleaned transcript for the email pattern
    email_match = re.search(EMAIL_PATTERN, cleaned_transcript)
    if email_match:
        nlp_output['email'] = email_match.group(0)
        
    # Phone Number Extraction: Search the cleaned transcript for the phone pattern
    phone_match = re.search(PHONE_PATTERN, cleaned_transcript)
    if phone_match:
        # Clean the phone number (remove spaces, hyphens, parentheses) for database storage
        phone_number = re.sub(r'[\s()-]', '', phone_match.group(0))
        nlp_output['phone_number'] = phone_number

    # 3. Final Validation Check
    visit_type_match = False
    
    # Check for Business Type
    if re.search(r'\b(business)\b', cleaned_transcript, re.IGNORECASE):
        nlp_output['visit_type'] = "BUSINESS"
        visit_type_match = True
    # Check for Operation Type
    elif re.search(r'\b(operation)\b', cleaned_transcript, re.IGNORECASE):
        nlp_output['visit_type'] = "OPERATION"
        visit_type_match = True
        
    # Final Success Check: Must have found a name AND a visit type
    if name_candidate != "N/A" and visit_type_match:
        nlp_output['lead_name'] = name_candidate
        
        # Ensure title is truncated nicely (Use original transcript for title)
        nlp_output['title'] = transcript[:40].strip() + "..."
        
        # Email and phone numbers were populated above, or they remain 'N/A'
        
        return nlp_output, (time.time() - start_time)
        
    # If basic extraction failed (no name or type found), trigger LLM fallback
    return None, (time.time() - start_time)