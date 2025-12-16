# models/nlp_core.py
import re
import time
from models.schema import VisitDetails
from typing import Tuple, Dict, Any, Optional

# Define patterns to trigger fallback (TEMPORAL PATTERNS REMOVED)
COMPLEX_PATTERNS = []

# Define action/preposition markers to stop the name extraction
START_MARKERS = ['with', 'for']
# Final STOP_MARKERS list
STOP_MARKERS = ['to', 'regarding', 'for', 'about', 'on', 'at', 'business', 
                'operation', 'discuss', 'review', 'close', 'account', 
                'structure', '1st', '2nd', '3rd', 'th', 'st', 'nd', 'rd', 
                'tomorrow', 'today', 'next', 'previous', 'yesterday', 'pm', 'am', 
                '.', ',', 'and', 'but', 'is', 'can', 'he', 'she', 'the', 'a']


def run_nlp_fast_path(transcript: str) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Runs the fast NLP path using aggressive string indexing. 
    It will succeed if Name/Type are found and no temporal data is in the COMPLEX_PATTERNS list.
    """
    start_time = time.time()
    nlp_output = {field: "N/A" for field in VisitDetails.model_fields.keys()}
    
    # Note: The lack of temporal checks means the fast path handles virtually all inputs now.
    
    # 1. Extract Basic Fields (Aggressive Name Capture)
    words = transcript.lower().split()
    name_candidate = "N/A"
    
    for marker in START_MARKERS:
        if marker in words:
            start_index = words.index(marker) + 1
            name_words = []
            
            # Capture words until a defined stop word or a number/date artifact is reached
            for word in words[start_index:]:
                # Check 1: Stop if word is a defined marker from the large list
                if word in STOP_MARKERS:
                    break
                
                # Check 2: Stop if word starts with a digit (e.g., '10th', '12')
                if word and word[0].isdigit():
                    break
                    
                name_words.append(word)
            
            if name_words:
                # 1. Join and Capitalize
                raw_name = ' '.join(name_words).title()
                
                # 2. FINAL CLEANUP: Remove trailing punctuation (periods, commas, etc.)
                # This uses regex to remove any punctuation or whitespace at the end of the string.
                name_candidate = re.sub(r'[.,\s]+$', '', raw_name)
                break

    # 2. Final Validation Check
    visit_type_match = False
    
    # Check for Business Type
    if re.search(r'\b(business)\b', transcript, re.IGNORECASE):
        nlp_output['visit_type'] = "BUSINESS"
        visit_type_match = True
    # Check for Operation Type
    elif re.search(r'\b(operation)\b', transcript, re.IGNORECASE):
        nlp_output['visit_type'] = "OPERATION"
        visit_type_match = True
        
    # Final Success Check: Must have found a name AND a visit type
    if name_candidate != "N/A" and visit_type_match:
        nlp_output['lead_name'] = name_candidate
        
        # Ensure title is truncated nicely
        nlp_output['title'] = transcript[:40].strip() + "..."
        
        # Ensure email/phone are N/A as NLP can't extract them
        nlp_output['email'] = "N/A"
        nlp_output['phone_number'] = "N/A"
        
        return nlp_output, (time.time() - start_time)
        
    # If basic extraction failed (no name or type found), trigger LLM fallback
    return None, (time.time() - start_time)