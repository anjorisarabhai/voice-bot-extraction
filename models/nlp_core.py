# models/nlp_core.py
import re
import time
from models.schema import VisitDetails
from typing import Tuple, Dict, Any, Optional

# Define patterns to trigger fallback (TEMPORAL PATTERNS REMOVED)
COMPLEX_PATTERNS = []

# Define action/preposition markers to stop the name extraction
START_MARKERS = ['with', 'for']
STOP_MARKERS = ['to', 'regarding', 'for', 'about', 'on', 'at', 'business', 'operation', 'discuss', 'review', 'close', 'account', 'structure']


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
            
            # Capture words until a defined stop word is reached
            for word in words[start_index:]:
                if word in STOP_MARKERS:
                    break
                name_words.append(word)
            
            if name_words:
                name_candidate = ' '.join(name_words).title()
                break

    # 2. Final Validation Check
    visit_type_match = False
    
    if re.search(r'\b(business)\b', transcript, re.IGNORECASE):
        nlp_output['visit_type'] = "BUSINESS"
        visit_type_match = True
    elif re.search(r'\b(operation)\b', transcript, re.IGNORECASE):
        nlp_output['visit_type'] = "OPERATION"
        visit_type_match = True
        
    # Final Success Check: Must have found a name AND a visit type
    if name_candidate != "N/A" and visit_type_match:
        nlp_output['lead_name'] = name_candidate
        nlp_output['title'] = transcript[:40].strip() + "..."
        
        # Ensure email/phone are N/A as NLP can't extract them
        nlp_output['email'] = "N/A"
        nlp_output['phone_number'] = "N/A"
        
        return nlp_output, (time.time() - start_time)
        
    # If basic extraction failed (no name or type found), trigger LLM fallback
    return None, (time.time() - start_time)