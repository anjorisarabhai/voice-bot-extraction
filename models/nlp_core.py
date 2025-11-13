# models/nlp_core.py
import re
import time
from models.schema import VisitDetails

# Define patterns to trigger fallback (any temporal data)
COMPLEX_PATTERNS = [
    r'\b(today|tomorrow|next|day|week|month|year|am|pm|\d{1,2}(:|\s(am|pm)))\b', 
    r'\d{1,2}(st|nd|rd|th)\b', 
    r'\d{4}-\d{2}-\d{2}', 
    r'(end|ending|later|after)\b'
]
# Define keywords to help with name extraction
START_MARKERS = ['with', 'for']
STOP_MARKERS = ['to', 'regarding', 'for', 'about', 'on', 'at', 'business', 'operation', 'discuss', 'review', 'close', 'account', 'structure']


def run_nlp_fast_path(transcript: str):
    """
    Runs the fast NLP path using aggressive string indexing.
    Returns extracted data (dict) and latency (float).
    """
    start_time = time.time()
    nlp_output = {field: "N/A" for field in VisitDetails.model_fields.keys()}
    
    # 1. Check for Complex Patterns (If found, we immediately fail the fast path)
    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, transcript, re.IGNORECASE):
            return None, (time.time() - start_time) # Fail fast: needs LLM
            
    # 2. Extract Basic Fields (Aggressive Name Capture)
    words = transcript.lower().split()
    name_candidate = "N/A"
    
    for marker in START_MARKERS:
        if marker in words:
            start_index = words.index(marker) + 1
            name_words = []
            
            # Capture words until a defined stop word is reached
            for word in words[start_index:]:
                # Use a combined list for stop words
                if word in STOP_MARKERS:
                    break
                name_words.append(word)
            
            if name_words:
                # Basic capitalization to handle names
                name_candidate = ' '.join(name_words).title()
                break

    # 3. Final Validation Check
    visit_type_match = False
    
    if re.search(r'\b(business)\b', transcript, re.IGNORECASE):
        nlp_output['visit_type'] = "BUSINESS"
        visit_type_match = True
    elif re.search(r'\b(operation)\b', transcript, re.IGNORECASE):
        nlp_output['visit_type'] = "OPERATION"
        visit_type_match = True
        
    # Final Success Check: Must have found a name AND a visit type AND not have failed temporal check
    if name_candidate != "N/A" and visit_type_match:
        nlp_output['lead_name'] = name_candidate
        nlp_output['title'] = transcript[:40].strip() + "..."
        
        return nlp_output, (time.time() - start_time)
        
    return None, (time.time() - start_time)