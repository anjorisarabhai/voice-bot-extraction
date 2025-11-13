# main.py
import time
import json
from models.nlp_core import run_nlp_fast_path # <-- This import is fine now
from models.llm_fallback import extract_via_mercury_fallback

# --- CORE HYBRID EXTRACTION PIPELINE (Production Ready) ---
def run_hybrid_extraction_pipeline(transcript: str):
    """
    The central logic using the fast NLP path with Mercury as the production fallback.
    """
    
    # 1. Attempt FAST PATH (NLP)
    nlp_data, nlp_latency = run_nlp_fast_path(transcript)
    
    if nlp_data:
        # NLP SUCCESS: Return data from the fast path
        total_latency = nlp_latency
        method_used = "NLP_RULES"
        extracted_data = nlp_data
        
    else:
        # 2. NLP FAILED or Complex Temporal Data Detected -> FALLBACK to Mercury
        
        # Execute the Mercury Fallback
        llm_data, llm_latency = extract_via_mercury_fallback(transcript)
        
        total_latency = nlp_latency + llm_latency # Total time spent
        method_used = "MERCURY_dLLM"
        extracted_data = llm_data
        
    # Final metrics assembly
    metrics = {
        "method": method_used,
        "success": extracted_data is not None,
        "latency_sec": total_latency,
        "data": extracted_data
    }
    return metrics

# --- EXAMPLE EXECUTION (Demonstration) ---

if __name__ == "__main__":
    
    # Example 1: FAST PATH TEST (No time keywords)
    fast_transcript = "Schedule a business visit with Mr. George to discuss final signatures."
    print(f"\n--- Running FAST TEST: {fast_transcript[:40]}... ---")
    fast_metrics = run_hybrid_extraction_pipeline(fast_transcript)
    
    # Example 2: SLOW PATH TEST (Complex time keyword)
    slow_transcript = "Schedule a business visit with Dr. Patel for tomorrow at 2 PM."
    print(f"\n--- Running SLOW TEST: {slow_transcript[:40]}... ---")
    slow_metrics = run_hybrid_extraction_pipeline(slow_transcript)
    
    print("\n\n--- FINAL HYBRID PIPELINE DEMO ---")
    print("-----------------------------------")
    
    # Report Fast Path
    print(f"1. FAST PATH | Method: {fast_metrics['method']} | Latency: {fast_metrics['latency_sec']:.4f}s")
    print(f"   Data: {fast_metrics['data'].get('lead_name', 'N/A')} - {fast_metrics['data'].get('date', 'N/A')}")

    # Report Slow Path
    print(f"2. SLOW PATH | Method: {slow_metrics['method']} | Latency: {slow_metrics['latency_sec']:.4f}s")
    print(f"   Data: {slow_metrics['data'].get('lead_name', 'N/A')} - {slow_metrics['data'].get('date', 'N/A')}")