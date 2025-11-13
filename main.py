# main.py
import time
import json
from datetime import datetime
from models.schema import VisitDetails
from models.nlp_core import run_nlp_fast_path
from models.llm_fallback import extract_via_llama_fallback, extract_via_mercury_fallback

# --- CORE HYBRID EXTRACTION PIPELINE ---
def run_hybrid_extraction_pipeline(transcript: str, use_mercury_fallback=True):
    """
    The central logic that decides between the fast NLP path and the slow LLM fallback.
    """
    
    # 1. Attempt FAST PATH (NLP)
    nlp_data, nlp_latency = run_nlp_fast_path(transcript)
    
    if nlp_data:
        # NLP SUCCESS: Return data from the fast path
        total_latency = nlp_latency
        method_used = "NLP_RULES"
        extracted_data = nlp_data
        
    else:
        # 2. NLP FAILED or Complex Temporal Data Detected -> FALLBACK
        
        # Decide which slow path to use for demonstration
        if use_mercury_fallback:
            llm_func = extract_via_mercury_fallback
            llm_name = "MERCURY_dLLM"
        else:
            llm_func = extract_via_llama_fallback
            llm_name = "LLAMA3_PROXY"
        
        # Execute the Fallback
        llm_data, llm_latency = llm_func(transcript)
        
        total_latency = nlp_latency + llm_latency # Total time spent
        method_used = llm_name
        extracted_data = llm_data
        
    # Final metrics assembly
    metrics = {
        "method": method_used,
        "success": extracted_data is not None,
        "latency_sec": total_latency,
        "data": extracted_data
    }
    return metrics

# --- EXAMPLE EXECUTION (Comparison Test) ---

if __name__ == "__main__":
    
    # Example Test Cases (Forcing the fallback path)
    COMPLEX_TEST_CASES = [
        "Schedule a business visit with Dr. Patel for tomorrow at 2 PM.",
        "Book an operation visit with Tom Harris next week on Wednesday at 3 PM.",
        "Make a visit with Jane Smith on Dec 20th at 11:00 AM, ending 15 minutes later."
    ]

    for i, transcript in enumerate(COMPLEX_TEST_CASES):
        print(f"\n--- Running Test {i+1}: {transcript[:40]}... ---")
        
        # Run 1: Test with Slow Llama 3 Proxy
        llama_metrics = run_hybrid_extraction_pipeline(transcript, use_mercury_fallback=False)
        
        # Run 2: Test with Fast Mercury dLLM
        mercury_metrics = run_hybrid_extraction_pipeline(transcript, use_mercury_fallback=True)
        
        if llama_metrics['success'] and mercury_metrics['success']:
            speed_ratio = llama_metrics['latency_sec'] / mercury_metrics['latency_sec']
            
            print(f"LLAMA3 LATENCY: {llama_metrics['latency_sec']:.4f}s")
            print(f"MERCURY LATENCY: {mercury_metrics['latency_sec']:.4f}s")
            print(f"SPEEDUP: {speed_ratio:.2f}x (Mercury is faster)")
            print(f"Extracted Date (M): {mercury_metrics['data'].get('date', 'N/A')}")
        else:
            print("One or both fallbacks failed.")