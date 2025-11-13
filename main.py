# main.py
import time
import json
import os
import sys
from datetime import datetime

# Add the project root to the path for absolute imports
# This ensures imports like 'from models.nlp_core import ...' work correctly.
sys.path.append(os.path.dirname(__file__))

# Import core components and settings from the local modules
from models.nlp_core import run_nlp_fast_path
from models.llm_fallback import extract_via_mercury_fallback
from models.schema import VisitDetails 


# --- CORE HYBRID EXTRACTION PIPELINE ---

def run_hybrid_extraction_pipeline(transcript: str):
    """
    The central logic using the fast NLP path with Mercury as the production fallback.
    """
    
    # 1. Attempt FAST PATH (NLP)
    # This function returns None if the input contains temporal keywords or fails basic checks.
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

# --- TEST CASE HANDLER ---

def get_benchmark_tests():
    """Loads benchmark tests from the JSON file."""
    # Construct the path relative to the script's execution
    test_file_path = os.path.join(os.path.dirname(__file__), 'tests', 'test_cases.json')
    
    try:
        with open(test_file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Test case file not found at {test_file_path}")
        return []
    except json.JSONDecodeError:
        print("ERROR: Failed to decode JSON from test case file. Check syntax.")
        return []

# --- MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    TEST_CASES = get_benchmark_tests()
    FINAL_REPORT = []
    
    if not TEST_CASES:
        print("Aborting benchmark: No test cases loaded. Ensure tests/test_cases.json exists.")
        # We exit with a non-zero code to indicate failure
        sys.exit(1)
        
    print(f"\n--- Running FINAL {len(TEST_CASES)}-CASE BENCHMARK ---")

    for i, test_case in enumerate(TEST_CASES): 
        transcript = test_case['transcript']
        print(f"[Test {i+1}/{len(TEST_CASES)}]: {transcript[:60]}...")

        # Run the Hybrid Pipeline
        metrics = run_hybrid_extraction_pipeline(transcript)
        
        # Log the result
        FINAL_REPORT.append({
            "Test_ID": test_case.get('id', i + 1), # Use 'id' from JSON or index
            "Transcript": transcript,
            "Extraction_Method": metrics["method"],
            "Success": metrics["success"],
            "Latency_sec": metrics["latency_sec"],
            "Data": metrics["data"],
        })
    
    # --- FINAL REPORT GENERATION ---
    print("\n\n#####################################################")
    print("##### FINAL 15-CASE DATA EXTRACTION REPORT #####")
    print("#####################################################")

    columns = ["Test_ID", "Extraction_Method", "Success", "Latency_sec", "Transcript", 
               "Lead_Name", "Visit_Type", "Date", "Start_Time", "Email", "Phone"]
    print("\t".join(columns))
    print("---------------------------------------------------------------------------------------------------------------------------------")

    for entry in FINAL_REPORT:
        data = entry["Data"]
        
        # --- CRITICAL FIX: Defensive Reporting Check ---
        if data is None:
            # If data is None (LLM failed), use a placeholder for all fields
            row_data = [
                str(entry["Test_ID"]),
                entry["Extraction_Method"],
                str(entry["Success"]),
                f"{entry['Latency_sec']:.4f}",
                entry["Transcript"],
                "API_FAILURE", # Explicitly mark the failure
                "N/A", "N/A", "N/A", "N/A", "N/A"
            ]
        else:
            # Normal processing if data is a valid dictionary
            row_data = [
                str(entry["Test_ID"]),
                entry["Extraction_Method"],
                str(entry["Success"]),
                f"{entry['Latency_sec']:.4f}",
                entry["Transcript"],
                data.get("lead_name", "N/A"),
                data.get("visit_type", "N/A"),
                data.get("date", "N/A"),
                data.get("start_time", "N/A"),
                data.get("email", "N/A"),
                data.get("phone_number", "N/A"),
            ]
        print("\t".join(row_data))

    print("---------------------------------------------------------------------------------------------------------------------------------")