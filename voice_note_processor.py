# voice_note_processor.py
import time
import json
import sys
import os

# Add the project root to the path for absolute imports
sys.path.append(os.path.dirname(__file__))

# Import the necessary logic and setup models
from models.llm_fallback import extract_context_and_summarize
from models.schema import NoteSummary, VisitDetails # Import VisitDetails for schema dependency
from config import MERCURY_API_KEY # Ensure the key is loaded

# --- MOCK CRM DATABASE (Contextual Memory) ---
# Simulates the records your bot searches against for "before and after" recognition.
MOCK_CRM_LEADS = {
    "anjori sarabhai": {"Lead_ID": "L456", "Status": "Active", "Last_Visit_ID": "V101", "Last_Visit_Date": "2025-11-05"},
    "john smith": {"Lead_ID": "L123", "Status": "Active", "Last_Visit_ID": "V202", "Last_Visit_Date": "2025-11-04"},
    "dr. patel": {"Lead_ID": "L789", "Status": "Active", "Last_Visit_ID": "V303", "Last_Visit_Date": "2025-10-29"},
}

def mock_database_lookup(name: str):
    """Simulates searching the CRM database for a lead name (Context Linking)."""
    # Simple, case-insensitive lookup, stripping common titles
    clean_name = name.lower().replace("dr.", "").replace("mr.", "").replace("ms.", "").strip()
    return MOCK_CRM_LEADS.get(clean_name, None)

def process_voice_note_log(transcript: str):
    """
    End-to-end pipeline for voice note logging and contextual look-up.
    This replaces the need for a full scheduling form.
    """
    print(f"\n--- Processing Voice Note ---")
    print(f"Transcript: \"{transcript[:70]}...\"")
    
    # 1. LLM Context Extraction (The memory and summarization step)
    summary_data, llm_latency = extract_context_and_summarize(transcript)
    
    print(f"LLM Summarization Latency: {llm_latency:.4f}s")

    if not summary_data:
        print(" FAILED: Could not extract structured context from the voice note.")
        return

    # 2. Contextual Database Lookup (Linking the note to the Lead)
    extracted_name = summary_data['lead_name']
    lead_record = mock_database_lookup(extracted_name)

    if lead_record:
        print(f"\n SUCCESS: Context Found for '{extracted_name}'.")
        print(f"  - **Linked Lead ID:** {lead_record['Lead_ID']}")
        print(f"  - **Log Target (Before/After):** Last Visit ID {lead_record['Last_Visit_ID']}")
        print(f"  - **Summary:** {summary_data['summary_of_note']}")
        print(f"  - **Action Required:** {summary_data['action_required']}")
    else:
        print(f"\n WARNING: Lead '{extracted_name}' mentioned, but no matching active record found. Note may require manual entry.")

if __name__ == "__main__":
    # Example Test Cases (Simulating ASR output)
    test_notes = [
        "Anjori Sarabhai was very upset with the interest rate update. Reschedule her next appointment and send her a new quote immediately.",
        "The call with John Smith concluded well. He signed the paperwork but needs confirmation via SMS by Friday.",
        "I spoke with Dr. Patel. He needs us to reschedule the visit to next month, he said email is best.",
        "Customer Peter Jones needs us to call him back immediately regarding the loan status." # New lead test
    ]

    for i, note in enumerate(test_notes):
        print("\n" + "="*50)
        print(f"Running Test Note {i+1}")
        print("="*50)
        process_voice_note_log(note)