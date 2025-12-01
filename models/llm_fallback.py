# models/llm_fallback.py
import time
import json
import re
from datetime import datetime
from pydantic import ValidationError
import httpx # Required for asynchronous HTTP calls
from models.schema import VisitDetails
from config import MERCURY_API_KEY, MERCURY_API_ENDPOINT


# --- Mercury (dLLM) Function (The Production Fallback) ---
async def extract_via_mercury_fallback(transcript: str):
    """
    Runs the Mercury dLLM API using the Tool Calling method for structured output.
    This function is async, essential for FastAPI performance.
    """
    llm_start = time.time()
    current_date = datetime.now().strftime("%Y-%m-%d")

    # --- Tool Definition and Payload Setup ---
    tool_definition = {
        "type": "function",
        "function": {
            "name": "schedule_visit",
            "description": "Extracts structured data for scheduling a CRM visit.",
            "parameters": VisitDetails.model_json_schema()
        }
    }
    
    system_message = (
        f"You are an expert CRM data extractor. Your task is to extract information from the user's transcript "
        f"and call the 'schedule_visit' tool with the extracted data. Current Date: {current_date}. "
        f"Strictly adhere to the provided JSON schema."
    )
    
    payload = {
        "model": "mercury", 
        "messages": [
            {"role": "system", "content": system_message}, 
            {"role": "user", "content": transcript}
        ],
        "tools": [tool_definition],
        "tool_choice": {"type": "function", "function": {"name": "schedule_visit"}},
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MERCURY_API_KEY}"
    }

    try:
        # CRITICAL FIX: Use httpx.AsyncClient and await the request
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                MERCURY_API_ENDPOINT, headers=headers, json=payload
            )
        response.raise_for_status() # Raise error for HTTP error codes

        raw_output = response.json()
        
        # 1. Get the raw arguments string
        tool_call_args_str = raw_output['choices'][0]['message']['tool_calls'][0]['function']['arguments']
        
        # 2. ULTIMATE DEFENSE STEP: Target the specific malformed JSON syntax (from previous debugging)
        # This cleans up the persistent internal spacing/quote error from the API.
        cleaned_args_str = tool_call_args_str.replace(':" "', ':"') 
        cleaned_args_str = re.sub(r',\s*', ',', cleaned_args_str)
        cleaned_args_str = re.sub(r'\s*:\s*', ':', cleaned_args_str)
        
        # 3. Final Parsing and Validation
        extracted_json = json.loads(cleaned_args_str)
        result = VisitDetails.model_validate(extracted_json)
        
        latency = time.time() - llm_start
        return result.model_dump(), latency
    
    except Exception as e:
        # Catch all exceptions (network, JSON parsing, validation)
        raise e # Re-raise the exception so the router can handle it explicitly