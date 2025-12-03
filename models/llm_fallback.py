# models/llm_fallback.py
import time
import requests
import json
import re
from datetime import datetime
from pydantic import ValidationError
import httpx # Required for asynchronous HTTP calls

# Import core schema and settings
from models.schema import VisitDetails
from config import MERCURY_API_KEY, MERCURY_API_ENDPOINT


# --- Mercury (dLLM) Function (The Production Fallback) ---
async def extract_via_mercury_fallback(transcript: str):
    """
    Runs the Mercury dLLM API using the Tool Calling method for structured output.
    Implements adaptive parsing and mapping to ensure schema validation succeeds.
    """
    llm_start = time.time()
    current_date = datetime.now().strftime("%Y-%m-%d")

    tool_definition = {
        "type": "function",
        "function": {
            "name": "schedule_visit",
            "description": "Extracts structured data for scheduling a CRM visit.",
            "parameters": VisitDetails.model_json_schema()
        }
    }
    
    # --- CRITICAL PROMPT CHANGE: Instruct LLM to ignore removed fields ---
    system_message = (
        f"You are an expert CRM data extractor. Your task is to extract the Event Name, Visit Type, and Lead Name "
        f"from the user's transcript and output a JSON object that strictly conforms to the provided JSON schema. "
        f"IMPORTANT: Ignore all date, time, and scheduling information, as those fields are handled separately."
    )
    # --- END CRITICAL PROMPT CHANGE ---
    
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
        # 1. Execute the async request
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                MERCURY_API_ENDPOINT, headers=headers, json=payload
            )
        response.raise_for_status() 
        raw_output = response.json()
        
        # 2. Adaptive Parsing: Check for tool_call or plain content
        json_string_to_parse = None
        message = raw_output['choices'][0]['message']
        tool_calls = message.get('tool_calls')

        if tool_calls and len(tool_calls) > 0:
            json_string_to_parse = tool_calls[0]['function']['arguments']
        else:
            content_string = message.get('content', '')
            if '```json' in content_string:
                json_string_to_parse = content_string.split('```json')[1].split('```')[0].strip()
            else:
                json_string_to_parse = content_string.strip()

        if not json_string_to_parse:
            raise ValueError("Model failed to provide parsable JSON.")

        # 3. Clean and Load Raw JSON
        cleaned_args_str = json_string_to_parse.replace(':" "', ':"') 
        cleaned_args_str = re.sub(r',\s*', ',', cleaned_args_str)
        cleaned_args_str = re.sub(r'\s*:\s*', ':', cleaned_args_str)
        
        extracted_json = json.loads(cleaned_args_str)

        # --- CRITICAL FIX: UNIFIED MAPPING AND DEFAULTING ---
        
        # Define a clean dictionary that maps all possible LLM keys to the schema's required keys
        final_mapped_data = {
            # Map LLM's 'title', 'event_name' to schema's 'title'
            "title": extracted_json.get('title', extracted_json.get('event_name', extracted_json.get('Event Name', 'N/A'))),
            
            # Map LLM's 'visit_type', 'Visit Type' to schema's 'visit_type'
            "visit_type": extracted_json.get('visit_type', extracted_json.get('Visit Type', 'N/A')), 
            
            # Map LLM's 'lead_name', 'Lead Name' to schema's 'lead_name'
            "lead_name": extracted_json.get('lead_name', extracted_json.get('Lead Name', 'N/A')),
            
            # Ensure new mandatory fields are present (defaulting to N/A)
            "email": extracted_json.get('email', 'N/A'),
            "phone_number": extracted_json.get('phone_number', 'N/A'),
        }

        # 4. Final Validation against the strict schema
        result = VisitDetails.model_validate(final_mapped_data)
        
        latency = time.time() - llm_start
        return result.model_dump(), latency
    
    except Exception as e:
        # Re-raise the exception for the router to handle explicitly
        raise e