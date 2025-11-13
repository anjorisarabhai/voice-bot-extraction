# models/llm_fallback.py
import time
import requests
import json
import re
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from models.schema import VisitDetails
from config import MERCURY_API_KEY, MERCURY_API_ENDPOINT, OLLAMA_URL, LLAMA3_MODEL

# --- Llama 3 (Ollama) Function (Used for comparison) ---
def extract_via_llama_fallback(transcript: str):
    """Runs the Llama 3 LLM for structured data extraction (Slow Path Proxy)."""
    raw_output_string = ""
    llm_start = time.time()
    
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        template = """SYSTEM INSTRUCTIONS: You are an expert CRM data extractor. Extract information strictly to YYYY-MM-DD and HH:MM format. If any field is missing, set its value to 'N/A'. USER TRANSCRIPT: {transcript} RESPONSE:"""
        prompt = PromptTemplate(template=template, input_variables=["transcript", "current_date"])

        llm = ChatOllama(model=LLAMA3_MODEL, base_url=OLLAMA_URL, format="json", temperature=0)
        chain = prompt | llm
        
        raw_output_message = chain.invoke({"transcript": transcript, "current_date": current_date})
        raw_output_string = raw_output_message.content 
        
        extracted_json = json.loads(raw_output_string)
        result = VisitDetails.model_validate(extracted_json)
        
        llama_latency = time.time() - llm_start 
        return result.model_dump(), llama_latency
    
    except Exception:
        llama_latency = time.time() - llm_start
        return None, llama_latency


# --- Mercury (dLLM) Function (Used for production Fallback and Comparison) ---
def extract_via_mercury_fallback(transcript: str):
    """
    Runs the Mercury dLLM API using the Tool Calling method (Final Production Fallback).
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
        response = requests.post(MERCURY_API_ENDPOINT, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        raw_output = response.json()
        
        # 1. Get the raw arguments string
        tool_call_args_str = raw_output['choices'][0]['message']['tool_calls'][0]['function']['arguments']
        
        # 2. ULTIMATE DEFENSE STEP: Target the specific malformed JSON syntax
        # This fixes the recurring error caused by the API inserting extra spaces/quotes.
        cleaned_args_str = tool_call_args_str.replace(':" "', ':"') 
        cleaned_args_str = re.sub(r',\s*', ',', cleaned_args_str)
        cleaned_args_str = re.sub(r'\s*:\s*', ':', cleaned_args_str)
        
        # 3. Final Parsing
        extracted_json = json.loads(cleaned_args_str)
        result = VisitDetails.model_validate(extracted_json)
        
        latency = time.time() - llm_start
        return result.model_dump(), latency
    
    except Exception:
        return None, (time.time() - llm_start)