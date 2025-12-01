# Voice Bot Data Extraction Microservice

This repository hosts the FastAPI backend for a voice-activated CRM system. The core function is to execute a Hybrid Extraction Pipeline that converts unstructured speech transcripts into highly structured, validated data for scheduling and contact logging. The architecture is optimized for minimal latency using dLLM technology for complex inputs.

## I. Architectural Overview

The system operates as a set of decoupled services, ensuring high performance (non-blocking I/O) and maintainability.

### A. Core Hybrid Strategy

The project's success relies on strategically separating inputs into two paths based on complexity:

- **Fast Path (Local NLP):** Handles simple structural extraction tasks (Name, Type). Because it avoids all network calls and large models, its latency is near-instantaneous (≈ 0.000s).
- **Slow Path (Mercury dLLM):** Handles complex, calculation-heavy tasks (Date arithmetic, detailed structure). This is routed to the external, high-performance Mercury dLLM API for guaranteed accuracy, even at higher latency (≈ 1.5s).

### B. Microservice Structure

| Service Prefix     | Primary Function          | Key Endpoints                       | Performance Goal                  |
|--------------------|---------------------------|-------------------------------------|-----------------------------------|
| `/api/utilities`   | Extraction Core (I/O Bound) | `/extract-data`                    | High Latency Tolerance (API wait) |
| `/api/search`      | Real-Time Validation      | `/search/attendees`, `/validate_name` | Ultra-Low Latency (Sub-100ms)     |

## II. Specialized Logic and Accuracy

### A. Asynchronous Data Flow

The entire backend is built using FastAPI and the non-blocking HTTP client httpx. This is critical because the ≈ 1.5s Mercury API call runs asynchronously, allowing the server to handle thousands of other client requests simultaneously without freezing.

### B. Accuracy Layers

- **Name Normalization (Indic Transliteration):**  
  The `models/demo_utils.py` logic runs the indic-transliteration library after ASR. This step standardizes the Romanized spelling of Indian names (e.g., corrects "Onjori" to "Anjori"), ensuring high accuracy for database lookups.

- **Schema Enforcement:**  
  The `models/llm_fallback.py` uses Pydantic's schema directly within the Mercury Tool Calling feature. This forces the LLM to output clean, validated JSON (`VisitDetails`), eliminating parsing errors on the backend.

## III. Repository Map
```
voice-bot-extraction/
├── api/ # FastAPI Backend and Routers
│ ├── main.py # Uvicorn Entry Point (Starts the server)
│ └── routers/utility.py # Defines the Hybrid Extraction and Search Endpoints
├── config/ # Environment and Endpoint Management
├── models/ # Core Logic Modules
│ ├── llm_fallback.py # Mercury dLLM Tool Calling Logic
│ └── nlp_core.py # Fast NLP/Regex Logic (Sub-millisecond latency)
├── main.py # Final Benchmark Runner (Verifies performance)
└── requirements.txt # Project Dependencies
```
## IV. Setup and Verification

### 1. Installation

Clone and activate venv (Virtual Environment).  
Install dependencies:
```
(venv) $ pip install -r requirements.txt
```

### 2. Execution

**Start the FastAPI Server:** This launches the API for external access.
```
(venv) $ uvicorn api.main:app --reload
```

**Run the Benchmark:** Execute the main script to validate the speed and accuracy split across 15 cases. This confirms the successful implementation of the hybrid strategy.
```
(venv) $ python main.py
```
