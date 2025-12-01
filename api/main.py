# api/main.py
from fastapi import FastAPI
from api.routers import utility

app = FastAPI(
    title="Voice Bot Extraction Backend",
    description="High-performance FastAPI service for Hybrid NLP/LLM data extraction."
)

# Include the utility router which handles the voice extraction logic
app.include_router(utility.router, prefix="/api/utilities", tags=["Utilities"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Voice Bot Extraction Service is running."}

# To run the server locally, use the command:
# uvicorn api.main:app --reload