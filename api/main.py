# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # <-- NEW IMPORT

from api.routers import utility

app = FastAPI(
    title="Voice Bot Data Extraction API",
    description="Backend service for transcribing audio and extracting structured data for CRM.",
    version="1.0.0"
)

# --- CRITICAL FIX: ADD CORS MIDDLEWARE ---
origins = [
    "http://127.0.0.1:8000",
    "http://localhost",
    "http://localhost:8080",
    # CRITICAL: Allow file:// access for local testing (though not technically a 'domain', it's safe for MVP demo)
    "null", # Represents file:// access in some browsers
    "*", # Wildcard is the safest way to ensure local file access works for the demo
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------------------

# Include all routers
app.include_router(utility.router, prefix="/api/utilities", tags=["Utilities"])
app.include_router(utility.router, prefix="/api/search", tags=["Search"])


@app.get("/")
async def root():
    return {"message": "Voice Bot Extraction Service is running. See /docs for API documentation."}