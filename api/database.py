# api/database.py
import logging

# --- MOCK DATABASE ---
# Kept for handover so the team can verify the pipeline immediately.
MOCK_ATTENDEES = [
    {"id": "L001", "name": "Anjori Sarabhai", "email": "anjori@corp.com", "phone": "9998887776"},
    {"id": "L002", "name": "Rajiv Sharma", "email": "rajiv@corp.com", "phone": "9876543210"},
    {"id": "L003", "name": "Akash Gupta", "email": "akash@corp.com", "phone": "9112233445"},
    {"id": "L004", "name": "Rahul Verma", "email": "rahul@corp.com", "phone": "9223344556"},
    {"id": "L005", "name": "Amit Shah", "email": "amit@corp.com", "phone": "9334455667"},
]

async def get_lead_by_name(name: str):
    """
    HOOK FOR CLIENT TEAM:
    Replace the loop below with your SQL/ORM query.
    Example: return await db.fetch_one("SELECT * FROM leads WHERE name = :name", {"name": name})
    """
    logging.info(f"DB Search: {name}")
    
    if not name or name == "N/A":
        return None

    # Search logic for the Mock DB
    for att in MOCK_ATTENDEES:
        if att["name"].lower() == name.lower():
            return att
            
    return None