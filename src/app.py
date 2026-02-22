from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.train import predict_contacts, get_latest_model
from src.preprocessing import load_data

app = FastAPI(title="MLOps Contact Tracing API")

# Request Schema
class ContactRequest(BaseModel):
    user_name: str

# Response Schema
class ContactResponse(BaseModel):
    user_of_interest: str
    potential_contacts: List[str]

# Load model once at startup for efficiency
model = get_latest_model()

@app.post("/trace", response_model=ContactResponse)
async def trace(request: ContactRequest):
    df = load_data()
    if df is None:
        raise HTTPException(status_code=500, detail="Data load failed")

    try:
        contacts = predict_contacts(df, request.user_name, model)
        return {
            "user_of_interest": request.user_name,
            "potential_contacts": contacts
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
