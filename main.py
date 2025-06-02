# Entry point for FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
from models import load_model, classify_email
from utils import mask_pii_multilingual

app = FastAPI()
tokenizer, model, device = load_model()

class EmailInput(BaseModel):
    input_email_body: str

@app.post("/classify")
async def classify_route(request: EmailInput):
    text = request.input_email_body
    masked_text, entities = mask_pii_multilingual(text)
    category = classify_email(masked_text, tokenizer, model, device)
    return {
        "input_email_body": text,
        "list_of_masked_entities": entities,
        "masked_email": masked_text,
        "category_of_the_email": category
    }
    