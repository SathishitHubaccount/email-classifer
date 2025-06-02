import re
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
def mask_pii_multilingual(text: str):

    # Load model only once globally if needed
    model_name = "Davlan/xlm-roberta-base-ner-hrl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    regex_patterns = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone_number": r"(?:\+?\d{1,3})?[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}",
        "dob": r"\b(0?[1-9]|[12][0-9]|3[01])[-/](0?[1-9]|1[012])[-/](19[5-9]\d|20[0-3]\d)\b",
        "aadhar_num": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "credit_debit_no": r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
        "cvv_no": r"\b\d{3,4}\b",
        "expiry_no": r"\b(0[1-9]|1[0-2])[/-]?(?:\d{2}|\d{4})\b"
    }

    entities = []
    masked_text = text
    offsets = []

    # Step 1: Apply regex PII masking first
    for entity_type, pattern in regex_patterns.items():
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            if any(start < e[1] and end > e[0] for e in offsets):
                continue
            token = f"[{entity_type}]"
            entity_val = text[start:end]
            masked_text = masked_text[:start] + token + masked_text[end:]
            offsets.append((start, end))
            entities.append({
                "position": [start, end],
                "classification": entity_type,
                "entity": entity_val
            })

    # Step 2: Run NER on updated masked_text to avoid overlap
    ner_results = ner_pipe(masked_text)
    for ent in ner_results:
        start, end = ent["start"], ent["end"]
        if ent["entity_group"] != "PER":
            continue
        if any(start < e[1] and end > e[0] for e in offsets):
            continue
        token = "[full_name]"
        entity_val = text[start:end]
        masked_text = masked_text[:start] + token + masked_text[end:]
        entities.append({
            "position": [start, end],
            "classification": "full_name",
            "entity": entity_val
        })
        offsets.append((start, end))

    # Sort final result
    entities.sort(key=lambda x: x["position"][0])
    return masked_text, entities
