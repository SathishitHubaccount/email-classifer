from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model():
    model_path = "sathish2352/email-classifier-model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def classify_email(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", max_length=256, padding="max_length", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    label_map = {0: "Incident", 1: "Request", 2: "Change", 3: "Problem"}
    pred = torch.argmax(logits, dim=1).item()
    return label_map[pred]
