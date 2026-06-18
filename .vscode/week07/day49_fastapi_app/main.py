from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import json
import os
model_path = 'bbc_distilbert_final'

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

with open(os.path.join(model_path, 'label_map.json')) as f:
    maps = json.load(f)
id2label = {int(k): v for k, v in maps['id2label'].items()}

app = FastAPI(title="BBC News Classifier")

class NewsRequest(BaseModel):
    text: str

class NewsResponse(BaseModel):
    category: str
    confidence: float

@app.get("/")
def root():
    return {"message": "BBC news classifier api, see /docs"}

@app.post("/predict", response_model=NewsResponse)
def predict(req: NewsRequest):
    inputs = tokenizer(req.text, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred_id = probs.argmax(dim=1).item()
    confidence = probs[0][pred_id].item()
    return NewsResponse(category=id2label[pred_id], confidence=confidence)