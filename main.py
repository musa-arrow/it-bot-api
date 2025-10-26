from fastapi import FastAPI
from pydantic import BaseModel  
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import uvicorn
import os
import sys
print("--- Python Sürümü:", sys.version)
print("--- Çalışma Dizini:", os.getcwd())


MODEL_DIR = "./it_ticket_classifier_model" 
print(f"--- Model Klasörü Aranıyor: {MODEL_DIR}")
print(f"--- Klasör Var mı?: {os.path.exists(MODEL_DIR)}")
if os.path.exists(MODEL_DIR):
    print(f"--- Klasör İçeriği: {os.listdir(MODEL_DIR)}")
    model_file_path = os.path.join(MODEL_DIR, 'model.safetensors')
    print(f"--- model.safetensors Var mı?: {os.path.exists(model_file_path)}")

try:
    print("--- Model yüklenmeye BAŞLIYOR... (RAM artışı beklenebilir)")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

    model.eval()
    print("--- Model ve Tokenizer başarıyla yüklendi. API hazır. ---")
except Exception as e:
    print(f"HATA: Model yüklenemedi. '{MODEL_DIR}' klasörünün doğru yerde olduğundan emin misin?")
    import traceback
    print("--- HATA DETAYI (Full Traceback): ---")
    traceback.print_exc() 
    print("--- HATA BİTTİ ---")
    model = None
    tokenizer = None

kategoriler = [
    'Access',
    'Administrative rights',
    'HR Support',
    'Hardware',
    'Internal Project',
    'Miscellaneous',
    'Purchase',
    'Storage'
]

app = FastAPI(
    title="IT Ticket Classifier API",
    description="Copilot Studio için özel eğitilmiş NLP modelini sunan API."
)


class TicketInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    kategori: str
    kategori_id: int
    skor: float 

@app.get("/")
def read_root():
    return {"message": "IT Ticket Classifier API'si çalışıyor. Analiz için /analyze endpoint'ine POST isteği atın."}

@app.post("/analyze", response_model=PredictionOutput)
async def analyze_ticket(ticket: TicketInput):
    """
    Gelen metni analiz eder ve kategorisini tahmin eder.
    """
    if not model or not tokenizer:
        return {"kategori": "HATA", "kategori_id": -1, "skor": 0.0}
        
    inputs = tokenizer(
        ticket.text,
        return_tensors="pt", 
        truncation=True,
        padding=True,
        max_length=128 
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=1)
    
    skor, predicted_class_id = torch.max(probabilities, dim=1)
    
    kategori_id = predicted_class_id.item()
    kategori_skoru = skor.item()

    kategori_adi = "Bilinmiyor"
    if kategori_id < len(kategoriler):
         kategori_adi = kategoriler[kategori_id]

    return {
        "kategori": kategori_adi,
        "kategori_id": kategori_id,
        "skor": kategori_skoru
    }

# Bu, dosyayı doğrudan 'python main.py' ile de çalıştırabilmeni sağlar (opsiyonel)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
