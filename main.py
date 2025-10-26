from fastapi import FastAPI
from pydantic import BaseModel  # Girdi/Çıktı veri tiplerini tanımlamak için
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import uvicorn
import os
import sys
print("--- Python Sürümü:", sys.version)
print("--- Çalışma Dizini:", os.getcwd())

# --- 1. Model Yükleme (Uygulama Başlarken Sadece 1 Kez) ---
# Bu, API'mizin hızlı çalışmasını sağlar. Her istekte modeli yüklemeyiz.

MODEL_DIR = "./it_ticket_classifier_model" # Model klasörünün yolu
print(f"--- Model Klasörü Aranıyor: {MODEL_DIR}")
print(f"--- Klasör Var mı?: {os.path.exists(MODEL_DIR)}")
if os.path.exists(MODEL_DIR):
    print(f"--- Klasör İçeriği: {os.listdir(MODEL_DIR)}")
    model_file_path = os.path.join(MODEL_DIR, 'model.safetensors')
    print(f"--- model.safetensors Var mı?: {os.path.exists(model_file_path)}")

try:
    # Modeli PyTorch ile yüklüyoruz (çünkü böyle eğittik)
    print("--- Model yüklenmeye BAŞLIYOR... (RAM artışı beklenebilir)")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

    # Modeli "değerlendirme" moduna alıyoruz (eğitim yapmayacak)
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

# --- Kategori Eşleştirme (ÇOK ÖNEMLİ) ---
# Model bize '0', '1', '2' gibi ID'ler verecek.
# Bu ID'leri metinlere çevirmemiz lazım.
# LabelEncoder kategorileri alfabetik olarak sıralar.
# Kaggle veri setindeki (Arama Sonucu 1.4) kategorilerin alfabetik sırası:
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

# FastAPI uygulamasını başlat
app = FastAPI(
    title="IT Ticket Classifier API",
    description="Copilot Studio için özel eğitilmiş NLP modelini sunan API."
)

# --- 2. Girdi ve Çıktı Modelleri (Veri Tipi Kontrolü) ---

# API'nin DIŞARIDAN ne alacağını tanımlar
class TicketInput(BaseModel):
    text: str

# API'nin DIŞARIYA ne vereceğini tanımlar
class PredictionOutput(BaseModel):
    kategori: str
    kategori_id: int
    skor: float  # Modelin bu tahmininden ne kadar emin olduğu

# --- 3. API Endpoint (Copilot'un konuşacağı adres) ---

@app.get("/")
def read_root():
    return {"message": "IT Ticket Classifier API'si çalışıyor. Analiz için /analyze endpoint'ine POST isteği atın."}

@app.post("/analyze", response_model=PredictionOutput)
async def analyze_ticket(ticket: TicketInput):
    """
    Gelen metni analiz eder ve kategorisini tahmin eder.
    """
    if not model or not tokenizer:
        # Model yüklenmemişse hata döndür
        return {"kategori": "HATA", "kategori_id": -1, "skor": 0.0}

    # 1. Gelen metni tokenize et (PyTorch tensörleri olarak)
    inputs = tokenizer(
        ticket.text,
        return_tensors="pt",  # PyTorch tensörü olarak döndür
        truncation=True,
        padding=True,
        max_length=128 # Eğitimdekiyle aynı
    )

    # 2. Modeli çalıştır (torch.no_grad() ile gereksiz hesaplamaları kapatırız)
    with torch.no_grad():
        logits = model(**inputs).logits

    # 3. Sonuçları işle
    # Logits -> Softmax (olasılıklara dönüştür)
    probabilities = torch.softmax(logits, dim=1)

    # En yüksek skoru ve ID'sini bul
    skor, predicted_class_id = torch.max(probabilities, dim=1)

    # Tensor'dan Python sayısına çevir
    kategori_id = predicted_class_id.item()
    kategori_skoru = skor.item()

    # ID'yi metin etikete çevir (listemizden)
    kategori_adi = "Bilinmiyor"
    if kategori_id < len(kategoriler):
         kategori_adi = kategoriler[kategori_id]

    # 4. JSON olarak döndür
    return {
        "kategori": kategori_adi,
        "kategori_id": kategori_id,
        "skor": kategori_skoru
    }

# Bu, dosyayı doğrudan 'python main.py' ile de çalıştırabilmeni sağlar (opsiyonel)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
