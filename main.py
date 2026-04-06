import os
import torch
import uvicorn
import uuid
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# Initialize FastAPI
app = FastAPI(title="Indic Audio Studio API")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "ai4bharat/indic-parler-tts"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Model
print(f"--- Loading Indic Parler-TTS on {device} ---")
# To save VRAM, you can use torch_dtype=torch.float16 if device == "cuda:0"
try:
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("--- Model Loaded Successfully ---")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

class TTSRequest(BaseModel):
    text: str
    description: str = "A clear female voice, moderate pace, high quality audio."

@app.post("/generate")
async def generate_audio(request: TTSRequest):
    try:
        # Tokenize inputs
        input_ids = tokenizer(request.description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(request.text, return_tensors="pt").input_ids.to(device)

        # Generate audio
        with torch.inference_mode():
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        
        audio_arr = generation.cpu().numpy().squeeze()
        
        # Save file with unique ID
        file_id = f"audio_{uuid.uuid4().hex[:8]}.wav"
        file_path = os.path.join(OUTPUT_DIR, file_id)
        sf.write(file_path, audio_arr, model.config.sampling_rate)
        
        return {"url": f"/outputs/{file_id}", "status": "success"}
    
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (Frontend and Generated Audio)
if os.path.exists("outputs"):
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    # Start server on 0.0.0.0 so it's accessible externally
    uvicorn.run(app, host="0.0.0.0", port=8000)
