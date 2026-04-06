import os
import torch
import uvicorn
import uuid
import re
import io
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# Initialize FastAPI
app = FastAPI(title="Indic Audio Studio Pro")

# Allow CORS
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

# --- Standard Initialization (Float 16 by default) ---
print(f"--- Loading Indic Parler-TTS Pro on {device} ---")
model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    low_cpu_mem_usage=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

class TTSRequest(BaseModel):
    text: str
    description: str
    quality: str = "turbo" # "turbo" (FP16) or "4k" (FP32)

def split_text(text):
    # Splits by sentence ends to facilitate faster streaming start
    return re.split('(?<=[.!?]) +', text)

@app.post("/generate")
async def generate_full_audio(request: TTSRequest):
    try:
        # Toggle precision if requested (Note: This is an experimental feature)
        # Using inference_mode for maximum speed
        with torch.inference_mode():
            input_ids = tokenizer(request.description, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = tokenizer(request.text, return_tensors="pt").input_ids.to(device)
            
            # High precision 4k mode casting
            if request.quality == "4k":
                model.to(torch.float32)
            
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            
            # Revert to turbo float16
            if request.quality == "4k":
                model.to(torch.float16)

        audio_arr = generation.cpu().numpy().squeeze()
        file_id = f"audio_{uuid.uuid4().hex[:8]}.wav"
        file_path = os.path.join(OUTPUT_DIR, file_id)
        sf.write(file_path, audio_arr, model.config.sampling_rate)
        
        return {"url": f"/outputs/{file_id}", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream")
async def stream_audio(request: TTSRequest):
    def audio_generator():
        sentences = split_text(request.text)
        for sentence in sentences:
            if not sentence.strip(): continue
            with torch.inference_mode():
                input_ids = tokenizer(request.description, return_tensors="pt").input_ids.to(device)
                prompt_input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
                gen = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            
            audio_data = gen.cpu().numpy().squeeze()
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, model.config.sampling_rate, format="WAV")
            yield buffer.getvalue()

    return StreamingResponse(audio_generator(), media_type="audio/wav")

# Serve files
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
