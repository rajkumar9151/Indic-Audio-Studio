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

app = FastAPI(title="Indic Audio Studio Ultimate")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "ai4bharat/indic-parler-tts"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"--- Loading Indic Parler-TTS Ultimate on {device} ---")
model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    low_cpu_mem_usage=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

class TTSRequest(BaseModel):
    text: str
    description: str
    quality: str = "turbo" 

def split_text(text):
    return re.split('(?<=[.!?]) +', text)

@app.post("/generate")
async def generate_full_audio(request: TTSRequest):
    try:
        with torch.inference_mode():
            input_ids = tokenizer(request.description, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = tokenizer(request.text, return_tensors="pt").input_ids.to(device)
            
            if request.quality == "4k":
                model.to(torch.float32)
            
            generation = model.generate(
                input_ids=input_ids, 
                prompt_input_ids=prompt_input_ids,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
            
            if request.quality == "4k":
                model.to(torch.float16)

        # FIX: Ensure we cast to float32 before saving to a WAV file
        audio_arr = generation.cpu().numpy().astype("float32").squeeze()
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
                gen = model.generate(
                    input_ids=input_ids, 
                    prompt_input_ids=prompt_input_ids,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95
                )
            
            # FIX: Ensure we cast to float32 for WAV stream
            audio_data = gen.cpu().numpy().astype("float32").squeeze()
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, model.config.sampling_rate, format="WAV")
            yield buffer.getvalue()

    return StreamingResponse(audio_generator(), media_type="audio/wav")

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
