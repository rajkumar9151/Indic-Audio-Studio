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
import numpy as np
from pydub import AudioSegment
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
    low_cpu_mem_usage=True,
    attn_implementation="eager" # Compatible Eager Mode
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# NEW: AI Memory Cache (Prompt Caching)
encoded_prompts_cache = {}

def get_encoded_prompt(description):
    if description not in encoded_prompts_cache:
        encoded_prompts_cache[description] = tokenizer(description, return_tensors="pt").input_ids.to(device)
    return encoded_prompts_cache[description]

# Store job status for bulk processing
bulk_jobs = {}

class TTSRequest(BaseModel):
    text: str
    description: str
    quality: str = "turbo" 
    format: str = "wav" # Default to WAV

class BulkRequest(BaseModel):
    text: str
    description: str
    job_id: str

def split_text(text):
    # Pre-process text to replace '...' with a pause token for the model
    text = text.replace("...", " [pause] ")
    # Robust regex to split on . ! ? even without a following space. 
    # Also filters out empty strings and trailing whitespace.
    chunks = [s.strip() for s in re.split(r'(?<=[.!?])\s*', text) if s.strip()]
    return chunks

@app.post("/generate")
async def generate_full_audio(request: TTSRequest):
    try:
        # AI Memory Speedup: Reuse the voice prompt
        input_ids = get_encoded_prompt(request.description)
        
        with torch.inference_mode():
            prompt_input_ids = tokenizer(request.text, return_tensors="pt").input_ids.to(device)
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, do_sample=True, temperature=0.8)
            
        audio_data = generation.cpu().numpy().astype("float32").squeeze()
        
        # Hardware Silence fix for (...) dots
        padding_len = 1.0 if "..." in request.text else 0.4
        padding = np.zeros(int(model.config.sampling_rate * padding_len), dtype=np.float32)
        audio_final = np.concatenate([audio_data, padding])

        file_id = f"audio_{uuid.uuid4().hex[:8]}.wav"
        file_path = os.path.join(OUTPUT_DIR, file_id)
        sf.write(file_path, audio_final, model.config.sampling_rate)

        # Convert to MP3 if requested
        if request.format == "mp3":
            audio_seg = AudioSegment.from_wav(file_path)
            mp3_id = file_id.replace(".wav", ".mp3")
            mp3_path = os.path.join(OUTPUT_DIR, mp3_id)
            audio_seg.export(mp3_path, format="mp3", bitrate="192k")
            return {"url": f"/outputs/{mp3_id}", "status": "success"}
        
        return {"url": f"/outputs/{file_id}", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream")
async def stream_audio(request: TTSRequest):
    try:
        # AI Memory Speedup
        input_ids = get_encoded_prompt(request.description)
        
        with torch.inference_mode():
            prompt_input_ids = tokenizer(request.text, return_tensors="pt").input_ids.to(device)
            gen = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, do_sample=True, temperature=0.8)
            
        audio_data = gen.cpu().numpy().astype("float32").squeeze()
        
        # Hardware Silence fix
        padding_len = 0.8 if "..." in request.text else 0.2
        padding = np.zeros(int(model.config.sampling_rate * padding_len), dtype=np.float32)
        audio_final = np.concatenate([audio_data, padding])

        buffer = io.BytesIO()
        sf.write(buffer, audio_final, model.config.sampling_rate, format="WAV")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system-health")
async def get_health():
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        return {
            "device": device,
            "cuda_available": True,
            "memory_free": f"{free_mem / 1024**3:.2f} GB",
            "memory_total": f"{total_mem / 1024**3:.2f} GB"
        }
    return {"device": "cpu", "cuda_available": False, "memory_free": "0 GB"}

@app.post("/bulk-generate")
async def bulk_generate(request: BulkRequest):
    job_id = request.job_id
    bulk_jobs[job_id] = {
        "status": "processing", 
        "progress": 0, 
        "total": 0, 
        "files": [],
        "start_time": time.time(),
        "avg_chunk_time": 0
    }
    
    try:
        sentences = split_text(request.text)
        bulk_jobs[job_id]["total"] = len(sentences)
        
        job_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        for i, sentence in enumerate(sentences):
            start_chunk = time.time()
            if not sentence.strip(): 
                bulk_jobs[job_id]["progress"] += 1
                continue
                
            with torch.inference_mode():
                input_ids = tokenizer(request.description, return_tensors="pt").input_ids.to(device)
                prompt_input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
                gen = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            
            audio_arr = gen.cpu().numpy().astype("float32").squeeze()
            file_name = f"part_{i:03d}.wav"
            file_path = os.path.join(job_dir, file_name)
            sf.write(file_path, audio_arr, model.config.sampling_rate)
            
            bulk_jobs[job_id]["files"].append(f"/outputs/{job_id}/{file_name}")
            bulk_jobs[job_id]["progress"] += 1
            
            # Calculate rolling average for more accurate time estimation
            chunk_duration = time.time() - start_chunk
            if bulk_jobs[job_id]["avg_chunk_time"] == 0:
                bulk_jobs[job_id]["avg_chunk_time"] = chunk_duration
            else:
                bulk_jobs[job_id]["avg_chunk_time"] = (bulk_jobs[job_id]["avg_chunk_time"] * 0.7) + (chunk_duration * 0.3)
            
            # Memory Wash to maintain GPU speed
            if i % 10 == 0:
                torch.cuda.empty_cache()
            
        bulk_jobs[job_id]["status"] = "completed"
        return {"status": "started", "job_id": job_id}
    except Exception as e:
        bulk_jobs[job_id]["status"] = "failed"
        bulk_jobs[job_id]["error"] = str(e)
        return {"status": "error", "message": str(e)}

@app.get("/bulk-status/{job_id}")
async def get_bulk_status(job_id: str):
    return bulk_jobs.get(job_id, {"status": "not_found"})

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
