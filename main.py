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

# Store job status for bulk processing
bulk_jobs = {}

class TTSRequest(BaseModel):
    text: str
    description: str
    quality: str = "turbo" 

class BulkRequest(BaseModel):
    text: str
    description: str
    job_id: str

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

@app.get("/system-health")
async def get_health():
    return {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB" if torch.cuda.is_available() else "0 GB"
    }

@app.post("/bulk-generate")
async def bulk_generate(request: BulkRequest):
    job_id = request.job_id
    bulk_jobs[job_id] = {"status": "processing", "progress": 0, "total": 0, "files": []}
    
    try:
        sentences = split_text(request.text)
        bulk_jobs[job_id]["total"] = len(sentences)
        
        job_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        for i, sentence in enumerate(sentences):
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
