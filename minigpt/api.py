from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from minigpt.generate import load_model

app=FastAPI(title="MiniGPT API",version="0.1.0")
model=tokenizer=device=None

class GenerateRequest(BaseModel):
    prompt: str = Field(default="",max_length=1000)
    max_new_tokens: int = Field(default=100,ge=1,le=500)
    temperature: float = Field(default=0.8,gt=0.0,le=2.0)

@app.on_event("startup")
def startup():
    global model,tokenizer,device
    if Path("checkpoints/latest.pt").exists() and Path("artifacts/vocab.json").exists():
        model,tokenizer,device=load_model()

@app.get("/health")
def health():
    return {"status":"ok","model_loaded":model is not None,"device":str(device) if device else None}

@app.post("/generate")
def generate(req: GenerateRequest):
    if model is None: raise HTTPException(status_code=503,detail="Train MiniGPT before using generation")
    try:
        ids=tokenizer.encode(req.prompt) if req.prompt else [0]
        tokens=torch.tensor([ids],dtype=torch.long,device=device)
        out=model.generate(tokens,req.max_new_tokens,req.temperature)
        return {"text":tokenizer.decode(out[0].tolist())}
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
