import argparse
import torch
from minigpt.model import GPTConfig, MiniGPT
from minigpt.tokenizer import CharTokenizer
from minigpt.train import choose_device

def load_model(checkpoint_path="checkpoints/latest.pt",vocab_path="artifacts/vocab.json",device=None):
    device = device or choose_device()
    checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=True)
    cfg = GPTConfig(**checkpoint["config"]); tokenizer = CharTokenizer.load(vocab_path)
    if tokenizer.vocab_size != cfg.vocab_size: raise ValueError("Tokenizer and checkpoint vocabulary mismatch")
    model = MiniGPT(cfg).to(device); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    return model,tokenizer,device

def generate_text(prompt="",max_new_tokens=200,temperature=0.8,checkpoint_path="checkpoints/latest.pt",vocab_path="artifacts/vocab.json"):
    model,tokenizer,device = load_model(checkpoint_path,vocab_path)
    ids = tokenizer.encode(prompt) if prompt else [0]
    out = model.generate(torch.tensor([ids],dtype=torch.long,device=device),max_new_tokens,temperature)
    return tokenizer.decode(out[0].tolist())

def main():
    p=argparse.ArgumentParser(); p.add_argument("--prompt",default=""); p.add_argument("--max-new-tokens",type=int,default=200)
    p.add_argument("--temperature",type=float,default=0.8); p.add_argument("--checkpoint",default="checkpoints/latest.pt"); p.add_argument("--vocab",default="artifacts/vocab.json")
    a=p.parse_args(); print(generate_text(a.prompt,a.max_new_tokens,a.temperature,a.checkpoint,a.vocab))

if __name__ == "__main__": main()
