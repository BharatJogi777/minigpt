import argparse
import random
from pathlib import Path
import torch
from minigpt.data import get_batch, load_corpus, split_encoded_data
from minigpt.model import GPTConfig, MiniGPT
from minigpt.tokenizer import CharTokenizer

def choose_device(requested="auto"):
    if requested == "cpu": return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available(): raise RuntimeError("MPS is unavailable")
        return torch.device("mps")
    if requested == "auto": return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    raise ValueError(f"Unknown device: {requested}")

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters):
    model.eval(); result = {}
    for name, data in (("train", train_data), ("val", val_data)):
        values = []
        for _ in range(eval_iters):
            x,y = get_batch(data,batch_size,block_size,device)
            _,loss = model(x,y); values.append(loss.item())
        result[name] = sum(values)/len(values)
    model.train(); return result

def train(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = choose_device(args.device)
    text = load_corpus(args.data); tokenizer = CharTokenizer(text)
    train_data,val_data = split_encoded_data(text,tokenizer)
    if min(train_data.numel(),val_data.numel()) <= args.block_size:
        raise ValueError("Train and validation splits must each exceed block_size; add data or reduce context")
    cfg = GPTConfig(tokenizer.vocab_size,args.block_size,args.n_embd,args.n_layer,args.n_head,args.dropout)
    model = MiniGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    print(f"device={device} vocab={cfg.vocab_size} parameters={sum(p.numel() for p in model.parameters()):,}")
    for step in range(args.steps):
        x,y = get_batch(train_data,args.batch_size,args.block_size,device)
        _,loss = model(x,y); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip); optimizer.step()
        if step % args.log_every == 0 or step == args.steps-1:
            m = estimate_loss(model,train_data,val_data,args.batch_size,args.block_size,device,args.eval_iters)
            print(f"step {step:5d} | train {m['train']:.4f} | val {m['val']:.4f}")
    Path(args.artifacts_dir).mkdir(parents=True,exist_ok=True); Path(args.checkpoint_dir).mkdir(parents=True,exist_ok=True)
    tokenizer.save(Path(args.artifacts_dir)/"vocab.json")
    path = Path(args.checkpoint_dir)/"latest.pt"
    torch.save({"model_state_dict":model.state_dict(),"config":cfg.to_dict(),"step":args.steps},path)
    print(f"Saved checkpoint: {path}"); return path

def build_parser():
    p=argparse.ArgumentParser(description="Train MiniGPT from random initialization")
    p.add_argument("--data",default="data/input.txt"); p.add_argument("--steps",type=int,default=2000)
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--block-size",type=int,default=128)
    p.add_argument("--n-embd",type=int,default=128); p.add_argument("--n-layer",type=int,default=4)
    p.add_argument("--n-head",type=int,default=4); p.add_argument("--dropout",type=float,default=0.1)
    p.add_argument("--learning-rate",type=float,default=3e-4); p.add_argument("--grad-clip",type=float,default=1.0)
    p.add_argument("--log-every",type=int,default=100); p.add_argument("--eval-iters",type=int,default=10)
    p.add_argument("--seed",type=int,default=1337); p.add_argument("--device",choices=["auto","cpu","mps"],default="auto")
    p.add_argument("--artifacts-dir",default="artifacts"); p.add_argument("--checkpoint-dir",default="checkpoints")
    return p

if __name__ == "__main__": train(build_parser().parse_args())
