import pytest
import torch
from minigpt.model import GPTConfig, MiniGPT

def test_forward_and_loss():
    m=MiniGPT(GPTConfig(20,16,32,2,4,0.0)); x=torch.randint(0,20,(2,12))
    logits,loss=m(x,x); assert logits.shape==(2,12,20); assert torch.isfinite(loss)

def test_attention_is_causal():
    torch.manual_seed(7); m=MiniGPT(GPTConfig(20,16,32,2,4,0.0)).eval()
    a=torch.tensor([[1,2,3,4]]); b=torch.tensor([[1,2,9,8]])
    with torch.no_grad(): la,_=m(a); lb,_=m(b)
    assert torch.allclose(la[:,:2],lb[:,:2],atol=1e-6)

def test_context_limit():
    m=MiniGPT(GPTConfig(10,4,16,1,2))
    with pytest.raises(ValueError): m(torch.ones((1,5),dtype=torch.long))

def test_generate_length():
    m=MiniGPT(GPTConfig(10,4,16,1,2)); out=m.generate(torch.tensor([[1,2]]),3)
    assert out.shape==(1,5)
