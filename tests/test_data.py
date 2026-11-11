import torch
from minigpt.data import get_batch, split_encoded_data
from minigpt.tokenizer import CharTokenizer

def test_split_and_batch():
    text="abcdefghij"*20; tok=CharTokenizer(text); train,val=split_encoded_data(text,tok)
    x,y=get_batch(train,4,8,torch.device("cpu"))
    assert x.shape==(4,8) and y.shape==(4,8)
    assert torch.equal(x[:,1:],y[:,:-1]); assert val.numel()>0
