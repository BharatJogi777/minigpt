import pytest
from minigpt.tokenizer import CharTokenizer

def test_round_trip():
    t=CharTokenizer("hello world"); assert t.decode(t.encode("hello")) == "hello"

def test_unknown_character():
    t=CharTokenizer("abc")
    with pytest.raises(ValueError): t.encode("z")

def test_save_load(tmp_path):
    t=CharTokenizer("hello"); path=tmp_path/"vocab.json"; t.save(path)
    restored=CharTokenizer.load(path); assert restored.decode(restored.encode("hello")) == "hello"
