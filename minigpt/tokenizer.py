"""Simple deterministic character tokenizer; BPE is a later extension."""
import json
from pathlib import Path

class CharTokenizer:
    def __init__(self, text: str):
        if not text:
            raise ValueError("Tokenizer training text must not be empty")
        self.itos = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"Characters outside vocabulary: {unknown!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        try:
            return "".join(self.itos[i] for i in ids)
        except (IndexError, TypeError) as exc:
            raise ValueError("Invalid token IDs") from exc

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls.__new__(cls)
        obj.itos = data["itos"]
        if not obj.itos or len(set(obj.itos)) != len(obj.itos):
            raise ValueError("Invalid vocabulary")
        obj.stoi = {ch: i for i, ch in enumerate(obj.itos)}
        return obj
