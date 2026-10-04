from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class Utf8JsonlLogger:
    def __init__(self, path: str | Path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.path.open("a",encoding="utf-8",newline="\n")
    def write(self, event: str, payload: Mapping[str, Any]):
        self.handle.write(json.dumps({"event":event,**payload},ensure_ascii=False)+"\n");self.handle.flush()
    def close(self): self.handle.close()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
