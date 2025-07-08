from dataclasses import dataclass

@dataclass
class FileModel:
    path: str
    data: dict