from abc import ABC, abstractmethod
from typing import Any, List

from src.utils import resolve_relative_path


class Processor(ABC):
    
    def __init__(self, name: str, model_path: str, providers: List[str]):
        self.name = name
        self.model_path = resolve_relative_path(model_path) # Path to the model
        self.model = None  # Placeholder for the model instance
        self.providers = providers  # Store the execution providers

    @abstractmethod
    def load_model(self) -> Any:
        pass

    def clear_model(self) -> None:
        self.model=None

