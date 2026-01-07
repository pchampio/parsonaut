"""Example showing inline comments automatically become help text."""

from pathlib import Path
from typing import Any

from parsonaut import Parsable

class ExperimentConfig(Parsable):
    """Configuration for machine learning experiment"""
    
    def __init__(
        self,
        name: str = "experiment1",           # experiment name
        learning_rate: float = 0.001,        # learning rate for optimizer
        batch_size: int = 32,                # training batch size
        epochs: int = 10,                    # number of training epochs
        output_dir: Path = Path("/tmp"),     # output directory
        tags: list[str] = [],                # experiment tags
        custom_config: list[Any] | str | None = None,  # custom config as list, string, or None
    ):
        self.name = name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dir = output_dir
        self.tags = tags
        self.custom_config = custom_config


if __name__ == "__main__":
    params = ExperimentConfig.parse_args()
    
    print("Parsed configuration:")
    print(f"  name: {params.name}")
    print(f"  learning_rate: {params.learning_rate}")
    print(f"  batch_size: {params.batch_size}")
    print(f"  epochs: {params.epochs}")
    print(f"  output_dir: {params.output_dir}")
    print(f"  tags: {params.tags}")
    print(f"  custom_config: {params.custom_config}", type(params.custom_config))
