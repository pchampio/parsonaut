"""Example showing inline comments automatically become help text."""

from pathlib import Path

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
    ):
        self.name = name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dir = output_dir


if __name__ == "__main__":
    params = ExperimentConfig.parse_args()
    
    print("Parsed configuration:")
    print(f"  name: {params.name}")
    print(f"  learning_rate: {params.learning_rate}")
    print(f"  batch_size: {params.batch_size}")
    print(f"  epochs: {params.epochs}")
    print(f"  output_dir: {params.output_dir}")
