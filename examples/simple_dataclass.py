"""
Simple example using @dataclass from parsonaut.

Shows how inline comments automatically become CLI help text.
"""

from pathlib import Path

from parsonaut import ArgumentParser, Lazy, dataclass


@dataclass
class Config:
    """Simple configuration example"""
    
    name: str = "experiment"        # name of the experiment
    lr: float = 0.001               # learning rate
    batch_size: int = 32            # batch size for training
    output: Path = Path("/tmp")     # output directory


if __name__ == "__main__":
    # Method 1: Using ArgumentParser directly
    parser = ArgumentParser()
    parser.add_options(Lazy.from_class(Config))
    config = parser.parse_args()
    
    print("Configuration:")
    print(f"  Name: {config.name}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Output: {config.output}")
    
    # You can also instantiate the dataclass directly
    instance = config.to_eager()
    print(f"\nDataclass instance: {instance}")
