"""
Example showcasing the @dataclass decorator from parsonaut.

This is a drop-in replacement for the standard library @dataclass that
automatically extracts inline comments as help text for CLI arguments.
"""

from pathlib import Path

from parsonaut import ArgumentParser, dataclass


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    model_name: str = "resnet50"                    # name of the model architecture
    learning_rate: float = 0.001                    # initial learning rate
    batch_size: int = 32                            # training batch size
    epochs: int = 100                               # number of training epochs
    weight_decay: float = 0.0001                    # L2 regularization weight decay
    momentum: float = 0.9                           # SGD momentum
    use_gpu: bool = True                            # whether to use GPU acceleration
    checkpoint_dir: Path = Path("/tmp/checkpoints") # directory to save model checkpoints
    num_workers: int = 4                            # number of data loading workers
    seed: int = 42                                  # random seed for reproducibility


@dataclass
class DataConfig:
    """Configuration for dataset"""
    
    dataset_path: Path = Path("/data")  # path to dataset directory
    train_split: float = 0.8            # fraction of data for training
    val_split: float = 0.1              # fraction of data for validation
    test_split: float = 0.1             # fraction of data for testing
    image_size: int = 224               # input image size
    augmentation: bool = True           # whether to use data augmentation


if __name__ == "__main__":
    print("=" * 70)
    print("Training Configuration Example")
    print("=" * 70)
    
    # Create parser and add dataclass options
    parser = ArgumentParser(description="Training script with dataclass configuration")
    
    # You can add multiple dataclasses as separate groups
    from parsonaut import Lazy
    training_config = Lazy.from_class(TrainingConfig)
    data_config = Lazy.from_class(DataConfig)
    
    parser.add_options(training_config, dest="train")
    parser.add_options(data_config, dest="data")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Access the configurations
    print("\nParsed Training Configuration:")
    print(f"  Model: {args.train.model_name}")
    print(f"  Learning rate: {args.train.learning_rate}")
    print(f"  Batch size: {args.train.batch_size}")
    print(f"  Epochs: {args.train.epochs}")
    print(f"  Weight decay: {args.train.weight_decay}")
    print(f"  Momentum: {args.train.momentum}")
    print(f"  Use GPU: {args.train.use_gpu}")
    print(f"  Checkpoint dir: {args.train.checkpoint_dir}")
    print(f"  Num workers: {args.train.num_workers}")
    print(f"  Seed: {args.train.seed}")
    
    print("\nParsed Data Configuration:")
    print(f"  Dataset path: {args.data.dataset_path}")
    print(f"  Train split: {args.data.train_split}")
    print(f"  Val split: {args.data.val_split}")
    print(f"  Test split: {args.data.test_split}")
    print(f"  Image size: {args.data.image_size}")
    print(f"  Augmentation: {args.data.augmentation}")
    
    print("\nExample commands:")
    print("  python dataclass_only_example.py --train.learning_rate 0.01 --train.epochs 50")
    print("  python dataclass_only_example.py --data.dataset_path /my/data --train.batch_size 64")
    print("  python dataclass_only_example.py --help")
