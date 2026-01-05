"""
Example demonstrating JSON dict and Path argument support.

usage: json_path_example.py [-h] [--hyperparams json] [--output_dir path] [--name str] [--max_files int]

options:
  -h, --help            show this help message and exit
  --hyperparams json    Hyperparameters as JSON dict
  --output_dir path     Output directory path
  --name str            Experiment name
  --max_files int       Maximum number of files to process
"""

from pathlib import Path

from parsonaut import Parsable


class ExperimentConfig(Parsable):
    def __init__(
        self,
        hyperparams: dict = {"learning_rate": 0.001, "batch_size": 32},  # hyperparameters as JSON dict
        output_dir: Path = Path("/tmp/experiments"),  # output directory path
        name: str = "default_experiment",             # experiment name
        max_files: int = 100,                         # maximum number of files to process
    ):
        self.hyperparams = hyperparams
        self.output_dir = output_dir
        self.name = name
        self.max_files = max_files


if __name__ == "__main__":
    params = ExperimentConfig.parse_args()

    print("\nParsed configuration:")
    print(f"  Hyperparams dict: {params.hyperparams}")
    print(f"  Output dir: {params.output_dir} (type: {type(params.output_dir).__name__})")
    print(f"  Name: {params.name}")
    print(f"  Max files: {params.max_files}")

    print("\nExample usage:")
    print('  python json_path_example.py --hyperparams \'{"learning_rate": 0.01}\' --output_dir /my/path')

    eager = params.to_eager()
    print("\nEager instance:")
    print(f"  hyperparams: {eager.hyperparams}")
    print(f"  output_dir: {eager.output_dir}")
