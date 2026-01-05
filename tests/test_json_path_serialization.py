import tempfile
from pathlib import Path

import pytest

from parsonaut import Parsable


class ModelWithDictAndPath(Parsable):
    def __init__(
        self,
        config: dict = {"lr": 0.001},
        output_path: Path = Path("/tmp"),
        name: str = "model",
    ) -> None:
        self.config = config
        self.output_path = output_path
        self.name = name


def test_serialize_to_yaml_with_dict_and_path():
    model = ModelWithDictAndPath.as_lazy(
        config={"lr": 0.01, "batch_size": 32},
        output_path=Path("/my/custom/path"),
        name="test_model",
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        model.to_yaml(f.name)
        yaml_path = f.name

    loaded = ModelWithDictAndPath.from_yaml(yaml_path)

    assert loaded.config == {"lr": 0.01, "batch_size": 32}
    assert loaded.output_path == Path("/my/custom/path")
    assert loaded.name == "test_model"


def test_serialize_to_json_with_dict_and_path():
    model = ModelWithDictAndPath.as_lazy(
        config={"epochs": 100},
        output_path=Path("/data/results"),
        name="experiment_1",
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        model.to_json(f.name)
        json_path = f.name

    loaded = ModelWithDictAndPath.from_json(json_path)

    assert loaded.config == {"epochs": 100}
    assert loaded.output_path == Path("/data/results")
    assert loaded.name == "experiment_1"


def test_roundtrip_yaml():
    model = ModelWithDictAndPath.as_lazy(
        config={"param1": "value1", "param2": 123},
        output_path=Path("/test/path"),
        name="roundtrip",
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        model.to_file(f.name)
        file_path = f.name

    loaded = ModelWithDictAndPath.from_file(file_path)

    assert loaded == model


def test_roundtrip_json():
    model = ModelWithDictAndPath.as_lazy(
        config={"nested": {"key": "value"}},
        output_path=Path("/another/path"),
        name="json_roundtrip",
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        model.to_file(f.name)
        file_path = f.name

    loaded = ModelWithDictAndPath.from_file(file_path)

    assert loaded == model
