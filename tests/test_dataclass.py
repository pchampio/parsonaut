from dataclasses import fields

from parsonaut import dataclass


@dataclass
class SimpleConfig:
    name: str = "test"          # user name
    age: int = 25               # age in years
    value: float = 3.14         # some value


def test_parsable_creates_fields():
    config = SimpleConfig()
    assert config.name == "test"
    assert config.age == 25
    assert config.value == 3.14


def test_parsable_extracts_comments():
    flds = fields(SimpleConfig)
    metadata = {f.name: f.metadata.get("help") for f in flds}
    
    assert metadata["name"] == "user name"
    assert metadata["age"] == "age in years"
    assert metadata["value"] == "some value"


def test_parsable_with_custom_values():
    config = SimpleConfig(name="custom", age=30)
    assert config.name == "custom"
    assert config.age == 30
    assert config.value == 3.14
