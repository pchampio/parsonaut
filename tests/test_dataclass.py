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
    """Field help is lazily extracted via __field_help__ when accessed."""
    field_help = SimpleConfig.__field_help__
    
    assert field_help["name"] == "user name"
    assert field_help["age"] == "age in years"
    assert field_help["value"] == "some value"


def test_parsable_with_custom_values():
    config = SimpleConfig(name="custom", age=30)
    assert config.name == "custom"
    assert config.age == 30
    assert config.value == 3.14


@dataclass
class BaseConfig:
    model_path: str = "/path/to/model"  # Path to the model
    language: str = "en"                 # Language code
    optional_field: str | None = None    # Optional field with help


@dataclass
class ChildConfig(BaseConfig):
    learning_rate: float = 0.001  # Learning rate for training
    batch_size: int = 32          # Batch size


def test_field_help_inheritance():
    """Child dataclass should inherit __field_help__ from parent."""
    field_help = getattr(ChildConfig, '__field_help__', {})
    
    # Parent fields
    assert field_help.get("model_path") == "Path to the model"
    assert field_help.get("language") == "Language code"
    assert field_help.get("optional_field") == "Optional field with help"
    
    # Child fields
    assert field_help.get("learning_rate") == "Learning rate for training"
    assert field_help.get("batch_size") == "Batch size"


def test_field_help_inheritance_values():
    """Child can access inherited field values."""
    config = ChildConfig()
    assert config.model_path == "/path/to/model"
    assert config.language == "en"
    assert config.optional_field is None
    assert config.learning_rate == 0.001
    assert config.batch_size == 32
