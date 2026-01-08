"""
Dataclass decorator with lazy inline comment extraction for help text.

The @dataclass decorator is a drop-in replacement for stdlib @dataclass that:
- Converts a class into a dataclass
- Lazily extracts inline comments via __field_help__ (only when accessed)
- Handles mutable defaults properly (dict, list, set)

Help text extraction is deferred until __field_help__ is accessed (e.g., when
--help is requested), avoiding expensive AST parsing during normal execution.
"""
from dataclasses import MISSING
from dataclasses import dataclass as stdlib_dataclass
from dataclasses import field

from .help_extraction import LazyFieldHelp, extract_field_help_from_class


def dataclass(cls):
    """
    Decorator that converts a class into a dataclass and extracts inline comments
    as field metadata['help'] for use in argument parsing.
    
    Usage:
        @dataclass
        class Config:
            name: str = "default"  # user name
            age: int = 25          # age in years
    
    The inline comments become help text when using ArgumentParser.
    Help extraction is deferred until --help is actually used.
    """
    # Handle mutable defaults by converting to field() with default_factory
    for name in list(getattr(cls, '__annotations__', {}).keys()):
        default_value = getattr(cls, name, MISSING)
        if default_value is not MISSING and isinstance(default_value, (dict, list, set)):
            setattr(cls, name, field(
                default_factory=lambda v=default_value: v.copy() if isinstance(v, dict) else list(v) if isinstance(v, (list, set)) else v
            ))
    
    # Convert to dataclass
    dataclass_cls = stdlib_dataclass(cls)
    
    # Install lazy descriptor for field help (defers AST parsing until accessed)
    dataclass_cls.__field_help__ = LazyFieldHelp(extract_field_help_from_class)
    
    return dataclass_cls
