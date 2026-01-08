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


def _extract_field_help(cls) -> dict[str, str]:
    """Lazily extract inline comments from class field definitions."""
    import ast
    import inspect
    import io
    import tokenize
    
    # Inherit field_help from parent classes
    field_help = {}
    for base in cls.__mro__[1:]:
        parent_help = getattr(base, '__field_help__', None)
        if parent_help is not None:
            # If parent has lazy descriptor, access triggers extraction
            if isinstance(parent_help, dict):
                field_help.update(parent_help)
    
    try:
        source = inspect.getsource(cls)
        source_ast = ast.parse(source)
        
        # Collect comments by line number
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        comment_map = {}
        for tok_type, tok_str, start, _, _ in tokens:
            if tok_type == tokenize.COMMENT:
                lineno = start[0]
                comment_map[lineno] = tok_str.lstrip("# ").strip()
        
        # Find annotated fields and their comments
        class_def = source_ast.body[0]
        for node in class_def.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                help_msg = comment_map.get(node.lineno)
                if help_msg:
                    field_help[name] = help_msg
    except (OSError, TypeError, IndexError):
        pass
    
    return field_help


class LazyFieldHelp:
    """Descriptor that lazily extracts field help from class definition comments."""
    
    def __get__(self, obj, cls):
        if cls is None:
            return self
        cache_attr = '_cached_field_help'
        if not hasattr(cls, cache_attr):
            setattr(cls, cache_attr, _extract_field_help(cls))
        return getattr(cls, cache_attr)


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
    dataclass_cls.__field_help__ = LazyFieldHelp()
    
    return dataclass_cls
