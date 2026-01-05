"""
Dataclass decorator with inline comment extraction for help text.

The @dataclass decorator is a drop-in replacement for stdlib @dataclass that:
- Converts a class into a dataclass
- Extracts inline comments as field metadata['help']
- Handles mutable defaults properly (dict, list, set)
"""
import ast
import inspect
import io
import tokenize
from dataclasses import MISSING
from dataclasses import dataclass as stdlib_dataclass
from dataclasses import field


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
    """
    source = inspect.getsource(cls)
    source_ast = ast.parse(source)

    # Collect comments by line number
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    comment_map = {}
    for tok_type, tok_str, start, _, _ in tokens:
        if tok_type == tokenize.COMMENT:
            lineno = start[0]
            comment_map[lineno] = tok_str.lstrip("# ").strip()

    # Store help text in a class attribute for later use
    field_help = {}

    # Attach metadata to each annotated field
    class_def = source_ast.body[0]
    for node in class_def.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            help_msg = comment_map.get(node.lineno)

            # Pull default if it exists in class dict
            default_value = getattr(cls, name, MISSING)

            # Store help for later retrieval
            if help_msg:
                field_help[name] = help_msg

            # Rewrite attribute with field()
            kwargs = {}
            if help_msg:
                kwargs["metadata"] = {"help": help_msg}
            if default_value is not MISSING:
                # Use default_factory for mutable defaults
                if isinstance(default_value, (dict, list, set)):
                    kwargs["default_factory"] = lambda v=default_value: v.copy() if isinstance(v, dict) else list(v) if isinstance(v, (list, set)) else v
                else:
                    kwargs["default"] = default_value

            setattr(cls, name, field(**kwargs))

    # Convert to dataclass
    dataclass_cls = stdlib_dataclass(cls)
    
    # Store help text as class attribute
    dataclass_cls.__field_help__ = field_help
    
    return dataclass_cls
