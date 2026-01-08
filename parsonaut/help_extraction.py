"""
Lazy extraction of inline comments for help text.

This module provides utilities for extracting inline comments from Python source
code to use as help text in argument parsing. Extraction is deferred until
actually needed (e.g., when --help is requested).
"""
import ast
import inspect
import io
import textwrap
import tokenize


def _build_comment_map(source: str) -> dict[int, str]:
    """Build a map of line numbers to inline comments."""
    comment_map = {}
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for tok_type, tok_str, start, _, _ in tokens:
        if tok_type == tokenize.COMMENT:
            comment_map[start[0]] = tok_str.lstrip("# ").strip()
    return comment_map


def extract_field_help_from_class(cls) -> dict[str, str]:
    """Extract inline comments from class field definitions (for @dataclass)."""
    # Inherit field_help from parent classes
    field_help = {}
    for base in cls.__mro__[1:]:
        parent_help = getattr(base, '__field_help__', None)
        if parent_help is not None and isinstance(parent_help, dict):
            field_help.update(parent_help)
    
    try:
        source = inspect.getsource(cls)
        comment_map = _build_comment_map(source)
        source_ast = ast.parse(source)
        
        class_def = source_ast.body[0]
        for node in class_def.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                help_msg = comment_map.get(node.lineno)
                if help_msg:
                    field_help[node.target.id] = help_msg
    except (OSError, TypeError, IndexError, tokenize.TokenizeError):
        pass
    
    return field_help


def extract_field_help_from_init(cls) -> dict[str, str]:
    """Extract inline comments from __init__ method parameters (for Parsable)."""
    field_help = {}
    try:
        if not hasattr(cls, '__init__'):
            return field_help
            
        source = textwrap.dedent(inspect.getsource(cls.__init__))
        comment_map = _build_comment_map(source)
        source_ast = ast.parse(source)
        
        func_def = source_ast.body[0]
        if isinstance(func_def, ast.FunctionDef):
            for arg in func_def.args.args:
                if arg.annotation and arg.arg != 'self':
                    help_msg = comment_map.get(arg.lineno)
                    if help_msg:
                        field_help[arg.arg] = help_msg
    except (OSError, TypeError, IndexError, tokenize.TokenizeError):
        pass
    
    return field_help


class LazyFieldHelp:
    """Descriptor that lazily extracts field help on first access."""
    
    def __init__(self, extractor):
        self._extractor = extractor
    
    def __get__(self, obj, cls):
        if cls is None:
            return self
        cache_attr = '_cached_field_help'
        if not hasattr(cls, cache_attr):
            setattr(cls, cache_attr, self._extractor(cls))
        return getattr(cls, cache_attr)
