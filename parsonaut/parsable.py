from abc import ABCMeta
from typing import Callable, Type

from .lazy import Lazy, P, T
from .serialization import Serializable, is_module_available, open_best


def _extract_field_help_from_init(cls) -> dict[str, str]:
    """Extract inline comments from __init__ method parameters."""
    import ast
    import inspect
    import io
    import textwrap
    import tokenize
    
    field_help = {}
    try:
        if hasattr(cls, '__init__'):
            source = inspect.getsource(cls.__init__)
            source = textwrap.dedent(source)
            
            # Collect comments by line number
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            comment_map = {}
            for tok_type, tok_str, start, _, _ in tokens:
                if tok_type == tokenize.COMMENT:
                    lineno = start[0]
                    comment_map[lineno] = tok_str.lstrip("# ").strip()
            
            # Parse the __init__ method to find annotated parameters
            source_ast = ast.parse(source)
            func_def = source_ast.body[0]
            if isinstance(func_def, ast.FunctionDef):
                for arg in func_def.args.args:
                    if arg.annotation and arg.arg != 'self':
                        help_msg = comment_map.get(arg.lineno)
                        if help_msg:
                            field_help[arg.arg] = help_msg
    except (OSError, TypeError, IndexError):
        # Source not available (e.g., interactive mode, built-in classes)
        pass
    
    return field_help


class LazyFieldHelp:
    """Descriptor that lazily extracts field help from __init__ comments."""
    
    def __get__(self, obj, cls):
        if cls is None:
            return self
        cache_attr = '_cached_field_help'
        if not hasattr(cls, cache_attr):
            setattr(cls, cache_attr, _extract_field_help_from_init(cls))
        return getattr(cls, cache_attr)


class ParsableMeta(ABCMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        cls.__field_help__ = LazyFieldHelp()
        return cls
    
    def __call__(cls, *args, **kwargs):
        cfg = Lazy.from_class(cls, *args, **kwargs)

        # https://stackoverflow.com/a/73923070/8378586
        obj = cls.__new__(cls, *args, **kwargs)
        obj._cfg = cfg
        # Initialize the final object
        obj.__init__(*args, **kwargs)
        return obj


class class_or_instance_method(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, instance, owner):
        if instance is not None:
            class_or_instance = instance
        else:
            class_or_instance = owner

        def newfunc(*args, **kwargs):
            return self.f(class_or_instance, *args, **kwargs)

        return newfunc


if is_module_available("torch"):
    import torch
else:
    torch = None


class Parsable(Serializable, metaclass=ParsableMeta):

    _cfg: Lazy

    @class_or_instance_method
    def as_lazy(cls_or_self, *args, **kwargs) -> Lazy:
        if isinstance(cls_or_self, Parsable):
            assert not args, "Cannot override once configured Parsable."
            assert not kwargs, "Cannot override once configured Parsable."
            return cls_or_self._cfg
        else:
            return Lazy.from_class(cls_or_self, *args, **kwargs)

    def to_dict(
        self,
        with_class_tag: bool = False,
        with_class_tag_as_str: bool = False,
        flatten: bool = False,
        tuples_as_lists: bool = False,
    ):
        return self._cfg.to_dict(
            with_class_tag=with_class_tag,
            with_class_tag_as_str=with_class_tag_as_str,
            flatten=flatten,
            tuples_as_lists=tuples_as_lists,
        )

    @classmethod
    def from_dict(cls, dct) -> Lazy:
        return Lazy.from_class(cls).copy(dct)

    @classmethod
    def parse_args(cls: Type[T] | Callable[P, T], *args, **kwargs) -> Lazy[T, P]:
        from .parse import ArgumentParser

        parser = ArgumentParser()
        parser.add_options(cls.as_lazy(*args, **kwargs))
        params = parser.parse_args()
        return params

    # Methods related to torch serialization
    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self):
        raise NotImplementedError()

    @classmethod
    def from_checkpoint(cls, pth, map_location=None):
        assert (
            torch is not None
        ), f"Loading {cls} from checkpoint requires torch installed."

        pth = str(pth).rstrip("/")
        obj = cls.from_file(f"{pth}/config.yaml").to_eager()

        with open_best(f"{pth}/weights.pt", "rb") as f:
            state_dict = torch.load(f, map_location=map_location)
        obj.load_state_dict(state_dict)
        return obj

    def to_checkpoint(self, pth):
        assert (
            torch is not None
        ), f"Saving {self.__class__} to checkpoint requires torch installed."

        pth = str(pth)
        self.to_file(f"{pth}/config.yaml")
        state_dict = self.state_dict()
        with open_best(f"{pth}/weights.pt", "wb") as f:
            torch.save(state_dict, f)
