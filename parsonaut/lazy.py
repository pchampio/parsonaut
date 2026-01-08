from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, ParamSpec, Type, TypeVar, get_args

from .serialization import Serializable, maybe_import
from .typecheck import Missing, MissingType, get_union_args, is_dict_type, is_flat_tuple_type, is_parsable_type, is_path_type

T = TypeVar("T")
P = ParamSpec("P")

A = ParamSpec("A")
B = TypeVar("B")


TYPECHECK_EAGER = False
TYPE_NAME = "_class"


class Lazy(Generic[T, P], Serializable):
    """
    A mixin class that allows for lazy initialization of a class instance.
    """

    # TODO: we are missing tuple here
    KeyTypes = (
        tuple[type["Lazy"], "Lazy"]
        | tuple[type[bool], bool]
        | tuple[type[int], int]
        | tuple[type[str], str]
        | tuple[type[float], float]
    )

    def __init__(
        self, cls: Type[T] | Callable[P, T], signature: partial | Mapping[str, KeyTypes]
    ) -> None:
        # going around the freezing thingy in __setattr__
        # https://stackoverflow.com/a/4828492
        object.__setattr__(self, "cls", cls)
        object.__setattr__(self, "_signature", signature)

    def __hash__(self) -> int:
        def make_hashable(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, (list, tuple)):
                return tuple(make_hashable(item) for item in obj)
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        items = self.to_dict(
            with_annotations=True, with_class_tag=True, flatten=True
        ).items()
        return hash(tuple((k, make_hashable(v)) for k, v in items))

    def __eq__(self, __value: "object | Lazy") -> bool:
        return hash(self) == hash(__value)

    def __str__(self):
        return lazy_str(self.to_dict(with_class_tag=True))

    def __getattr__(self, x):
        signature = object.__getattribute__(self, "signature")
        if x not in signature:
            return object.__getattribute__(self, x)
        else:
            return signature[x][1]

    def __setattr__(self, *args):
        # This is here for Enum support, otherwise all frozen
        if args[0] in (
            "_value_",
            "_name_",
            "__objclass__",
            "_sort_order_",
        ):
            object.__setattr__(self, args[0], args[1])
        else:
            raise AssertionError("Cannot set attributes of Lazy class")

    @property
    def signature(self) -> Mapping[str, KeyTypes]:
        # This is here to prevent infinite recursion.
        _signature = object.__getattribute__(self, "_signature")
        if isinstance(_signature, partial):
            _signature = _signature()
            object.__setattr__(self, "_signature", _signature)
        return _signature

    @staticmethod
    def is_lazy_type(typ):
        origin = getattr(typ, "__origin__", None)
        if origin is None and isinstance(typ, type) and issubclass(typ, Lazy):
            return True
        elif origin == Lazy:
            return True
        else:
            return False

    @staticmethod
    def from_class(cl: Type[B] | Callable[A, B], *args, **kwargs) -> "Lazy[B, A]":

        if should_typecheck_eagerly():
            sig = Lazy.get_signature(cl, *args, **kwargs)
            return Lazy(cl, sig)
        else:
            return Lazy(
                cl,
                partial(
                    Lazy.get_signature,
                    cl,
                    *args,
                    **kwargs,
                ),
            )

    @staticmethod
    def get_signature(cl, *args, **kwargs) -> Mapping[str, KeyTypes]:
        from .parsable import Parsable  # needed for some asserts

        func = cl.__init__
        signature = get_signature(func, *args, **kwargs)

        res = dict()
        for name, (typ, value) in signature.items():

            # enforce default values
            if Lazy.is_lazy_type(typ):
                # fill in missing default
                if value is Missing and (subtyp := get_args(typ)):
                    subtyp, *_ = subtyp
                    assert issubclass(subtyp, Parsable)
                    value = Lazy.from_class(subtyp)  # type: ignore
                # or ensure correct type
                else:
                    assert isinstance(
                        value, Lazy
                    ), f"Expected value to be parsable or a Lazy. Got {type(value)}"
                res[name] = (typ, value)
                continue

            # fill in parsable type if value is Parsable
            if isinstance(value, Lazy):
                if typ == MissingType:
                    typ = Lazy
                else:
                    assert Lazy.is_lazy_type(typ)
                res[name] = (typ, value)
                continue

            # Skip variables without type annotation - the user can fill them in with to_eager() call
            if typ == MissingType:
                continue

            if not is_parsable_type(typ):
                continue

            # check if the provided value is parsable and matches the annotation
            assert value == Missing or is_parsable_type(typ, value), (
                f"Provided value {name}={value} does not match "
                f"the provided annotation {name}: {typ}"
            )
            res[name] = (typ, value)

        return res

    def copy(self: "Lazy[B, A]", fields: dict | None = None) -> "Lazy[B, A]":
        dct = self.to_dict(with_class_tag=True, flatten=True)
        if fields is None:
            return Lazy.from_dict(dct)

        # Wrap user dict values in fields before flattening
        cls_sig = get_signature(self.cls.__init__)
        for k in list(fields.keys()):
            if k in cls_sig and k != TYPE_NAME:
                typ, _ = cls_sig[k]
                if is_dict_type(typ, None) and isinstance(fields[k], dict):
                    fields[k] = _DictValue(fields[k])
        
        fields = flatten_dict(fields)
        
        # Convert Path strings to Path objects before merging
        cls_sig = get_signature(self.cls.__init__)
        for k in list(fields.keys()):
            if k in cls_sig:
                typ, _ = cls_sig[k]
                if is_path_type(typ, None) and isinstance(fields[k], str):
                    fields[k] = Path(fields[k])

        # Ensure we iterate over class types first (if present)
        fields = dict(
            sorted(
                fields.items(),
                key=lambda item: (not item[0].endswith(f".{TYPE_NAME}"), item[0]),
            )
        )
        for field, new_val in fields.items():
            # We are attempting to change subclass
            if field.endswith(TYPE_NAME) and dct[field] != new_val:
                prefix = field.removesuffix(TYPE_NAME)
                old_class_fields = {k for k in list(dct.keys()) if k.startswith(prefix)}
                new_class_fields = Lazy.from_class(maybe_import(new_val)).to_dict(
                    flatten=True, with_class_tag=True
                )
                new_class_fields = {
                    f"{prefix}{k}": v for k, v in new_class_fields.items()
                }

                # Remove old class fields completely -> if the classes share some fields, we always use the new default values.
                # That is, if the user does not specify the value explicitly, we use default from the new class.
                # If the new class has the attribute missing, it will be missing too even if the old class had it.\

                # Warning: The defaults are currently taken from the new class init. NOT from the Choices default!
                for p in old_class_fields:
                    del dct[p]

                # Add new class fields if defaults are available (to not trigger assert later)
                for k, v in new_class_fields.items():
                    if k not in dct:
                        dct[k] = v

                # Finally set the class type
                dct[field] = new_val
            # Simple case, where user simply changes base args
            else:
                assert (
                    field in dct
                ), f"Attempted to copy with {field=} that is not present."
                dct[field] = new_val

        return Lazy.from_dict(dct)

    def to_dict(
        self,
        recursive: bool = True,
        with_annotations: bool = False,
        with_class_tag: bool = False,
        with_class_tag_as_str: bool = False,
        flatten: bool = False,
        tuples_as_lists: bool = False,
    ):
        dct = dict()
        if with_class_tag:
            dct[TYPE_NAME] = self.cls
        elif with_class_tag_as_str:
            dct[TYPE_NAME] = f"{self.cls.__module__}.{self.cls.__name__}"
        for k, (typ, value) in sorted(self.signature.items()):
            if tuples_as_lists and is_flat_tuple_type(typ, value):
                value = list(value)

            if is_path_type(typ, value):
                value = str(value) if value is not None else None

            if Lazy.is_lazy_type(typ):
                if recursive:
                    assert isinstance(value, Lazy)
                    value = value.to_dict(
                        recursive=recursive,
                        with_annotations=with_annotations,
                        with_class_tag=with_class_tag,
                        with_class_tag_as_str=with_class_tag_as_str,
                    )
                dct[k] = value
            else:
                if with_annotations:
                    dct[k] = (typ, value)
                else:
                    dct[k] = value

        if flatten:
            # Wrap user dict values before flattening
            for k, (typ, val) in sorted(self.signature.items()):
                if k in dct and isinstance(dct[k], dict) and not Lazy.is_lazy_type(typ):
                    # This is a user dict value, not a nested Lazy config
                    dct[k] = _DictValue(dct[k])
            dct = flatten_dict(dct)

        return dct

    @staticmethod
    def from_dict(dct):
        # For now we assume the dict contains TYPE_NAME
        # In the future, we should be able to infer the TYPE_NAME also for sub-classes from defaults
        if any("." in k for k in dct):
            dct = unflatten_dict(dct)

        signature = dict()

        cls = maybe_import(dct[TYPE_NAME])

        # Get ALL parameter types (not just parsable ones) for type conversion
        raw_sig = get_signature(cls.__init__)
        lazy_sig = Lazy.get_signature(cls)
        
        for k, v in dct.items():
            if k == TYPE_NAME:
                continue
            elif isinstance(v, dict):
                if TYPE_NAME in v:
                    signature[k] = Lazy.from_dict(v)
                elif k in lazy_sig:
                    typ, _ = lazy_sig[k]
                    # Check if type is dict or a union containing dict
                    union_args, _ = get_union_args(typ)
                    is_user_dict = is_dict_type(typ, None) or dict in union_args
                    if is_user_dict:
                        signature[k] = v
                    else:
                        signature[k] = Lazy.from_dict(v)
                else:
                    signature[k] = v
            elif isinstance(v, str):
                if k in lazy_sig:
                    typ, _ = lazy_sig[k]
                    if is_path_type(typ, None):
                        signature[k] = Path(v)
                    else:
                        signature[k] = v
                else:
                    signature[k] = v
            else:
                # We store tuples as lists in json/yaml. Here we convert them back.
                if isinstance(v, list) and k in raw_sig:
                    typ, _ = raw_sig[k]
                    # Only convert to tuple if the type annotation is a tuple type
                    if is_flat_tuple_type(typ, None):
                        v = tuple(v)
                signature[k] = v
        
        # Note: Allow loading dict that can be missing non-parsable parameters.
        #    Such a dict is created eg when usign Parsable.to_checkpoint() --> we only export parsable parameters and rest is ommited.
        #    When we load it back, we want Lazy to fill in the non-parsable stuff with defaults from the class.
        return Lazy.from_class(cls, **signature)

    def to_eager(self, *args: P.args, **kwargs: P.kwargs) -> T:
        assert not args, "Please pass named parameters only."
        kwargs2 = self.to_dict(recursive=False)
        kwargs = {**kwargs2, **kwargs}
        kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, MissingType)}

        return self.cls(
            *args,
            **kwargs,
        )


class Choices(Lazy, Enum):
    def __new__(cls, value):
        assert isinstance(
            value, Lazy
        ), f"Choice values must be an instance of Lazy. Got {type(value)}"
        obj = Lazy.__new__(cls)
        return obj

    def __init__(self, *args):
        (orig_lazy,) = args
        object.__setattr__(self, "cls", orig_lazy.cls)
        object.__setattr__(self, "_signature", orig_lazy._signature)


def should_typecheck_eagerly():
    return TYPECHECK_EAGER


class typecheck_eager:
    def __init__(self):
        global TYPECHECK_EAGER
        TYPECHECK_EAGER = True

    def __enter__(self):
        pass

    def __exit__(self, *args, **kws):
        global TYPECHECK_EAGER
        TYPECHECK_EAGER = False


def set_typecheck_eager(eager: bool = True):
    global TYPECHECK_EAGER
    TYPECHECK_EAGER = eager


def get_signature(func: Callable, *args, **kwargs) -> dict[str, tuple[Type, Any]]:
    """Get the signature of a function, including the types of the arguments."""
    from dataclasses import MISSING as DC_MISSING
    from dataclasses import fields as dataclass_fields
    from inspect import _empty, signature

    sig = signature(func)
    if "self" in sig.parameters:
        bound = sig.bind_partial(None, *args, **kwargs)
    else:
        bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()

    # Get dataclass defaults if applicable (resolves field factories to actual values)
    dataclass_defaults = {}
    if hasattr(func, '__qualname__') and '.' in func.__qualname__:
        class_name = func.__qualname__.rsplit('.', 1)[0]
        cls = func.__globals__.get(class_name)
        if cls and hasattr(cls, '__dataclass_fields__'):
            for f in dataclass_fields(cls):
                if f.default is not DC_MISSING:
                    dataclass_defaults[f.name] = f.default
                elif f.default_factory is not DC_MISSING:
                    dataclass_defaults[f.name] = f.default_factory()

    ret = dict()
    for param_name, param in bound.signature.parameters.items():
        if param_name == "self":
            continue

        value = bound.arguments.get(
            param_name, param.default if param.default != _empty else Missing
        )

        # Resolve dataclass field factory to actual default
        if param_name in dataclass_defaults and type(value).__name__ == '_HAS_DEFAULT_FACTORY_CLASS':
            value = dataclass_defaults[param_name]

        annotation = param.annotation if param.annotation != _empty else MissingType
        ret[param_name] = (annotation, value)

    return ret


class _DictValue:
    """Wrapper to prevent user dict values from being flattened."""
    def __init__(self, value):
        self.value = value


def flatten_dict(dct: dict) -> dict:

    def _flatten(dct, prefix: str):
        out = list()
        for k, v in dct.items():
            if prefix:
                k = f"{prefix}.{k}"
            if isinstance(v, _DictValue):
                out.append((k, v.value))
            elif isinstance(v, dict):
                out.extend(_flatten(v, prefix=k))
            else:
                out.append((k, v))
        return out

    return dict(_flatten(dct, ""))


def unflatten_dict(flat: dict) -> dict:

    base = {}
    for key, value in flat.items():
        root = base

        if "." in key:
            *parts, key = key.split(".")

            for part in parts:
                # This should ignore choice flags such as --encoder ENCODER
                if part in root and not isinstance(root[part], dict):
                    root[part] = {}
                if part not in root:
                    root[part] = {}
                root = root[part]

        root[key] = value

    return base


def lazy_str(dct: dict, level: int = 1):

    def format_attr(k, v):
        if isinstance(v, str):
            return f"{k}='{v}'"
        else:
            return f"{k}={v}"

    header = f'{dct["_class"].__name__}'
    attrs = [
        (
            f"{k}={lazy_str(v, level=level + 1)}"
            if isinstance(v, dict)
            else format_attr(k, v)
        )
        for k, v in dct.items()
        if k != "_class"
    ]
    indent = "    "
    attrs = f",\n{indent * level}".join(attrs)
    out = f"{header}(\n{indent * level}{attrs},\n{indent * (level - 1)})"
    return out
