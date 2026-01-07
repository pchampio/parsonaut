from builtins import Ellipsis
from functools import lru_cache
from pathlib import Path
from types import UnionType
from typing import Any, Type, Union, get_args, get_origin

BASIC_TYPES = (int, float, bool, str)
EXTENDED_TYPES = (int, float, bool, str, dict, Path)


class MissingType:
    def __repr__(self):
        return "???"


Missing = MissingType()


def _is_basic_type(typ: Type, basic_typ, value: Any | None = None) -> bool:
    assert basic_typ in BASIC_TYPES
    typ_ok = typ == basic_typ
    if value is None:
        return typ_ok
    else:
        return typ_ok and isinstance(value, basic_typ)


def is_float_type(typ: Type, value: Any | None = None) -> bool:
    return _is_basic_type(typ, float, value)


def is_int_type(typ: Type, value: Any | None = None) -> bool:
    return _is_basic_type(typ, int, value)


def is_bool_type(typ: Type, value: Any | None = None) -> bool:
    return _is_basic_type(typ, bool, value)


def is_str_type(typ: Type, value: Any | None = None) -> bool:
    return _is_basic_type(typ, str, value)


def is_dict_type(typ: Type, value: Any | None = None) -> bool:
    typ_ok = typ == dict or get_origin(typ) == dict
    if value is None:
        return typ_ok
    else:
        return typ_ok and isinstance(value, dict)


def is_path_type(typ: Type, value: Any | None = None) -> bool:
    typ_ok = typ == Path
    if value is None:
        return typ_ok
    else:
        return typ_ok and isinstance(value, Path)


def is_list_type(typ: Type, value: Any | None = None) -> bool:
    typ_ok = typ == list or get_origin(typ) == list
    if value is None:
        return typ_ok
    else:
        return typ_ok and isinstance(value, list)


def is_flat_list_type(typ: Type, value: Any | None = None) -> bool:
    """Check if `typ` is a flat list type, optionally validate a value against it.

    The inner type must be one of the basic types - int, float, bool or str.
    All elements must be of the same type.

    Examples of passing inputs:

            - list[int], [1, 2]
            - list[str], ["a", "b", "c"]

    Examples of failing inputs:

            - list[int], [1, "2"]
            - list, [1, 2, 3]

    Args:
        typ (Type): The type to check.
        value (Any | None, optional): The value to compare against the type. Defaults to None.

    Returns:
        bool: True if the type is a flat list type, False otherwise.
    """
    args = get_args(typ)

    if value is None:
        return _is_flat_list_type(typ, args)
    else:
        if not (isinstance(value, list) and _is_flat_list_type(typ, args)):
            return False
        # For list[Any], accept any list contents
        if len(args) > 0 and args[0] is Any:
            return True
        # For other types, validate element types
        return len(value) == 0 or all(isinstance(item, args[0]) for item in value)


@lru_cache(maxsize=1)
def _is_flat_list_type(typ: Type, args):
    return (
        # Container is a list and contains inner annotation
        get_origin(typ) == list
        # The inner annotation is a BasicType or Any
        and len(args) > 0
        and (args[0] in BASIC_TYPES or args[0] is Any)
    )


def is_flat_tuple_type(typ: Type, value: Any | None = None) -> bool:
    """Check if `typ` is a flat tuple type, optionally validate a value against it.

    The inner type must be one of the basic types - int, float, bool or str.
    If more inner type values are provided, they must all be of the same type.

    Examples of passing inputs:

            - tuple[int, int], (1, 2)
            - tuple[int, ...], (1, 2, 3)

    Examples of failing inputs:

            - tuple[int, str], (1, 2)
            - tuple[int, int], (1, 2, 3)

    Args:
        typ (Type): The type to check.
        value (Any | None, optional): The value to compare against the type. Defaults to None.

    Returns:
        bool: True if the type is a flat tuple type, False otherwise.
    """
    args = get_args(typ)

    if value is None:
        return _is_flat_tuple_type(typ, args)
    else:
        return (
            isinstance(value, tuple)
            and _is_flat_tuple_type(typ, args)
            and (len(args) == len(value) or Ellipsis in args)
            and all(isinstance(item, args[0]) for item in value)
        )


def is_union_type(typ: Type):
    """Check if typ is a Union type."""
    return (
        isinstance(typ, UnionType)
        or getattr(typ, "__origin__", None) is Union
    )


def get_union_args(typ: Type):
    """Get union arguments, separating None from other types."""
    if not is_union_type(typ):
        return [], False
    
    args = get_args(typ)
    non_none_args = [a for a in args if a is not type(None)]
    has_none = len(non_none_args) < len(args)
    return non_none_args, has_none


def is_optional_single_type(typ: Type, value: Any | None):
    """
    Returns (True, T) if tp is exactly Optional[T], i.e., Union[T, None] with only one non-None type.
    Returns (False, tp) otherwise.
    """
    non_none_args, has_none = get_union_args(typ)
    if has_none and len(non_none_args) == 1:
        typ = non_none_args[0]
        is_ok = True if value is None else isinstance(value, typ)
        return is_ok, non_none_args[0]
    return False, typ


@lru_cache(maxsize=1)
def _is_flat_tuple_type(typ: Type, args):
    return (
        # Container is a tuple and contains inner annotation
        get_origin(typ) == tuple
        # The inner annotation is a BasicType
        and args[0] in BASIC_TYPES
        # the follow-up annotations are of the same type, or an Ellipsis
        and all(subt in (args[0], Ellipsis) for subt in args)
    )


def is_parsable_type(typ: Type, value: Any | None = None) -> bool:
    """Check if the given type is parsable.

    Args:
        typ (Type): The type to check.
        value (Any | None, optional): The value to check against. Defaults to None.

    Returns:
        bool: True if the type is parsable, False otherwise.
    """
    # Check for multi-type unions
    union_args, has_none = get_union_args(typ)
    if len(union_args) > 1:
        # Multi-type union: check if value matches one of the types
        if value is None:
            # For None value, just check if the union allows None and has at least one parsable type
            return has_none and any(is_parsable_type_single(t, None) or is_bare_container(t) for t in union_args)
        else:
            # Check if value matches any of the union types
            return any(is_parsable_type_single(t, value) for t in union_args)
    
    is_optional, inner_type = is_optional_single_type(typ, value)
    if is_optional:
        return is_parsable_type_single(inner_type, value)
    else:
        return is_parsable_type_single(typ, value)


def is_bare_container(typ: Type) -> bool:
    """Check if type is a bare container like list or dict without type parameters."""
    return typ in (list, dict)


def is_parsable_type_single(typ: Type, value: Any | None = None) -> bool:
    """Check if the given type is parsable.

    Args:
        typ (Type): The type to check.
        value (Any | None, optional): The value to check against. Defaults to None.

    Returns:
        bool: True if the type is parsable, False otherwise.
    """
    # Handle bare containers (list, dict) when checking values
    if value is not None:
        if typ == list and isinstance(value, list):
            return True
        if typ == dict and isinstance(value, dict):
            return True
    
    return any(
        (
            is_int_type(typ, value),
            is_float_type(typ, value),
            is_bool_type(typ, value),
            is_str_type(typ, value),
            is_flat_tuple_type(typ, value),
            is_flat_list_type(typ, value),
            is_dict_type(typ, value),
            is_path_type(typ, value),
        )
    )


def get_flat_tuple_inner_type(typ: Type[tuple]) -> tuple[Type, int]:
    """
    Get the inner type and length of a flat tuple.

    The length is -1 if the tuple has an ellipsis,
    indicating that it can have any number of elements.

    Args:
        typ (Type[tuple]): The type of the tuple.

    Returns:
        tuple[Type, int]: A tuple containing the inner type and length of the flat tuple.

    Raises:
        AssertionError: If the type is not a valid flat tuple type.

    """
    args = get_args(typ)
    assert len(args) > 0, "Tuple type must have at least one argument."
    basetype = get_args(typ)[0]
    assert basetype in BASIC_TYPES or is_flat_tuple_type(basetype), (
        "The inner type must be one of the basic types: "
        f"{BASIC_TYPES} or a flat tuple type."
    )
    if Ellipsis in args:
        assert len(args) == 2, "Ellipsis must be the second argument."
        return basetype, -1
    else:
        assert all(
            subt == basetype for subt in args
        ), "All inner types must be the same."
        return basetype, len(args)


def get_flat_list_inner_type(typ: Type[list]) -> Type:
    """
    Get the inner type of a flat list.

    Args:
        typ (Type[list]): The type of the list.

    Returns:
        Type: The inner type of the flat list.

    Raises:
        AssertionError: If the type is not a valid flat list type.

    """
    args = get_args(typ)
    assert len(args) > 0, "List type must have at least one argument."
    basetype = args[0]
    assert basetype in BASIC_TYPES, (
        "The inner type must be one of the basic types: "
        f"{BASIC_TYPES}."
    )
    return basetype
