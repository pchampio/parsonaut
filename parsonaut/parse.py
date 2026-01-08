import json
import re
import sys
from argparse import SUPPRESS, Action, RawDescriptionHelpFormatter
from argparse import ArgumentParser as _ArgumentParser
from argparse import ArgumentTypeError
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from parsonaut.lazy import TYPE_NAME, Choices, Lazy
from parsonaut.typecheck import (
    Missing,
    get_flat_list_inner_type,
    get_flat_tuple_inner_type,
    get_literal_choices,
    get_union_args,
    is_bool_type,
    is_dict_type,
    is_flat_list_type,
    is_flat_tuple_type,
    is_float_type,
    is_int_type,
    is_literal_type,
    is_optional_single_type,
    is_path_type,
    is_str_type,
)

BOOL_TRUE_FLAGS = ("yes", "true", "t", "y", "1")
BOOL_FALSE_FLAGS = ("no", "false", "f", "n", "0")


class InlineHelpFormatter(RawDescriptionHelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=40, width=100)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            (metavar,) = self._metavar_formatter(action, default)(1)
            return metavar
        else:
            parts = []
            if action.nargs == 0:
                parts.extend(action.option_strings)
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append(f'{option_string} {args_string}')
            return ', '.join(parts)


class ArgumentParser(_ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.lazy_roots = list()
        self.args = dict()
        self.aliases = dict()
        self.choices = defaultdict(list)
        self.choices_defaults = dict()
        # We allow add_options without dest if it is the only source of
        # args.
        self._lazy_without_dest = False
        # Store class references for lazy field_help lookup
        self._field_help_classes = {}

        kwargs.setdefault('formatter_class', InlineHelpFormatter)
        super().__init__(*args, **kwargs)

    def add_options(self, lzy: Lazy, dest: str | None = None):
        assert (
            not self._lazy_without_dest
        ), "Cannot add lazy options without a destination name if other lazy was already added."
        if dest is not None:
            assert "." not in dest
            assert dest not in self.lazy_roots
            self.lazy_roots.append(dest)
        else:
            assert set(self.args.keys()) == {
                "--help"
            }, "Cannot add lazy options without a destination name if other args are present"

        prefix = f"{dest}." if dest is not None else ""
        self._add_options(lzy, prefix=prefix)
        if dest is None:
            self._lazy_without_dest = True

    def _add_options(self, lzy: Lazy, prefix: str = ""):
        self.add_argument(f"--{prefix}_class", default=lzy.cls, help=SUPPRESS)
        
        # Store class reference for lazy field_help lookup during parse_args
        self._field_help_classes[prefix] = lzy.cls
        
        for k, (typ, value) in sorted(lzy.signature.items()):
            if Lazy.is_lazy_type(typ):
                if isinstance(value, Choices):
                    self.add_argument(
                        f"--{prefix}{k}",
                        type=str,
                        choices=[e.name for e in type(value)],
                        default=value.name,
                    )
                    self.choices_defaults[f"{prefix}{k}"] = value.name
                    for e in type(value):
                        self.choices[f"{prefix}{k}"].append(e.name)
                        # Add a [] marker to highlight choice values.
                        # We use the marks later to trim the choices.
                        self._add_options(e, prefix=f"{prefix}{k}.[{e.name}].")
                else:
                    self._add_options(value, prefix=f"{prefix}{k}.")
            else:
                # Defer help_text - pass field_key for later lookup
                self.add_option(f"{prefix}{k}", value, typ, help_text=None, field_key=(prefix, k))

    def add_option(self, name, value, typ, help_text=None, field_key=None):
        assert isinstance(name, str)
        assert not self._lazy_without_dest

        name = f"--{name}"
        check_val = value if value is not Missing else None
        required = False
        
        # Store field_key for lazy help lookup
        if field_key is not None:
            if not hasattr(self, '_field_keys'):
                self._field_keys = {}
            self._field_keys[name] = field_key
        
        def add_default_to_help(help_text, default_val):
            """Helper to append default value to help text."""
            if default_val is None or default_val is Missing:
                return help_text
            # Format default value nicely
            default_str = f"(default: {default_val})"
            if help_text:
                return f"{help_text} {default_str}"
            return default_str

        # Check for multi-type unions (e.g., list | str | None)
        union_args, has_none = get_union_args(typ)
        if len(union_args) > 1:
            # Multi-type union: try str2union converter
            default_val = value if value is not Missing else None
            kwargs = {
                "type": str2union(*union_args, has_none=has_none),
                "default": default_val,
                "metavar": "value",
                "required": required,
                "nargs": "?" if has_none else None,
            }
            final_help = add_default_to_help(help_text, default_val)
            if final_help:
                kwargs["help"] = final_help
            self.add_argument(name, **kwargs)
            return

        # Optional args not supported for tuples yet
        is_optional, typ = is_optional_single_type(typ, None)

        # Literal (must come before str check since Literal values are often strings)
        if is_literal_type(typ):
            choices = get_literal_choices(typ)
            default_val = value if value is not Missing else None
            kwargs = {
                "type": type(choices[0]) if choices else str,
                "choices": list(choices),
                "default": default_val,
                "required": required,
                "nargs": "?" if is_optional else None,
            }
            if help_text:
                # Append default value to help text
                if default_val is not None:
                    kwargs["help"] = f"{help_text} (default: {default_val})"
                else:
                    kwargs["help"] = help_text
            elif default_val is not None:
                kwargs["help"] = f"(default: {default_val})"
            self.add_argument(name, **kwargs)
        # bool
        elif is_bool_type(typ):
            default_val = value if value is not Missing else None
            kwargs = {
                "type": str2bool,
                "default": default_val,
                "metavar": f"{typ.__name__}",
                "required": required,
                "nargs": "?" if is_optional else None,
            }
            final_help = add_default_to_help(help_text, default_val)
            if final_help:
                kwargs["help"] = final_help
            self.add_argument(name, **kwargs)
        # dict (JSON)
        elif is_dict_type(typ, check_val):
            default_val = value if value is not Missing else None
            kwargs = {
                "type": str2json,
                "default": default_val,
                "metavar": "json",
                "required": required,
                "nargs": "?" if is_optional else None,
            }
            final_help = add_default_to_help(help_text, default_val)
            if final_help:
                kwargs["help"] = final_help
            self.add_argument(name, **kwargs)
        # Path
        elif is_path_type(typ, check_val):
            default_val = value if value is not Missing else None
            kwargs = {
                "type": str2path,
                "default": default_val,
                "metavar": "path",
                "required": required,
                "nargs": "?" if is_optional else None,
            }
            final_help = add_default_to_help(help_text, default_val)
            if final_help:
                kwargs["help"] = final_help
            self.add_argument(name, **kwargs)
        # int | float | str
        elif (
            is_int_type(typ, check_val)
            or is_str_type(typ, check_val)
            or is_float_type(typ, check_val)
        ):
            kwargs = {
                "type": typ,
                "default": value,
                "metavar": f"{typ.__name__}",
                "required": required,
                "nargs": "?" if is_optional else None,
            }
            final_help = add_default_to_help(help_text, value)
            if final_help:
                kwargs["help"] = final_help
            self.add_argument(name, **kwargs)
        # tuple[bool | int | float |str , ...]
        elif is_flat_tuple_type(typ, check_val):
            subtyp, nitems = get_flat_tuple_inner_type(typ)
            nargs = "*" if nitems == -1 else nitems
            if nargs == "*":
                metavar = f"{subtyp.__name__},"
            else:
                metavar = f"{subtyp.__name__}"

            default_val = tuple(value) if value is not Missing else None
            kwargs = {
                "nargs": nargs,
                "metavar": metavar,
                "type": subtyp if subtyp != bool else str2bool,
                "default": default_val,
                "required": required,
                "action": collect_as(tuple),
            }
            final_help = add_default_to_help(help_text, default_val)
            if final_help:
                kwargs["help"] = final_help
            self.add_argument(name, **kwargs)
        # list[bool | int | float | str]
        elif is_flat_list_type(typ, check_val):
            default_val = value if value is not Missing else None
            kwargs = {
                "type": str2json_list,
                "default": default_val,
                "metavar": "json",
                "required": required,
                "nargs": "?" if is_optional else None,
            }
            final_help = add_default_to_help(help_text, default_val)
            if final_help:
                kwargs["help"] = final_help
            self.add_argument(name, **kwargs)
        else:
            raise

    def add_argument(self, *name_or_flags, **kwargs):
        assert not self._lazy_without_dest
        if len(name_or_flags) == 2:
            alias, name = name_or_flags
            assert alias not in self.aliases
            self.aliases[name] = alias
        elif len(name_or_flags) == 1:
            (name,) = name_or_flags

        self.args[name] = kwargs

    def _inject_field_help(self):
        """Lazily inject field help text from class __field_help__ attributes."""
        field_keys = getattr(self, '_field_keys', {})
        
        for arg_name, (prefix, field_name) in field_keys.items():
            cls = self._field_help_classes.get(prefix)
            if cls is None:
                continue
            field_help = getattr(cls, '__field_help__', {})
            help_text = field_help.get(field_name)
            if help_text and arg_name in self.args:
                kwargs = self.args[arg_name]
                # Merge help text with existing default info
                existing_help = kwargs.get('help')
                if existing_help:
                    # existing_help is like "(default: value)", prepend the description
                    kwargs['help'] = f"{help_text} {existing_help}"
                else:
                    kwargs['help'] = help_text

    def parse_args(self, args=None):  # noqa: C901
        from collections import defaultdict

        args = sys.argv[1:] if args is None else args

        # Choices trimming:
        # Check if user provided a specific value for a choice and trim the other options
        choices = sorted(
            self.choices.items(), key=lambda x: len(x[0].split(".")), reverse=True
        )
        for k, v in choices:
            # If yes, read it from the command line
            prefix = re.sub(r"\[.*?\]\.", "", f"--{k}")
            if prefix in args:
                position = args.index(prefix)
                assert (
                    len(args) > position + 1
                ), f"Expected a value after the choice {prefix}"
                val = args[args.index(prefix) + 1]
                assert (
                    val in v
                ), f"error: argument {prefix}: invalid choice '{val}' (choose from {', '.join(v)})"
            else:
                val = self.choices_defaults[k]

            # We remove choice options:
            # - not selected by the user
            # - not default
            for choice in v:
                if choice != val:
                    remove_prefix = f"--{k}.[{choice}]"
                    for key in self.args.copy():
                        if key.startswith(remove_prefix):
                            del self.args[key]

        for k in self.args.copy():
            if "[" in k:
                kk = re.sub(r"\[.*?\]\.", "", k)
                self.args[kk] = self.args[k]
                del self.args[k]

        # Inject help text only when --help is requested (lazy extraction)
        if '--help' in args or '-h' in args:
            self._inject_field_help()

        for name in self.args:
            if name in self.aliases:
                arg = (self.aliases[name], name)
            else:
                arg = (name,)
            super().add_argument(*arg, **self.args[name])

        args = super().parse_args(args)

        # we can build the Lazy objects from the recursive dicts
        args_dict = vars(args)
        args_grouped = defaultdict(dict)
        for k, v in args_dict.items():
            # Do not add choices names
            if k in self.choices:
                continue
            if not k.startswith(tuple(self.lazy_roots)):
                args_grouped[k] = v
            else:
                root = [root for root in self.lazy_roots if k.startswith(root)]
                assert len(root) == 1
                root = root[0]
                args_grouped[root][k.split(f"{root}.", 1)[-1]] = v

        args_grouped = dict(args_grouped)
        if self.lazy_roots:
            for root in self.lazy_roots:
                args_grouped[root] = Lazy.from_dict(args_grouped[root])

            return SimpleNamespace(**args_grouped)
        elif TYPE_NAME in args_grouped:
            return Lazy.from_dict(args_grouped)
        else:
            return SimpleNamespace(**args_grouped)


def collect_as(coll_type):
    class Collect_as(Action):
        def __call__(self, parser, namespace, values, options_string=None):
            setattr(namespace, self.dest, coll_type(values))

    return Collect_as


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in BOOL_TRUE_FLAGS:
        return True
    elif v.lower() in BOOL_FALSE_FLAGS:
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def str2json(v):
    if isinstance(v, dict):
        return v
    try:
        result = json.loads(v)
        if not isinstance(result, dict):
            raise ArgumentTypeError("JSON value must be a dict/object.")
        return result
    except json.JSONDecodeError as e:
        raise ArgumentTypeError(f"Invalid JSON: {e}")


def str2path(v):
    if isinstance(v, Path):
        return v
    return Path(v)


def str2json_list(v):
    if isinstance(v, list):
        return v
    try:
        result = json.loads(v)
        if not isinstance(result, list):
            raise ArgumentTypeError("JSON value must be a list/array.")
        return result
    except json.JSONDecodeError as e:
        raise ArgumentTypeError(f"Invalid JSON: {e}")


def str2union(*types, has_none=False):
    """Create a converter for union types like list | str."""
    def converter(v):
        # If the union includes None and we get an empty string, return None
        if has_none and v == '':
            return None
        
        # Try each type in order
        for typ in types:
            try:
                if typ == str:
                    return v
                elif typ == int:
                    return int(v)
                elif typ == float:
                    return float(v)
                elif typ == bool:
                    return str2bool(v)
                elif typ == list or (hasattr(typ, '__origin__') and typ.__origin__ == list):
                    return str2json_list(v)
                elif typ == dict or (hasattr(typ, '__origin__') and typ.__origin__ == dict):
                    return str2json(v)
                elif typ == Path:
                    return str2path(v)
            except (ValueError, ArgumentTypeError, json.JSONDecodeError):
                continue
        raise ArgumentTypeError(f"Value must be one of {[t.__name__ if hasattr(t, '__name__') else str(t) for t in types]}")
    return converter
