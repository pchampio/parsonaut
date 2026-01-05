from pathlib import Path

import pytest

from parsonaut import Parsable
from parsonaut.lazy import Missing
from parsonaut.parse import ArgumentParser, str2json, str2path


def test_str2json_valid():
    assert str2json('{"a": 1, "b": 2}') == {"a": 1, "b": 2}
    assert str2json('{"nested": {"key": "value"}}') == {"nested": {"key": "value"}}
    assert str2json("{}") == {}


def test_str2json_already_dict():
    d = {"a": 1}
    assert str2json(d) == d


def test_str2json_invalid():
    with pytest.raises(Exception):
        str2json('{"invalid json')
    with pytest.raises(Exception):
        str2json("[1, 2, 3]")  # list, not dict


def test_str2path():
    assert str2path("/tmp/foo") == Path("/tmp/foo")
    assert str2path("relative/path") == Path("relative/path")


def test_str2path_already_path():
    p = Path("/tmp")
    assert str2path(p) == p


def test_ArgumentParser_add_option_dict():
    parser = ArgumentParser()
    parser.add_option("config", {"a": 1}, dict)
    args = parser.parse_args([])
    assert args.config == {"a": 1}

    parser = ArgumentParser()
    parser.add_option("config", Missing, dict)
    args = parser.parse_args(["--config", '{"b": 2}'])
    assert args.config == {"b": 2}


def test_ArgumentParser_add_option_path():
    parser = ArgumentParser()
    parser.add_option("output", Path("/tmp"), Path)
    args = parser.parse_args([])
    assert args.output == Path("/tmp")

    parser = ArgumentParser()
    parser.add_option("output", Missing, Path)
    args = parser.parse_args(["--output", "/var/log"])
    assert args.output == Path("/var/log")


class WithDict(Parsable):
    def __init__(
        self,
        config: dict = {"default": "value"},
    ) -> None:
        self.config = config


class WithPath(Parsable):
    def __init__(
        self,
        output_dir: Path = Path("/tmp"),
    ) -> None:
        self.output_dir = output_dir


class WithBoth(Parsable):
    def __init__(
        self,
        config: dict = {"key": "value"},
        output_dir: Path = Path("/tmp"),
        name: str = "test",
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.name = name


def test_parsable_with_dict():
    parser = ArgumentParser()
    parser.add_options(WithDict.as_lazy())
    args = parser.parse_args([])
    assert args == WithDict.as_lazy(config={"default": "value"})

    parser = ArgumentParser()
    parser.add_options(WithDict.as_lazy())
    args = parser.parse_args(["--config", '{"custom": "data"}'])
    assert args == WithDict.as_lazy(config={"custom": "data"})


def test_parsable_with_path():
    parser = ArgumentParser()
    parser.add_options(WithPath.as_lazy())
    args = parser.parse_args([])
    assert args == WithPath.as_lazy(output_dir=Path("/tmp"))

    parser = ArgumentParser()
    parser.add_options(WithPath.as_lazy())
    args = parser.parse_args(["--output_dir", "/custom/path"])
    assert args == WithPath.as_lazy(output_dir=Path("/custom/path"))


def test_parsable_with_both():
    parser = ArgumentParser()
    parser.add_options(WithBoth.as_lazy())
    args = parser.parse_args([])
    assert args == WithBoth.as_lazy(
        config={"key": "value"}, output_dir=Path("/tmp"), name="test"
    )

    parser = ArgumentParser()
    parser.add_options(WithBoth.as_lazy())
    args = parser.parse_args(
        [
            "--config",
            '{"a": 1}',
            "--output_dir",
            "/my/path",
            "--name",
            "mytest",
        ]
    )
    assert args == WithBoth.as_lazy(
        config={"a": 1}, output_dir=Path("/my/path"), name="mytest"
    )


def test_parsable_optional_dict():
    class WithOptionalDict(Parsable):
        def __init__(
            self,
            config: dict | None = None,
        ) -> None:
            self.config = config

    parser = ArgumentParser()
    parser.add_options(WithOptionalDict.as_lazy())
    args = parser.parse_args([])
    assert args == WithOptionalDict.as_lazy(config=None)

    parser = ArgumentParser()
    parser.add_options(WithOptionalDict.as_lazy())
    args = parser.parse_args(["--config", '{"x": 1}'])
    assert args == WithOptionalDict.as_lazy(config={"x": 1})

    parser = ArgumentParser()
    parser.add_options(WithOptionalDict.as_lazy())
    args = parser.parse_args(["--config"])
    assert args == WithOptionalDict.as_lazy(config=None)


def test_parsable_optional_path():
    class WithOptionalPath(Parsable):
        def __init__(
            self,
            output: Path | None = None,
        ) -> None:
            self.output = output

    parser = ArgumentParser()
    parser.add_options(WithOptionalPath.as_lazy())
    args = parser.parse_args([])
    assert args == WithOptionalPath.as_lazy(output=None)

    parser = ArgumentParser()
    parser.add_options(WithOptionalPath.as_lazy())
    args = parser.parse_args(["--output", "/some/path"])
    assert args == WithOptionalPath.as_lazy(output=Path("/some/path"))

    parser = ArgumentParser()
    parser.add_options(WithOptionalPath.as_lazy())
    args = parser.parse_args(["--output"])
    assert args == WithOptionalPath.as_lazy(output=None)
