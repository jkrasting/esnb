import sys
import os
import pytest

from esnb.engine import (
    is_url,
    open_source_notebook,
    clear_notebook_contents,
    write_notebook,
)

from importlib_resources import files
basic_catalog = str(files("esnb.templates") / "basic.ipynb")

def deep_getsizeof(obj, seen=None):
    """Recursively finds the total memory footprint of a Python object."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Avoid double-counting the same object
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(
            deep_getsizeof(k, seen) + deep_getsizeof(v, seen) for k, v in obj.items()
        )
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_getsizeof(i, seen) for i in obj)

    return size


def is_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        first_1024_bytes = file.read(1024).strip()
        if first_1024_bytes.startswith("{") or first_1024_bytes.startswith("["):
            is_json = True
        else:
            is_json = False

    return is_json


def is_html_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        first_1024_bytes = file.read(1024).strip()
        if (
            "<html" in first_1024_bytes.lower()
            or "<!doctype html" in first_1024_bytes.lower()
        ):
            is_html = True
        else:
            is_html = False

    return is_html


def test_is_url_1():
    string = "https://google.com"
    assert is_url(string) == True


def test_is_url_2():
    string = "/path/to/some/file"
    assert is_url(string) == False


def test_open_source_notebook_1():
    notebook_path = "https://raw.githubusercontent.com/fengzydy/CM5scripts/refs/heads/main/ghg/ghg_forcing.ipynb"
    _ = open_source_notebook(notebook_path)


def test_open_source_notebook_2():
    notebook_path = basic_catalog
    _ = open_source_notebook(notebook_path)

def test_open_source_notebook_3():
    notebook_path = "abcdef0123456"
    with pytest.raises(Exception):
        _ = open_source_notebook(notebook_path)

def test_clear_notebook_contents():
    notebook_path = "https://raw.githubusercontent.com/fengzydy/CM5scripts/refs/heads/main/ghg/ghg_forcing.ipynb"
    nb = open_source_notebook(notebook_path)
    cleared_nb = clear_notebook_contents(nb)
    assert deep_getsizeof(nb) > deep_getsizeof(cleared_nb)


def test_write_notebook_1():
    notebook_path = basic_catalog
    nb = open_source_notebook(notebook_path)
    output = "test.file"
    write_notebook(nb, output)
    assert is_json_file(output)
    os.remove(output)


def test_write_notebook_2():
    notebook_path = basic_catalog
    nb = open_source_notebook(notebook_path)
    output = "test.file"
    write_notebook(nb, output, fmt="html")
    assert is_html_file(output)
    os.remove(output)
