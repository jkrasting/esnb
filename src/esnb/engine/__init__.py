"""Module for notebook engine"""

import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import jupyter_client
import nbformat
import requests
from nbclient import NotebookClient
from nbconvert import HTMLExporter

__all__ = [
    "clear_notebook_contents",
    "identify_current_kernel_name",
    "is_url",
    "open_source_notebook",
    "run_notebook",
    "write_notebook",
]


def clear_notebook_contents(nb):
    result = copy.deepcopy(nb)
    for cell in result.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return result


def identify_current_kernel_name():
    python_exec = sys.executable
    existing_kernels = jupyter_client.kernelspec.find_kernel_specs()

    kernelspecs = [
        (os.path.join(path, "kernel.json"), name)
        for name, path in existing_kernels.items()
    ]
    kernelspecs = [x for x in kernelspecs if os.path.exists(x[0])]

    kernel_name = None

    for kernel in kernelspecs:
        try:
            with open(kernel[0]) as f:
                spec = json.load(f)
                if spec["argv"][0] == python_exec:
                    print(f"We found a match: {kernel[0]}")
                    kernel_name = kernel[1]
        except Exception:
            continue

    if kernel_name is None:
        raise RuntimeError("Current kernel spec must be registered.")
    else:
        print(f"Using kernel: {kernel_name}")

    return kernel_name


def is_url(url):
    """Check if a string is a valid HTTPS URL."""
    return isinstance(url, str) and url.lower().startswith("https://")


def open_source_notebook(notebook_path, version=4):
    if is_url(notebook_path):
        print(f"Opening notebook from web location: {notebook_path}")
        response = requests.get(notebook_path)
        response.raise_for_status()
        nb = nbformat.reads(response.text, as_version=version)
    elif Path(notebook_path).exists():
        print(f"Opening notebook from file location: {notebook_path}")
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=version)
    else:
        raise ValueError(f"Unable to load source notebook: {notebook_path}")

    return nb


def run_notebook(notebook_path, output_dir):
    current_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    print(f"Created tempdir: {temp_dir}")
    os.chdir(temp_dir)

    output_dir = Path(output_dir)
    notebook_path = Path(notebook_path)
    file_stem = notebook_path.stem

    nb = open_source_notebook(notebook_path)
    nb = clear_notebook_contents(nb)
    kernel_name = identify_current_kernel_name()

    client = NotebookClient(
        nb, timeout=600, kernel_name=kernel_name, allow_errors=False
    )
    _ = client.execute()

    write_notebook(nb, str(output_dir / file_stem) + ".html", fmt="html")
    write_notebook(nb, str(output_dir / file_stem) + ".ipynb", fmt="ipynb")

    extra_files = os.listdir(temp_dir)
    for fname in extra_files:
        print(f"Copying extra file: {fname}")
        shutil.copy2(os.path.join(temp_dir, fname), output_dir)

    os.chdir(current_dir)
    shutil.rmtree(temp_dir)


def write_notebook(nb, output_name, fmt="ipynb"):
    output_name = Path(output_name)
    output_name = output_name.resolve()
    dirpath = output_name.parent
    os.makedirs(dirpath, exist_ok=True)

    if fmt == "ipynb":
        with open(output_name, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

    elif fmt == "html":
        html_exporter = HTMLExporter()
        (body, resources) = html_exporter.from_notebook_node(nb)
        with open(output_name, "w", encoding="utf-8") as f:
            f.write(body)

    else:
        raise ValueError(f"Unknown output format: {fmt}")

    print(f"File written: {output_name}")
