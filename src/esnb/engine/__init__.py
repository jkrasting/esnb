"""Module for notebook engine"""

import copy
import filecmp
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import jupyter_client
import nbformat
import requests
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from esnb.sites.gfdl import slurm_stub

__all__ = [
    "activate_conda_env",
    "canopy_launcher",
    "clear_notebook_contents",
    "create_script",
    "identify_current_kernel_name",
    "is_url",
    "is_python_conda",
    "open_source_notebook",
    "run_notebook",
    "write_notebook",
]

logger = logging.getLogger(__name__)


def activate_conda_env(env_path: str):
    bash_cmd = (
        # load Lmod (if present) and the conda module, but don't fail if absent
        'source "$MODULESHOME/init/bash" 2>/dev/null || true; '
        "module load conda 2>/dev/null || true; "
        # enable `conda activate`
        'source "$(conda info --base)/etc/profile.d/conda.sh"; '
        # activate and print environment as NUL-separated entries
        f'conda activate "{env_path}"; env -0'
    )
    # use a login shell (-l) to mimic batch environment closely
    proc = subprocess.Popen(
        ["bash", "-lc", bash_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Activation failed: {err.decode(errors='ignore')}")
    # Parse NUL-separated KEY=VALUE entries robustly
    updates = {}
    for entry in out.split(b"\x00"):
        if not entry:
            continue
        key, sep, value = entry.partition(b"=")
        if not sep:
            # skip junk lines without '='
            continue
        updates[key.decode()] = value.decode()
    os.environ.update(updates)


def canopy_launcher(run_settings, case_settings=None, verbose=False):
    conda_env_root = run_settings["conda_env_root"]
    notebook_path = run_settings["notebook_path"]
    outdir = run_settings["outdir"]
    scripts_dir = run_settings["scripts_dir"]

    scheduler = slurm_stub

    script_path = create_script(
        notebook_path,
        outdir,
        conda_prefix=conda_env_root,
        scripts_dir=scripts_dir,
        scheduler=scheduler,
        case_settings=case_settings,
        verbose=verbose,
    )

    return script_path


def clear_notebook_contents(nb):
    result = copy.deepcopy(nb)
    for cell in result.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return result


def create_script(
    notebook_path,
    output_path,
    conda_prefix=None,
    scripts_dir=None,
    scheduler=None,
    verbose=False,
    case_settings=None,
):
    notebook_path = Path(notebook_path)
    notebook_name = notebook_path.stem

    if case_settings is not None:
        assert isinstance(case_settings, dict), "case_settings must be a dict object"
        case_settings = dict_to_key_value_string(str(case_settings))

    if conda_prefix is None:
        interpreter = sys.executable
        conda_prefix = (
            os.environ["CONDA_PREFIX"] if is_python_conda(interpreter) else None
        )
    else:
        interpreter = Path(f"{conda_prefix}/bin/python")

    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as script:
        script_path = script.name

        try:
            # print hashbang w/ interpreter path
            script.write(f"#!{interpreter}\n")

            # insert scehduler stub, optional
            if scheduler is not None:
                directives = scheduler(jobname=notebook_name, outputdir=output_path)
                script.write(directives + "\n\n")

            # import some basics
            script.write("import os\n")
            script.write("import sys\n")
            script.write("import subprocess\n\n")

            if conda_prefix is not None:
                logger.debug(f"Telling script to use conda environment: {conda_prefix}")
                script.write("# initialize conda \n")
                script.write("moduleshome = os.environ['MODULESHOME']\n")
                script.write("exec(open(f'{moduleshome}/init/python.py').read())\n")
                script.write("module('load conda')\n")
                script.write("module('list')\n\n")

                script.write("# activate conda environment \n")
                script.write(f"os.environ['PROJ_LIB'] = '{conda_prefix}/share/proj'\n")
                script.write(f"os.environ['PROJ_DATA'] = '{conda_prefix}/share/proj'\n")
                script.write("from esnb.engine import activate_conda_env\n")
                script.write(f"activate_conda_env('{conda_prefix}')\n\n")

            if case_settings is not None:
                script.write("# case settings override\n")
                script.write(f"os.environ['ESNB_CASE_DATA'] = \"{case_settings}\"\n\n")

            script.write("# run notebook \n")
            script.write("from esnb.engine import run_notebook\n")
            script.write(f"run_notebook('{notebook_path}', '{output_path}')\n\n")

            script.write("sys.exit()")

        except Exception as exc:
            logger.debug(f"Removing temp file: {script_path}")
            os.remove(script_path)
            raise exc

    if scripts_dir is not None:
        scripts_dir = Path(scripts_dir)
        if not scripts_dir.exists():
            logger.debug(f"Making scripts dir: {scripts_dir}")
            os.makedirs(scripts_dir)
        new_script_path = scripts_dir / f"esnb_{notebook_name}.py"
        shutil.move(script_path, new_script_path)
        script_path = new_script_path

    permissions = 0o755
    logger.debug(f"Changing permissions: {script_path}")
    os.chmod(script_path, permissions)

    logger.debug(f"Finished writing to {script_path}\n\n")

    if verbose:
        with open(script_path, "r") as f:
            print(f.read())

    return str(script_path)


def dict_to_key_value_string(text):
    allowed_punctuation = r":\[\]\(\)\/\-\,\_\."
    pattern = rf"[^a-zA-Z0-9{allowed_punctuation}]"
    return re.sub(pattern, "", text)


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


def is_python_conda(interpreter):
    interpreter = (
        Path(interpreter) if not isinstance(interpreter, Path) else interpreter
    )
    if "CONDA_PREFIX" in os.environ.keys():
        conda_prefix = Path(os.environ["CONDA_PREFIX"])
        conda_interpreter = Path(conda_prefix / "bin" / interpreter.name)
        result = filecmp.cmp(conda_interpreter, interpreter)
    else:
        result = False
    return result


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

    import nest_asyncio

    nest_asyncio.apply()
    print("doing async io")

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name=kernel_name,
        allow_errors=True,
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
