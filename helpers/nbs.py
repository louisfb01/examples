import glob
import io
import json
import os
import sys
import time
from pathlib import Path

from invoke import task
import nbformat
from fastcore.basics import num_cpus
from fastcore.utils import parallel
from nbconvert.preprocessors import ExecutePreprocessor
from nbdev.clean import clean_nb


TEST_PATH = Path("colabs")


def is_nb(fname):
    "filter files that are notebooks"
    return (
        (fname.suffix == ".ipynb")
        and (not fname.name.startswith("_"))
        and (not "checkpoint" in str(fname))
    )

TEST_NBS = [f for f in list(TEST_PATH.glob("**/*.ipynb")) if is_nb(f)]

def read_nb(fname):
    "Read the notebook in `fname`."
    with open(Path(fname), "r", encoding="utf8") as f:
        return nbformat.reads(f.read(), as_version=4)


def test_one(fname, verbose=True):
    "Run nb `fname` and timeit, recover exception"
    print(f"testing {fname}")
    start = time.time()
    try:
        notebook = read_nb(fname)
        processor = ExecutePreprocessor(timeout=600, kernel_name="python3")
        pnb = nbformat.from_dict(notebook)
        processor.preprocess(pnb)
        return True, time.time() - start
    except Exception as e:
        if verbose:
            print(f"\nError in executing {fname}\n{e}")
        return False, time.time() - start


def print_output(notebook):
    "Print `notebook` in stdout for git things"
    output_stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    x = json.dumps(notebook, sort_keys=True, indent=1, ensure_ascii=False)
    output_stream.write(x)
    output_stream.write("\n")
    output_stream.flush()


def clean_one(fname, clear_all=False, disp=False):
    "Clean notebooks outputs"
    if not str(fname).endswith(".ipynb"):
        return
    notebook = json.load(open(fname, "r", encoding="utf-8"))
    clean_nb(notebook, clear_all=clear_all)
    if disp:
        print_output(notebook)
    else:
        x = json.dumps(notebook, sort_keys=True, indent=1, ensure_ascii=False)
        with io.open(fname, "w", encoding="utf-8") as f:
            f.write(x)
            f.write("\n")


def _clean(fname=None, clear_all=True, disp=False):
    "Strip all notebooks from meta"
    print(
        "\n---------------------------------------------------------------\nStriping notebooks"
    )
    files = TEST_NBS if fname is None else glob.glob(fname)
    for f in files:
        print(f"Striping: {f}")
        clean_one(f, clear_all, disp)


def _test(fname=None, n_workers=None, verbose=True, timing=False, pause=0.5):
    """Test in parallel the notebooks matching `fname`
    fname: "A notebook name or glob to convert" = None,
    n_workers: "Number of workers to use" = None,
    verbose: "Print errors along the way" = True,
    timing: "Timing each notebook to see the ones are slow" = False,
    pause: "Pause time (in secs) between notebooks to avoid race conditions" = 0.5,"""
    if fname is None:
        files = TEST_NBS
    else:
        files = glob.glob(fname)
    files = [Path(f).absolute() for f in sorted(files)]
    if n_workers is None:
        n_workers = 0 if len(files) == 1 else min(num_cpus(), 8)
    # make sure we are inside the notebook folder of the project
    os.chdir(TEST_PATH)
    # results = parallel(
    #     test_one, files, verbose=verbose, n_workers=n_workers, pause=pause
    # )
    for nb in files:
        test_one(nb, verbose=verbose)
        time.sleep(pause)
    passed, times = [r[0] for r in results], [r[1] for r in results]
    if all(passed):
        print("All tests are passing!")
    else:
        msg = "The following notebooks failed:\n"
        raise Exception(
            msg + "\n".join([f.name for p, f in zip(passed, files) if not p])
        )
    if timing:
        for i, t in sorted(enumerate(times), key=lambda o: o[1], reverse=True):
            print(f"Notebook {files[i].name} took {int(t)} seconds")


@task
def test(c):
    "Test notebooks inside `examples` folder"
    _test()


@task
def clean(c):
    "Clean notebooks from useless metadata"
    _clean()