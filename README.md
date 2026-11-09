# Rysy_dl

This Python library supports the creation and use of combinatorial circuits used for "ask-me-another" exercises.[^ask-me-another] It includes:

* A command-line tool for generating a database of circuits
* filters to remove unsuitable circuits from consideration for use in exercises
* functions to randomly modify a circuit (thereby generating a new problem)
* functions to generate a circuit's diagram, Boolean expression and/or truth table

It is difficult to randomly generate circuits that are suitable for practice problems. Simple generation 
techniques tend to over-sample certain underlying structures. Such techniques also tend to produce "dumb"
circuits (e.g., a circuit with an AND gate that has both `X` and `X'` as inputs).

At a high level, our approach is to generate many circuits and filter the bad ones out. The challenge 
is to find a balance between 
   * The size of the circuit database and
   * The amount of work to be done "live" when generating a new exercise for a student.

The number of possible circuits grows exponentially. It is not possible to enumerate them all, 
much less store all of the good ones inside a content tool like PrairieLearn.  
At the same time, the process of repeatedly generating 
random circuits until one is judged to be good is too time and resource intensive to be done 
in real time as students work through an assignment. 

`rysy_dl` balances these two concerns by breaking circuit generation into two phases:

* An offline _enumeration_ phase that builds a database containing one example circuit for each possible underlying graph. 
* A _gate assignment_ phase where gates are randomly assigned to a circuit randomly selected from the database.

By starting with a circuit in the database, we know we have a suitable base (e.g., a circuit with a reasonable number of
sources, and where all the gates have a reasonable fan-in and fan-out). Gates are assigned "live" when the student 
opens an assessment; but, the expected number of random gate assignments generated before finding one that produces a "good" circuit is small.  
In addition, the process of (approximately) producing a uniform distribution of gate assignments is much more straightforward than the process
of randomly producing a random circuit structure (i.e., the underlying DAG) with a uniform distribution.

## Terms
  * _graph_ / _digraph_: The raw directed acyclic graph (DAG) that underlies a circuit. No gates have been assigned to nodes.
  * _netlist_: A `networkx` graph object with sources and gates assigned to nodes.
  * _circuit_: An instance of the `Circuit` class. The `Circuit` class is a "thin" class that 
    delegates to functions in various circuit-related modules. Tools like PrairieLearn will primarily interact
    with `Circuit` objects.


## Installation

`rysy_dl` requires Python 3.9 or greater.

```
pip install rysy_dl
```

This installs the `rysy_dl` library along with the `rysy_dl` command-line tool.

### `nauty`

The circuit generator ensures a uniform distribution of circuit structures by using an external tool called [`nauty`](https://pallini.di.uniroma1.it/) 
to enumerate the underlying DAGs. You must  install `nauty` in order to build your own database of graphs. (You don't need it just to use a previously-created database.)

* **macOS:** `brew install nauty`
* **Debian/Ubuntu:** `sudo apt install nauty` (this installs the programs as `nauty-geng` and `nauty-directg`; `rysy_dl` finds either name)
* **Arch:** `sudo pacman -S nauty`
* **conda (Linux/macOS):** `conda install -c conda-forge nauty`
* **Windows:** use [WSL](https://learn.microsoft.com/windows/wsl/install), then follow the Debian/Ubuntu instructions. (`rysy_dl` runs `nauty` 
  through `bash`, so it won't run natively on Windows.)
* **From source:** download the tarball from the [nauty home page](https://pallini.di.uniroma1.it/) (or its 
  [mirror](https://users.cecs.anu.edu.au/~bdm/nauty/)), then run `./configure && make` in the unpacked directory.

`rysy_dl` looks for the `geng` and `directg` programs (or `nauty-geng` and `nauty-directg`) in this order:

1. The directory given by `--nauty-dir`.
2. The directory named by the `NAUTY` environment variable.
3. Your `PATH`.

Package managers put the programs on your `PATH`, so no extra setup is needed. If you built `nauty` from source, set `NAUTY` 
to the build directory (e.g., `export NAUTY=$HOME/nauty2_9_3`), pass `--nauty-dir`, or add the build directory to your `PATH`.


## Setup and Testing

* To set up a development environment:
  ```
  python -m venv venv
  source venv/bin/activate
  pip install -e '.[dev]'
  ```
* To run the unit tests: `python -m pytest`


## Releasing

1. Update `version` in `pyproject.toml`.
2. Build the sdist and wheel: `python -m build`
3. Check them: `python -m twine check dist/*`
4. Upload to TestPyPI first: `python -m twine upload --repository testpypi dist/*`
5. Upload to PyPI: `python -m twine upload dist/*`

## PrairieLearn

PrairieLearn Questions use `rysy_dl` from their `server.py`. PrairieLearn puts the course's `serverFilesCourse` directory on
the Python path, so installing `rysy_dl` into `serverFilesCourse` makes the library available to all questions in the course.

PrairieLearn's Python environment already includes `rysy_dl`'s runtime dependencies (`networkx`, `matplotlib`,
`schemdraw`, and `Pillow`), so only `rysy_dl` itself must be installed (`--no-deps`). (`cairosvg` is used only by a
debugging tool and isn't needed in PrairieLearn.)

The precise installation method depends on whether the user is modifying `rysy_dl`, or simply using the published package.

All of the options below require [Docker](https://www.docker.com/). They run commands inside the
`prairielearn/prairielearn` image so that the result matches the environment questions run in.

### Installing a release from PyPI

For users who do not intend to modify `rysy_dl`, simply run

```
docker run --rm -v /path/to/course:/course prairielearn/prairielearn \
  uv pip install --target /course/serverFilesCourse --no-deps rysy_dl==<version>
```

then commit `serverFilesCourse/rysy_dl` and `serverFilesCourse/rysy_dl-<version>.dist-info`.
(`serverFilesCourse/bin/rysy_dl` and `serverFilesCourse/.lock` are not needed and can be deleted.)  `/path/to/course` is the absolute path name of the cloned git repo for the course in question.

To update, first delete `serverFilesCourse/rysy_dl` and `serverFilesCourse/rysy_dl-*.dist-info`, then run the docker
command again with the new version. (`uv` does not remove files that the new version no longer contains.)

If you have a copy of this repository, `scripts/pl_course.sh` does all of this for you:

```
scripts/pl_course.sh install /path/to/course rysy_dl==<version>
```

### Installing from source (e.g., a fork)

If you have cloned this repository and want to install from source into a course, 
run this command from the repository's root:

```
scripts/pl_course.sh install /path/to/course .
```

This builds a package from the clone (exactly as it would be built for PyPI) and installs it the same way
as a release, replacing any previous copy in the course. Note that it builds from the files in the
clone, including uncommitted changes. The source may also be a git URL, e.g.,
`git+https://github.com/<you>/rysy_dl.git@<tag>`.

### Testing a development copy with a local PrairieLearn

If you are developing `rysy_dl` and want to run it in "editable" mode together a local instance of PrairieLearn, run

```
scripts/pl_course.sh dev /path/to/course
```

This command launches the docker command to run PrairieLearn locally.  (See https://docs.prairielearn.com/installing/#running-instructions for details.)

When launching the docker container, the `pl_course` script mounts `src/rysy_dl` over the course's `serverFilesCourse/rysy_dl`,
hiding any "installed" copy and using your "live" copy instead. If the course doesn't have
`serverFilesCourse/rysy_dl`, Docker creates an empty directory there as a mount point; don't commit it.
If your changes don't show up, restart the container.

Note: `pl_course.sh` does not handle advanced configurations such as multiple courses, external graders, or workspaces. To use
those features in dev mode, add  `-v path_to_rysy_dl/src/rysy_dl:/course/serverFilesCourse/rysy_dl` to your launch script.

Using source directly rather than a built package, won't catch packaging mistakes
(e.g., a file missing from `[tool.setuptools.package-data]`). Before releasing, install into a test course
with `scripts/pl_course.sh install` at least once.

On Linux, files created by Docker in the course directory may be owned by `root`.

## Limitations

The generation algorithm skips circuits where the output gate has both as an input and a sub-circuit that depends on X. 

The generation algorithm iterates over all n-node DAGs (including disconnected DAGs), then adds an additional node whose inputs are all the nodes with out-degree 0. This ensures that there is only one output.  However, it means that a source X is only an input to this last gate if it is not an input to any other gate. This algorithm will generate, for example, `AB + C`, but will not generate `AB + A + C`

Addressing this limitation will add considerably to the runtime.

  * Option 1: Iterate over all DAGs of size `n` instead of `n - 1`, then filter out any DAGs with more than
    one sink. When enumerating circuits where the number of inputs plus the number of gates is 9, this approach would mean 3.2 billion DAGs instead of 20 million.
  * Option 2: Iterate over all possible combinations of gates with out-degree >= 0 to be used as inputs to
    the output gate. This approach would possibly generate isomorphic circuits that would then need to be detected and removed.

We decided to avoid this expense because most of the circuits generated in this manner would not be particularly "interesting" for exercises. Any circuit where X and a subcircuit that depends on X are inputs to the final gate can be trivially simplified.  For example `X + G(X, ...) = X + G(0, ....)`.

## License

MIT -- see [LICENSE](LICENSE).

[^ask-me-another]: An "ask-me-another" exercise is a practice question that students can repeat as often as they like, 
    getting a new, randomly generated variant (here, a different circuit) each time.
