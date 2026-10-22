# Rysy_dl

This library supports the creation and use of combinatorial circuits used for "ask-me-another" exercises. It includes:

* A command-line tool for generating a database of graphs or circuits.
* A Python library that tools like PrairieLearn can use to build questions.

Generating random circuits and graphs uniformly (i.e., in a way that gives all circuits/graphs an equal probability of 
being generated) is difficult. Furthermore, many randomly generated circuits make for poor exercises. `rysy_dl` breaks 
exercise generation into two phases:

* An _enumeration_ phase that builds a database of all directed acyclic graphs (DAG) on which reasonable circuits can be built. 
* A _gate assignment_ phase where gates are randomly assigned to the inner nodes of a DAG randomly selected from the database.

The number of possible DAGs (and, by extension, the number of possible circuits) grows very quickly. However, the number of DAGs 
suitable for use in creating combinatorial circuits exercises is manageable. The number of circuits is not. Therefore, our approach
is to build a database of DAGs offline, which is then made available to the exercise generator (e.g., PrairieLearn). The exercise
generator then chooses a DAG at random from the database, randomly assigns gates to the DAG, then verifies that the resulting 
circuit makes for a good exercise. Splitting the process into two phases like this keeps the run-time exercise generation (e.g., the time needed 
for PrairieLearn to generate an exercise when a student opens an assessment) reasonable while also keeping the size of the database reasonable.

## Terms
  * _graph_ / _digraph_: The raw graph that underlies a circuit. No gates have been assigned to nodes.
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

The circuit generator uses an external tool called [`nauty`](https://pallini.di.uniroma1.it/) to enumerate the underlying DAGs. You must 
install `nauty` in order to build your own database of graphs. (You don't need it just to use a previously-created database.)

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

## License

MIT -- see [LICENSE](LICENSE).
