# Rysy_dl

This Python library supports the creation and use of combinatorial circuits used for "ask-me-another" exercises.[^ask-me-another] It includes:

* A command-line tool for generating a database of circuits
* filters to remove unsuitable circuits from consideration for use in exercises
* functions to randomly modify a circuit
* functions to generate a circuit's diagram, Boolean expression and/or truth table

It is difficult to randomly generate circuits that are suitable for practice problems. Simple generation 
techniques tend to overs-ample certain underlying structures. Such techniques also tend to produce "dumb"
circuits (e.g., a circuit with an AND gate that has both X and X' as inputs).

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
sources, and where all the gates have a reasonable fan-in and fan-out). The gate assignment is done "live" when the student 
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

## License

MIT -- see [LICENSE](LICENSE).

[^ask-me-another]: An "ask-me-another" exercise is a practice question that students can repeat as often as they like, 
    getting a new, randomly generated variant (here, a different circuit) each time.
