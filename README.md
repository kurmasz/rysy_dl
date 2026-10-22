# Rysy

This library supports the creation and use of combinatorial circuits used for "ask-me-another" exercises. It includes:

* A command-line tool for generating a database of circuits.
* A library used by PrairieLearn to build questions.

## Installation

```
pip install rysy_dl
```

This installs the `rysy_dl` library along with the `rysy_dl` command-line tool.

## Terms
  * _graph_/ _digraph_: The raw graph that underlies a circuit. No gates have been assigned to nodes.
  * _netlist_: A `networkx` graph object with sources and gates assigned to nodes.
  * _circuit_: An instance of the `Circuit` class. The `Circuit` class is a "thin" class that 
    delegates to functions in various circuit-related modules. 

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
