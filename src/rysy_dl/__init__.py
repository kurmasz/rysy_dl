"""Package marker for the rysy_dl module."""

from .circuit import Circuit
from .circuit_database import CircuitDatabase
from .circuit_layout import assign_source_labels

def _default_circuit_factory(entry):
    assign_source_labels(entry['graph'])
    return Circuit.from_entry(entry)
CircuitDatabase.default_circuit_factory = _default_circuit_factory
