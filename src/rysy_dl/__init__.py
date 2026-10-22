"""Package marker for the rysy_dl module."""

from .circuit import Circuit
from .circuit_database import CircuitDatabase
from .circuit_layout import assign_source_labels

_default_db = None

def _default_circuit_factory(entry):
    assign_source_labels(entry['graph'])
    return Circuit.from_entry(entry)
CircuitDatabase.default_circuit_factory = _default_circuit_factory
    
    

def get_default_db():
    """
    Return a process-wide `CircuitDatabase` connected to the packaged default database
    (`rysy_dl.db.DEFAULT_DB_PATH`), opening the connection on first call and reusing it on
    every subsequent call instead of at import time.

    :returns: The shared `CircuitDatabase` instance.
    """
    global _default_db
    if _default_db is None:
        _default_db = CircuitDatabase()
    return _default_db

