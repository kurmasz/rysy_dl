"""
circuit.py

Circuit: a thin, dot-friendly wrapper around the networkx DiGraph produced by
circuit_generator/circuit_database/boolean_expression_to_netlist. Every method
just delegates to the existing free functions in circuit_layout,
circuit_generator, circuit_analyzer, and circuit_serialization -- those stay
independent of this class so they can keep operating directly on a plain
graph wherever that's more convenient (e.g. circuit_generator's DAG
enumeration pipeline).

:copyright: 2026 Zachary Kurmas and Andrew Kurmas
"""

from . import boolean_expression_parser
from . import boolean_expression_to_netlist
from . import circuit_analyzer
from . import circuit_filters
from . import circuit_generator
from . import circuit_layout
from . import circuit_serialization


class Circuit:
    def __init__(self, netlist):
        assert netlist, "Parameter to Circuit.__init__ may not be None"
        self.netlist = netlist

    @classmethod
    def from_json_safe(cls, data):
        return cls(circuit_serialization.serializable_to_circuit(data))

    @classmethod
    def from_entry(cls, entry):
        """Build a Circuit from a database record (see `circuit_database.row_to_record`)."""
        return cls(entry['graph'])

    @classmethod
    def from_expression(cls, expression):
        """Build a Circuit directly from a boolean expression, e.g. "A*B + C'".

        The circuit's structure and every gate are fully determined by the
        expression, and source nodes are already labeled with their variable
        names -- unlike a randomly generated Circuit, `assign_gates_randomly`
        and `assign_source_labels` are neither needed nor applicable.
        """
        ast = boolean_expression_parser.parse_boolean_expression(expression)
        return cls(boolean_expression_to_netlist.build_netlist(ast))

    def to_json_safe(self):
        return circuit_serialization.circuit_to_serializable(self.netlist)

    def assign_source_labels(self, name_list=None):
        circuit_layout.assign_source_labels(self.netlist, name_list)
        
    def assign_gates_randomly(self, filters=(circuit_filters.no_collapsable_children,)):
        """Randomly assign gates. Because a random assignment can result in an invalid circuit,
           re-try until a valid circuit is generated. Give up and raise an exception if unsuccessful
           after many attempts.

           `filters` is an iterable of functions from circuit_filters (each takes the netlist and
           returns True if the circuit is acceptable). An assignment is kept only if every filter
           accepts it. valid_xor_xnor_fan_in_filter is always applied, whether or not it is listed.
           Pass `filters=()` to require only that.

           The filters judge the circuit as it is right after the gates are assigned. If you will
           add NOT gates afterwards (insert_random_source_nots), filters such as
           no_constant_subexpressions and no_visible_absorption may give a different answer for
           the finished circuit -- use `randomize` instead, which filters after adding the NOTs.

           The netlist is modified in place (`self.netlist` stays the same object). Any NOT gates
           already in it are kept."""
        self._randomize(filters, source_nots=False)

    def insert_random_source_nots(self):
        """Insert NOT gates at random on the edges leaving the sources (see
           circuit_generator.insert_random_source_edge_nots). The netlist must not already
           contain NOT gates, so that no input is inverted twice. No filters are applied."""
        self._require_no_not_gates('insert_random_source_nots')
        circuit_generator.insert_random_source_edge_nots(self.netlist)

    def randomize(self, filters=(circuit_filters.no_collapsable_children,), source_nots=True):
        """Randomly assign gates and (if `source_nots`) insert NOT gates on the edges leaving the
           sources, re-trying until the *finished* circuit passes every filter in `filters` (plus
           valid_xor_xnor_fan_in_filter, which is always applied). NOTs matter to some filters:
           they can turn A + A*B into the acceptable A + A'*B, or an acceptable A*B into the
           constant A*A'. Give up and raise an exception if unsuccessful after many attempts.

           Each attempt starts over from the netlist as it was when this method was called, so
           NOTs from rejected attempts never accumulate. With `source_nots`, the netlist must not
           already contain NOT gates. With `source_nots=False` this is assign_gates_randomly.

           The netlist is modified in place (`self.netlist` stays the same object)."""
        if source_nots:
            self._require_no_not_gates('randomize(source_nots=True)')
        self._randomize(filters, source_nots)

    # Each gate assignment that passes the gate-only filters gets this many random NOT patterns
    # before a new gate assignment is drawn: changing only the NOTs often fixes a rejected circuit.
    _NOT_PATTERNS_PER_GATE_ASSIGNMENT = 4

    def _require_no_not_gates(self, caller):
        if any(gate == 'NOT' for _, gate in self.netlist.nodes(data='gate')):
            raise ValueError(f'{caller} requires a netlist that has no NOT gates yet')

    def _restore(self, snapshot):
        # replace the netlist's contents with `snapshot`, keeping the same netlist object
        self.netlist.clear()
        self.netlist.update(snapshot)

    def _randomize(self, filters, source_nots):
        required = [circuit_filters.valid_xor_xnor_fan_in_filter]
        all_filters = required + [f for f in filters if f not in required]
        if source_nots:
            # Filters that can't be affected by source NOTs are checked once per gate assignment,
            # before any NOTs are added; the rest are checked on each finished circuit.
            before_nots = [f for f in all_filters if f in circuit_filters.GATE_ONLY_FILTERS]
            after_nots = [f for f in all_filters if f not in circuit_filters.GATE_ONLY_FILTERS]
        else:
            before_nots, after_nots = all_filters, []

        original = self.netlist.copy()

        def attempt(**gate_options):
            self._restore(original)
            circuit_generator.assign_gates_randomly(self.netlist, **gate_options)
            if not all(f(self.netlist) for f in before_nots):
                return False
            if not source_nots:
                return True
            gated = self.netlist.copy()
            for _ in range(self._NOT_PATTERNS_PER_GATE_ASSIGNMENT):
                self._restore(gated)
                circuit_generator.insert_random_source_edge_nots(self.netlist)
                if all(f(self.netlist) for f in after_nots):
                    return True
            return False

        for _ in range(0,100):
            if attempt():
                return

        # If we are unsuccessful at this point, don't assign XOR or XNOR (perhaps the graph has no nodes
        # with fan-in 2).
        for _ in range(0,500):
            if attempt(ensure_diversity=False, gate_list=['AND', 'OR', 'NAND', 'NOR']):
                return
        assert False, 'Unable to randomly assign gates!'

    def layout(self):
        return circuit_layout.layout_circuit(self.netlist)

    def layout_png(self):
        return circuit_layout.layout_circuit_png(self.netlist)

    def to_expression(self, format='latex', implicit_and=True):
        return circuit_analyzer.circuit_to_expression(self.netlist, format, implicit_and)

    def evaluate(self, input_value, variable_order=None):
        return circuit_analyzer.evaluate_circuit(self.netlist, input_value, variable_order)

    def generate_truth_table(self, variable_order=None):
        return circuit_analyzer.generate_truth_table(self.netlist, variable_order)
