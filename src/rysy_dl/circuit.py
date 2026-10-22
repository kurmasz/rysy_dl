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
        
    def assign_gates_randomly(self):
        """Randomly assign gates. Because a random assignment can result in an invalid circuit,
           re-try until a valid circuit is generated. Give up and raise an exception if unsuccessful 
           after many attempts."""
        for _ in range(0,100):
            circuit_generator.assign_gates_randomly(self.netlist)
            if circuit_filters.no_collapsable_children(self.netlist) and circuit_filters.valid_xor_xnor_fan_in_filter(self.netlist):
                return
            
        # If we are unsuccessful at this point, don't assign XOR or XNOR (perhaps the graph has no nodes
        # with fan-in 2).
        for _ in range(0,500):
            circuit_generator.assign_gates_randomly(self.netlist, ['AND', 'OR', 'NAND', 'NOR'])  
            if circuit_filters.no_collapsable_children(self.netlist):
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
