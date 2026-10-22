"""
circuit_analyzer.py

Code to examine or evaluate circuits.
"""

PRECEDENCE = {
    None: 0,
    'XNOR': 1,
    'XOR': 1,
    'NOR': 1,
    'OR': 1,
    'NAND': 2,
    'AND': 2,
    'NOT': 3,
}

def circuit_to_expression_helper(netlist, node, parent_gate, format, implicit_and, require_parens):
    node_data = netlist.nodes[node]
    gate = node_data.get('gate')
    is_nand_nor = (gate == 'NAND' or gate == 'NOR')
    is_latex = (format == 'latex')

    #Base case: Node Labeled Src
    if gate == 'src':
        if 'label' not in node_data:
            raise ValueError(f"Source node {node} has no assigned label.")
        return str(node_data['label'])

    parens_required = require_parens or (PRECEDENCE[gate] <= PRECEDENCE[parent_gate] and not (is_latex and is_nand_nor))

    #Not case: adds a ' after the following thing
    if gate == 'NOT':
        sub_expression = circuit_to_expression_helper(netlist, next(netlist.predecessors(node)),gate, format, implicit_and, require_parens)
        if is_latex:
            return f'\\overline{{{sub_expression}}}'
        else:
            return f"{sub_expression}'"

    # Figuring out what symbol to represent gate
    symbol = "err"
    if gate == 'AND' or gate == 'NAND':
        symbol = "" if implicit_and else "*"
    elif gate == 'OR' or gate == 'NOR':
        symbol = " + "
    elif gate == 'XOR':
        symbol = " \\oplus " if format == 'latex' else " ^ "

    # beginning gate output str
    if is_latex and is_nand_nor:
        output_str = '\\overline{'
        if parens_required or parent_gate == 'AND' or parent_gate == 'NAND':
            output_str += '('
    elif parens_required or (is_nand_nor and not is_latex):
        output_str = '('
    else:
        output_str = ""
    
    used_open_paren = '(' in output_str
        

    #Recursively checking predecessors 
    children = list(netlist.predecessors(node))
    for index in range(len(children) - 1):
        output_str += f"{circuit_to_expression_helper(netlist, children[index], gate, format, implicit_and, require_parens)}{symbol}"
    output_str += f"{circuit_to_expression_helper(netlist, children[len(children) - 1], gate, format, implicit_and, require_parens)}"

    # Adding a not (') if gate was NAND or NOR
    if used_open_paren:
        output_str += ')'    
    
    if is_nand_nor:
        output_str += '}' if is_latex else "'"    
    
    return output_str

def circuit_to_expression(netlist, format='ascii', implicit_and=True, require_parens=False):
    #finding the sink(output)
    sink = [node for node in netlist.nodes() if netlist.out_degree(node) == 0][0]
    #Running recursive function (source nodes are rendered using their assigned 'label', if any)
    return circuit_to_expression_helper(netlist, sink, None, format, implicit_and, require_parens)

def evaluate_circuit_helper(netlist, input_map, node):
    gate = netlist.nodes[node].get('gate')

    if gate == 'src':
        return input_map[node]

    if gate == 'NOT':
        predecessor = next(netlist.predecessors(node))
        return not evaluate_circuit_helper(netlist, input_map, predecessor)

    predecessors = list(netlist.predecessors(node))
    values = [evaluate_circuit_helper(netlist, input_map, predecessor) for predecessor in predecessors]

    if gate in {'AND', 'NAND'}:
        result = all(values)
        if gate == 'NAND':
            return not result
        return result

    if gate in {'OR', 'NOR'}:
        result = any(values)
        if gate == 'NOR':
            return not result
        return result

    if gate == 'XOR':
        result = False
        for value in values:
            result = result ^ bool(value)
        return result

    raise ValueError(f"Unsupported gate type: {gate}")


def evaluate_circuit(netlist, input, variable_order=None):
    """
    Evaluate `netlist` for the input assignment encoded by `input`.

    Args:
        netlist: The circuit's netlist.
        input (int): An integer whose binary representation gives each
            source node's value -- one bit per source, most significant bit
            first, in `variable_order` if given, otherwise in ascending
            node-id order.
        variable_order (list[str] | None): Source labels (as assigned by
            `circuit_layout.assign_source_labels`) giving the order bits
            should map to sources. Defaults to ascending node-id order.

    Raises:
        ValueError: If `variable_order` is given but doesn't contain exactly
            the circuit's actual source labels (extra or missing names).
    """
    source_nodes = sorted(node for node in netlist.nodes() if netlist.in_degree(node) == 0)

    if variable_order is not None:
        label_to_node = {netlist.nodes[node].get('label'): node for node in source_nodes}

        unknown = sorted(set(variable_order) - label_to_node.keys())
        missing = sorted(label_to_node.keys() - set(variable_order))
        if unknown or missing:
            raise ValueError(
                f"variable_order {variable_order!r} does not match this circuit's "
                f"source labels {sorted(label_to_node.keys())!r}"
                + (f"; not a source: {unknown}" if unknown else "")
                + (f"; missing from variable_order: {missing}" if missing else "")
            )

        source_nodes = [label_to_node[name] for name in variable_order]

    bit_string = format(int(input), 'b').zfill(len(source_nodes))
    input_map = {node: bit == '1' for node, bit in zip(source_nodes, bit_string)}

    sink = [node for node in netlist.nodes() if netlist.out_degree(node) == 0][0]
    return evaluate_circuit_helper(netlist, input_map, sink)

def calculate_propagation_delay_helper(netlist, node, gate_delays, memo):
    if node in memo:
        return memo[node]

    gate = netlist.nodes[node].get('gate')

    if gate == 'src':
        memo[node] = 0
        return 0

    if gate not in gate_delays:
        raise ValueError(f"No delay specified for gate type: {gate}")

    predecessors = list(netlist.predecessors(node))
    arrival = max(calculate_propagation_delay_helper(netlist, predecessor, gate_delays, memo) for predecessor in predecessors)
    delay = arrival + gate_delays[gate]
    memo[node] = delay
    return delay

def calculate_propagation_delay(netlist, gate_delays):
    #finding the sink(output)
    sink = [node for node in netlist.nodes() if netlist.out_degree(node) == 0][0]
    #Longest path (in total gate delay) from any source to the sink
    return calculate_propagation_delay_helper(netlist, sink, gate_delays, {})

def generate_truth_table(netlist, variable_order=None):
    """
    Args:
        netlist: The circuit's netlist.
        variable_order (list[str] | None): Source labels giving the bit
            order to use, as in `evaluate_circuit`. Defaults to ascending
            node-id order.
    """
    if variable_order is not None:
        num_inputs = len(variable_order)
    else:
        num_inputs = sum(1 for node in netlist.nodes() if netlist.in_degree(node) == 0)

    truth_table = {}
    for input in range(0, 2**num_inputs):
        bit_string = format(int(input), 'b').zfill(num_inputs)
        truth_table[bit_string] = 1 if evaluate_circuit(netlist, input, variable_order) else 0
    return truth_table