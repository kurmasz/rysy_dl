"""
circuit_generator.py

Exhaustively generate combinatorial circuits to be used as exercises in computer organization courses.

The basic approach is to enumerate all directed acyclic graphs (DAGs) on `n` nodes, then assign logic gates to the
"non-source" nodes in all possible ways. (The source nodes represent the input variables to the circuit.)

To efficiently enumerate all DAGs with a single sink, we generate all DAGs with `n-1` nodes,
then add a single sink node and connect all nodes with out_degree 0 to it. This ensures that the graph
has a single output node.

Notes:
  * This module relies on the `geng` and `directg` programs from the nauty package to enumerate all DAGs.

Common workflow:
    * `generate_circuits`
       * `generate_nauty_digraphs` (calls nauty's `geng | directg`)
       * `nauty_to_networkx` (convert nauty's text format to a networkx DiGraph)
       * `_add_single_sink` (add a single sink node to the graph)
       * Apply filters to remove graphs that won't make good circuits (e.g., too many/few sources)
       * Assign gates to the graphs (when are then called netlists)
       * Add netlist to db.

:copyright: 2026, Zachary and Andrew Kurmas
"""

import copy
import gc
import os
import random
import shlex
import shutil
import signal
import subprocess
import time
import networkx as nx
from itertools import product

from .logging_config import get_logger
from .graph_analyzers import num_sources_and_gates
from .graph_filters import (
    num_sources_in_range,
    non_source_in_degrees_in_range,
    non_sink_out_degrees_in_range,
)
from .circuit_filters import no_collapsable_children, _FORBIDDEN_CHILD_GATE

logger = get_logger(__name__)


# The nauty programs this module runs. Some packagers (e.g., Debian/Ubuntu's apt package) install
# them with a "nauty-" prefix (nauty-geng, nauty-directg) to avoid name clashes in /usr/bin.
_NAUTY_PROGRAMS = ("geng", "directg")
_NAUTY_PREFIX = "nauty-"


def verify_nauty(nauty_dir):
    """
    Verify that `nauty_dir` contains executable ``geng`` and ``directg`` binaries.

    Each binary may use either its plain name (``geng``) or its ``nauty-`` prefixed name
    (``nauty-geng``). The plain name is preferred when both are present.

    :param nauty_dir: Path to the directory expected to contain the nauty binaries.
    :returns: A dict mapping ``"geng"`` and ``"directg"`` to the full path of each binary.
    :raises NotADirectoryError: If `nauty_dir` does not exist or is not a directory.
    :raises FileNotFoundError: If ``geng`` or ``directg`` is missing from `nauty_dir`.
    :raises PermissionError: If ``geng`` or ``directg`` exists but is not executable.
    """
    if not os.path.isdir(nauty_dir):
        raise NotADirectoryError(f"nauty directory not found: {nauty_dir}")
    paths = {}
    for program in _NAUTY_PROGRAMS:
        candidates = [os.path.join(nauty_dir, name) for name in (program, _NAUTY_PREFIX + program)]
        existing = [path for path in candidates if os.path.isfile(path)]
        if not existing:
            raise FileNotFoundError(
                f"nauty binary not found: {candidates[0]} (also tried {candidates[1]})"
            )
        executable = [path for path in existing if os.access(path, os.X_OK)]
        if not executable:
            raise PermissionError(f"nauty binary is not executable: {existing[0]}")
        paths[program] = executable[0]
    return paths


def find_nauty(nauty_dir=None):
    """
    Locate the ``geng`` and ``directg`` binaries.

    The binaries are found as follows:
      1. In `nauty_dir`, if given.
      2. In the directory named by the ``NAUTY`` environment variable, if `nauty_dir` is None.
      3. On ``PATH`` (e.g., after ``brew install nauty`` or ``apt install nauty``).

    In every case, each binary may use either its plain name (``geng``) or its ``nauty-``
    prefixed name (``nauty-geng``). An explicit `nauty_dir` or ``NAUTY`` is used as-is: if it
    doesn't contain the binaries, the error is raised rather than falling back to ``PATH``.

    :param nauty_dir: Directory containing the nauty binaries, or None to search for them.
    :returns: A dict mapping ``"geng"`` and ``"directg"`` to the full path of each binary.
    :raises NotADirectoryError: If the explicit/``NAUTY`` directory does not exist.
    :raises FileNotFoundError: If a binary can't be found.
    :raises PermissionError: If a binary exists but is not executable.
    """
    if nauty_dir is None:
        nauty_dir = os.environ.get("NAUTY")
    if nauty_dir is not None:
        return verify_nauty(nauty_dir)

    paths = {}
    for program in _NAUTY_PROGRAMS:
        path = shutil.which(program) or shutil.which(_NAUTY_PREFIX + program)
        if path is None:
            raise FileNotFoundError(
                f"nauty binary not found: neither {program} nor {_NAUTY_PREFIX}{program} is on PATH. "
                f"Install nauty, or set $NAUTY / --nauty-dir to the directory containing it."
            )
        paths[program] = path
    return paths


def generate_nauty_digraphs(num_nodes, nauty_dir=None):
    """
    Enumerate all non-isomorphic connected DAGs on `num_nodes` nodes using nauty.

    The nauty binaries are located by `find_nauty` (explicit `nauty_dir`, then ``NAUTY``,
    then ``PATH``; plain or ``nauty-`` prefixed names).


    Runs ``geng`` to generate all undirected graphs, pipes the output
    through ``directg`` to produce all non-isomorphic orientations as DAGs, and
    yields each DAG as a string in nauty's sparse6/digraph6 text format as soon as
    ``geng | directg`` produces it, rather than waiting for the whole pipeline to
    finish and buffering all of its output. This lets downstream code (e.g. the
    filter chain in `generate_circuits`, or a multiprocessing pool) start working
    on early results while nauty is still enumerating the rest, instead of
    blocking until every DAG has been generated.

    Note: These DAGs are not necessarily connected. When creating netlists, all
    connected components become inputs to the final gate.

    :param num_nodes: Number of nodes in each generated graph.
    :param nauty_dir: Directory containing the nauty binaries, or None to search for them.
    :yields: Each DAG as a string in nauty's text format, one at a time.
    :raises subprocess.CalledProcessError: If the ``geng | directg`` pipeline fails.
    """

    nauty_paths = find_nauty(nauty_dir)
    geng = shlex.quote(nauty_paths["geng"])
    directg = shlex.quote(nauty_paths["directg"])

    # -o pipefail: Use the
    cmd = [
        "bash",
        "-o",
        "pipefail",
        "-c",
        f"{geng} {num_nodes} | {directg} -aTq",
    ]
    # start_new_session=True makes `proc` the leader of its own process group, so geng and
    # directg (bash's children, not ours) can be killed together with it via os.killpg below.
    # Without this, proc.terminate() would only kill the bash wrapper and leave geng/directg
    # running as orphans.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    def _stream():
        # stdout/stderr are only None if Popen wasn't given PIPE for them; we always do, but
        # narrowing from an assert doesn't cross into a nested function's scope, so the
        # asserts have to live here rather than next to the Popen call above.
        assert proc.stdout is not None
        assert proc.stderr is not None
        try:
            for line in proc.stdout:
                line = line.strip()
                if line:
                    yield line
        except GeneratorExit:
            # The caller stopped consuming early (Ctrl-C, break, or this generator was simply
            # garbage-collected before being exhausted) while geng/directg were still running.
            # Kill the whole process group (bash + geng + directg), not just the bash wrapper.
            # Don't check/report an exit status below -- closing stdout out from under a
            # still-writing process makes it die from SIGPIPE (exit status 141), which isn't a
            # real failure, just us walking away early.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass  # already exited on its own
            proc.stdout.close()
            proc.stderr.close()
            proc.wait()
            raise
        else:
            # The loop above ended because the pipeline closed stdout on its own, i.e. it's
            # actually finished. Only now is checking/reporting its exit status meaningful.
            proc.stdout.close()
            stderr_output = proc.stderr.read()
            proc.stderr.close()
            returncode = proc.wait()
            if returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode, cmd, output=None, stderr=stderr_output
                )

    return _stream()


def nauty_to_networkx(nauty_graph):
    """Converts one line of nauty's directg -aTq output to a networkx DiGraph
    The format of this line is num_vertices num_edges u1 v1 u2 v2 ... un vn, where each ui vi pair is a directed edge from ui to vi.
    """
    parts = list(map(int, nauty_graph.split()))
    nv, ne = parts[0], parts[1]
    G = nx.DiGraph()
    G.add_nodes_from(range(nv))
    for i in range(ne):
        # u and v come in pairs beginning with index 2. 2*i moves in pairs.
        # 2+ skips the first two elements (nv and ne) to get to the first edge pair.
        # 3+ does the same thing, but also moves to the second element of the pair (v).
        u, v = parts[2 + 2 * i], parts[3 + 2 * i]
        G.add_edge(u, v)
    return G


def _add_single_sink(networkx_graph):
    """Adds a single sink node to the given networkx DiGraph, and connects all nodes with out_degree 0 to it. This ensures that the graph has a single output node.

    There is no efficient way to have nauty generate only DAGs with a single sink. (Generating all possible graphs
    then filtering out any without a single sink would be inefficient). Our solution is to generate
    a (possibly non-connected) graph, then add a single sink node.
    """

    num_of_nodes = len(networkx_graph)
    networkx_graph.add_node(num_of_nodes)
    for node in list(networkx_graph.nodes())[:-1]:
        if networkx_graph.out_degree(node) == 0:
            networkx_graph.add_edge(node, num_of_nodes)

    return networkx_graph


# ── Gate Assignments ────────────────────────────────────────────────────────────────


def _apply_gate_assignment(graph, gates: tuple, gate_list):
    # Create a dictionary that maps node index to gate type, then use
    # networkx's set_node_attributes to assign the gates to the graph
    attrs = {}
    for node, gate in enumerate(gates):
        if graph.in_degree(node) == 0:
            attrs[node] = {"gate": "src"}
        else:
            attrs[node] = {"gate": gate_list[gate]}
    nx.set_node_attributes(graph, attrs)


def assign_all_gates_and(graph):
    """
    Assigns AND to every non-source node in a single graph
    """
    gates_tuple = tuple(0 for _ in range(len(graph)))
    _apply_gate_assignment(graph, gates_tuple, ["AND"])


def assign_gates_randomly_ensure_diversity(graph, include_xnor=False):
    # Randomly assign gates to a single graph, but
    #  (1) Priorities AND, OR, and XOR
    #  (2) Limits the number of NAND/NOR/XNOR gates
    #
    # The goals are to
    #  (1) Include as many gates as possible in a single problem to more thoroughly
    #      test the student's understanding (or provide the student needed practice)
    #  (2) Avoid making the problem needlessly complex. For example, including many NAND
    #      and NOR gates makes the problem more difficult, but doesn't necessarily make it any
    #      better at demonstrating student knowledge / ability.

    # We have omitted XNOR by default: Including it trips students up, but doesn't
    # appear to provide much additional pedagogical benefit.
    simple_gates = ["AND", "OR", "XOR"]
    tricky_gates = ["NAND", "NOR"]
    if include_xnor:
        tricky_gates.append("XNOR")

    num_gate_choices = len(simple_gates) + len(tricky_gates)
    max_gates_supported = num_gate_choices + 1 + 2 * len(simple_gates)

    _, num_gates, _ = num_sources_and_gates(graph)

    if num_gates <= len(simple_gates):
        chosen_gates = random.sample(simple_gates, num_gates)
    elif num_gates <= num_gate_choices:
        # Use each gate type at most once. Use simple gates first.
        chosen_gates = simple_gates + random.sample(
            tricky_gates, num_gates - len(simple_gates)
        )
    elif num_gates <= num_gate_choices + len(simple_gates):
        # Use each gate type once. Then choose any additionally needed gates from the simple gates
        chosen_gates = (
            simple_gates
            + tricky_gates
            + random.sample(simple_gates, num_gates - num_gate_choices)
        )
    elif num_gates <= max_gates_supported:
        # Use each gate type once. Then use an additional "tricky" gate. Then choose any additionally needed gates from
        # the simple gates.
        chosen_gates = (
            simple_gates
            + tricky_gates
            + [random.choice(["NAND", "NOR"])]
            + random.sample(
                simple_gates + simple_gates, num_gates - (num_gate_choices + 1)
            )
        )
    else:
        raise ValueError(
            f'assign_gates_randomly_ensure_diversity only supports up to {max_gates_supported} gates {"with" if include_xnor else "without"} XNOR (got {num_gates})'
        )

    random.shuffle(chosen_gates)

    # chosen_gates has exactly one entry per gate (non-source) node; walk the
    # graph in node order and hand them out, marking source nodes as 'src'
    chosen_iter = iter(chosen_gates)
    attrs = {
        node: {"gate": "src"} if graph.in_degree(node) == 0 else {"gate": next(chosen_iter)}
        for node in range(len(graph))
    }

    # XOR/XNOR are only defined for 2-input gates. If one landed on a node
    # whose fan-in isn't 2, swap it with a node that does have exactly 2 inputs.
    misplaced = [
        node
        for node, attr in attrs.items()
        if attr["gate"] in ("XOR", "XNOR") and graph.in_degree(node) != 2
    ]
    two_input_nodes = [
        node
        for node, attr in attrs.items()
        if graph.in_degree(node) == 2 and attr["gate"] not in ("XOR", "XNOR")
    ]
    for node in misplaced:
        if two_input_nodes:
            partner = random.choice(two_input_nodes)
            two_input_nodes.remove(partner)
            attrs[node]["gate"], attrs[partner]["gate"] = (
                attrs[partner]["gate"],
                attrs[node]["gate"],
            )
        else:
            # No 2-input node left to swap with; fall back to a gate that
            # supports this node's fan-in.
            attrs[node]["gate"] = random.choice(["AND", "OR"])

    nx.set_node_attributes(graph, attrs)

def assign_gates_randomly(netlist, ensure_diversity=True, gate_list=None):
    if ensure_diversity:
        assign_gates_randomly_ensure_diversity(netlist)
        return

    # Randomly generates gate assignments for a single graph
    # gate_list default
    if gate_list == None:
        gate_list = ["AND", "OR", "XOR", "NAND", "NOR"]

    # XOR/XNOR are only defined for 2-input gates, so nodes with fan-in != 2
    # must draw from the gate list with XOR/XNOR excluded
    full_indices = list(range(len(gate_list)))
    restricted_indices = [
        i for i in full_indices if gate_list[i] not in ("XOR", "XNOR")
    ]

    # randomly generating a list, then converting to a tuple, to pass to node_gate_assigner
    gates_list = []
    for node in range(len(netlist)):
        if netlist.in_degree(node) == 0:
            # source node -- _apply_gate_assignment overrides this to 'src'
            # regardless, so no need to draw (and restricted_indices may be
            # empty if gate_list is XOR/XNOR only)
            gates_list.append(0)
        else:
            candidates = full_indices if netlist.in_degree(node) == 2 else restricted_indices
            gates_list.append(random.choice(candidates))
    gates_tuple = tuple(gates_list)
    # passing tuple to node_gate_assigner
    _apply_gate_assignment(netlist, gates_tuple, gate_list)

def all_gate_assignment(
    graph, filter_list=(), gate_list=("AND", "OR", "XOR", "NAND", "NOR")
):
    """
    Returns an iterator that yields all possible gate assignments for a single graph
    :param graph: A networkx DiGraph representing a DAG.
    :param filter_list: A list of filter functions to apply to each generated circuit.
    :param gate_list: A list of gate types to assign to the non-source nodes in the graph.
    :returns: A list of networkx DiGraphs, each representing a unique circuit"""

    # Note: The use of tuples above is intentional: We assume this function will not mutate filter_list or gate_list.
    # By using tuples, we will get a TypeError if the caller tries to mutate them, which is better than silently allowing it.

    # XOR/XNOR are only defined for 2-input gates, so nodes with fan-in != 2
    # must draw from the gate list with XOR/XNOR excluded
    non_xor_indices = [
        i for i, gate in enumerate(gate_list) if gate not in ("XOR", "XNOR")
    ]
    # source nodes always get 'src' regardless of gate choice, so only combine over gate nodes
    gate_nodes = [node for node in sorted(graph.nodes()) if graph.in_degree(node) != 0]
    base_choices = {
        node: (
            list(range(len(gate_list)))
            if graph.in_degree(node) == 2
            else non_xor_indices
        )
        for node in gate_nodes
    }

    # The graph has a single output (sink) node. For each gate we could put there,
    # drop any choice for its immediate children that would trivially collapse into
    # it (e.g. AND feeding AND, or OR feeding NOR) -- other collapsable pairs
    # elsewhere in the graph are left for the caller's filter_list to catch.
    root = next((node for node in gate_nodes if graph.out_degree(node) == 0), None)
    root_children = list(graph.predecessors(root)) if root is not None else []

    # placeholder value when there's no root; never read since it's only used
    # inside the `if root is not None` branch below
    for root_gate_index in (base_choices[root] if root is not None else [0]):
        choices_per_node = dict(base_choices)
        if root is not None:
            choices_per_node[root] = [root_gate_index]
            forbidden_child_gate = _FORBIDDEN_CHILD_GATE.get(gate_list[root_gate_index])
            if forbidden_child_gate is not None:
                for child in root_children:
                    if child in choices_per_node:
                        choices_per_node[child] = [
                            i
                            for i in choices_per_node[child]
                            if gate_list[i] != forbidden_child_gate
                        ]

        for combo in product(*(choices_per_node[node] for node in gate_nodes)):
            # logger.debug(f"Processing gate assignment {combo}")
            gates_list = [0] * len(graph)
            for node, gate_index in zip(gate_nodes, combo):
                gates_list[node] = gate_index
            _apply_gate_assignment(graph, tuple(gates_list), gate_list)
            if filter_helper(graph, filter_list):
                yield copy.deepcopy(graph)



# ── Adding NOT gates ────────────────────────────────────────────────────────────────


def _not_node_inserter(netlist, source_node, sequence):
    """
    Inserts a NOT gate between source_node and the set of successors for which 
    corresponding value in _sequence_ is True.
    
    Note that at most 1 NOT gate is added.
    """
        
    # Return without adding a NOT gates if _sequence_ contains 0 only.
    if not any(sequence):
        return
    
    # Create a new node, add it to the netlist and make this new node a NOT gate.
    not_node = len(netlist)
    netlist.add_node(not_node)
    netlist.nodes[not_node]["gate"] = "NOT"
    
    
    # Routes wires through NOT gate according to value in _sequence_
    for index, successor in enumerate(list(netlist.successors(source_node))):
        # Routes through not gate if matching index in 'sequence' is True
        if sequence[index]:
            netlist.add_edge(source_node, not_node)
            netlist.add_edge(not_node, successor)
            netlist.remove_edge(source_node, successor)


def all_possible_not_gates(graph, node):
    # generates list of all possible combinations of not gates
    combos = list(product([False, True], repeat=graph.out_degree(node)))
    graphs_with_nots = []
    # generates a copy of the graph, adds the nots, then appends it to graphs_with_nots list to be returned
    for combo in combos:
        not_graph = copy.deepcopy(graph)
        _not_node_inserter(graph, node, combo)
        graphs_with_nots.append(not_graph)
    return graphs_with_nots


def all_source_nots(graph):
    # creates and returns a list of all possible permutations of not gates after sources for a graph
    graphs_with_nots = []
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            if len(graphs_with_nots) == 0:
                graphs_with_nots.extend(all_possible_not_gates(graph, node))
            else:
                temp_list = []
                for not_graph in graphs_with_nots:
                    temp_list.extend(all_possible_not_gates(not_graph, node))
                graphs_with_nots.extend(temp_list)
    return graphs_with_nots


def insert_random_source_edge_nots(netlist):
    # Iterate over all paths from a source to a gate and randomly choose whether to insert a NOT gate.
    for node in range(0, len(netlist)):
        if netlist.in_degree(node) == 0:
            not_list = []
            for i in range(0, netlist.out_degree(node)):
                not_list.append(random.choice([False, True]))
            not_tuple = tuple(not_list)
            # passes not_tuple into not_node_inserter and appends
            _not_node_inserter(netlist, node, not_tuple)




def filter_helper(graph, filter_list):
    # logic for going through all gate filters
    for filter in filter_list:
        if not filter(graph):
            return False
    return True


# ── generate_circuits ────────────────────────────────────────────────────────────────


def generate_circuits(
    dag_size,
    db,
    min_num_source,
    max_num_source,
    min_num_inputs,
    max_num_inputs,
    min_num_outputs,
    max_num_outputs,
    gate_assignment,
    display_circuits,
    output_dir,
    nauty_dir=None,
    quiet=False,
    dry_run=False,
):
    """
    Generate combinatorial circuits and save them to a db.

    Enumerates DAGs with `dag_size` nodes using nauty, filters them by
    source count and fan-in/fan-out constraints, assigns logic gates, and
    writes each circuit as a PNG image and a pickle file to `output_dir`.

    Note: `dag_size` is not the number of gates, but the total number of nodes in the DAG
    on which the circuit is based. The number of gates is equal to `dag_size + 1` minus the number of source nodes (input variables)
    plus the number of inserted NOT gates.

    Processing stops after 10,000 circuits or when all qualifying graphs are
    exhausted, whichever comes first.

    :param dag_size: Total number of nodes in each generated DAG.
    :param db: An open `circuit_database.CircuitDatabase` instance that every generated circuit
               is recorded to via `db.add`. The caller owns opening/closing it (e.g. as a
               context manager) and is responsible for any prompting about an already-existing
               database file.
    :param min_num_source: Minimum number of source (input variable) nodes.
    :param max_num_source: Maximum number of source (input variable) nodes.
    :param min_num_inputs: Minimum fan-in (inputs per gate).
    :param max_num_inputs: Maximum fan-in (inputs per gate).
    :param min_num_outputs: Minimum fan-out (outputs per gate).
    :param max_num_outputs: Maximum fan-out (outputs per gate).
    :param gate_assignment: ``'random'`` assigns one random gate per graph;
                            ``'all'`` enumerates every possible gate assignment;
                            ``'and-only'`` assigns AND to every gate.
    :param display_circuits: If ``True``, display each circuit interactively and wait
                        for a keypress before continuing. Press ``q`` or
                        Ctrl-C to quit early.
    :param output_dir: Directory where output PNG and pickle files are written.
    :param nauty_dir: Path to the directory containing the nauty binaries. Defaults
                      to the ``NAUTY`` environment variable, then ``PATH`` (see `find_nauty`).
    :param quiet: If ``True``, suppress printing the parameter values at startup.
    :param generate_image: If ``True`` (default), write each circuit's PNG image to
                           `output_dir`. If ``False``, only the pickle file is written.
    :param dry_run: If ``True``, run the full generation process but skip ``db.add`` so
                    nothing is written to the database. `db` may be ``None`` in this case.
    :returns: The number of circuits generated.
    """
    if not quiet:
        print(
            f"dag_size={dag_size}, db={db}, min_num_source={min_num_source}, max_num_source={max_num_source}, "
            f"min_num_inputs={min_num_inputs}, max_num_inputs={max_num_inputs}, "
            f"min_num_outputs={min_num_outputs}, max_num_outputs={max_num_outputs}, "
            f"gate_assignment={gate_assignment}, display_circuits={display_circuits}, output_dir={output_dir}, "
            f"nauty_dir={nauty_dir}, quiet={quiet}"
        )

    initial_networkx_dags_iter = map(
        nauty_to_networkx, generate_nauty_digraphs(dag_size, nauty_dir=nauty_dir)
    )
    
    # If the initial dag has only one sink, then this operation creates a graph
    # where the new sink has only input.  This graph is then filtered out by 
    # the non_sink_out_degrees_in_range
    networkx_dags_iter = map(_add_single_sink, initial_networkx_dags_iter)

    filters = [
        # Require a range of input counts for the resulting circuits.
        lambda g: num_sources_in_range(g, min_num_source, max_num_source),
        # Require each gate's fan-in to be within the specified range
        lambda g: non_source_in_degrees_in_range(g, min_num_inputs, max_num_inputs),
        # Require each gate's fan-out to be within the specified range
        lambda g: non_sink_out_degrees_in_range(g, min_num_outputs, max_num_outputs),
    ]

    filtered_dags_iter = networkx_dags_iter
    for f in filters:
        filtered_dags_iter = filter(f, filtered_dags_iter)

    n = 0
    dag_count = 0
    last_status_time = time.monotonic()
    for raw_digraph in filtered_dags_iter:
        dag_count += 1

        if gate_assignment == "random":
            assign_gates_randomly(raw_digraph)
            netlists_from_digraph = [raw_digraph]
        elif gate_assignment == "all":
            # all_gate_assignment returns a generator, not a list, so we don't need to store all of its results in memory at once. Instead, we can iterate over it directly below.
            netlists_from_digraph = all_gate_assignment(
                raw_digraph, [no_collapsable_children]
            )
        elif gate_assignment == "and-only":
            assign_all_gates_and(raw_digraph)
            netlists_from_digraph = [raw_digraph]
        else:
            raise ValueError(
                f"Invalid gate_assignment: {gate_assignment!r} (expected 'random', 'all', or 'and-only')"
            )

        for netlist in netlists_from_digraph:
            n += 1
            insert_random_source_edge_nots(netlist)
            if not dry_run:
                db.add(netlist)

            if display_circuits:
                # Imported here (rather than at module scope) so that circuit_generator_display's
                # heavy matplotlib dependency -- and the matplotlib Figure it creates at import
                # time -- is only paid for when interactive display is actually requested. In
                # particular, circuit.py's PL grading path imports this module but never sets
                # show_graphs. Python caches imports, so re-importing here each iteration is cheap
                # (especially considering this loop waits for human input)
                from . import circuit_generator_display

                #  show_netlist returns True if the user pressed 'q' to quit early, False otherwise.
                if circuit_generator_display.show_netlist(netlist, raw_digraph):
                    print(" .")
                    return n

            if n % 10000 == 0:
                now = time.monotonic()
                print(
                    f"Processed {n} graphs ({now - last_status_time:.1f}s since last update)"
                )
                last_status_time = now
                gc.collect()

    print(f"Generated {n} circuits ...")
    return n
