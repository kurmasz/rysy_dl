"""
Unit tests for circuit_generator_display.show_netlist's control flow.

Drawing and waiting for the user are mocked out, so these tests never open a window.
"""

import os
import sys
from unittest.mock import MagicMock

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rysy_dl import circuit_generator_display as display  # noqa: E402


def _and_circuit():
    graph = nx.DiGraph([(0, 2), (1, 2)])
    nx.set_node_attributes(graph, {0: {'gate': 'src'}, 1: {'gate': 'src'}, 2: {'gate': 'AND'}})
    return graph


@pytest.fixture
def mocks(monkeypatch):
    """Mock out drawing, and script the user's actions via `mocks.actions`."""
    mocks = MagicMock()
    mocks.actions = []
    mocks.titles = []
    monkeypatch.setattr(display.circuit_layout, 'layout_circuit', mocks.layout_circuit)
    monkeypatch.setattr(display, 'display_circuit', mocks.display_circuit)
    monkeypatch.setattr(display, 'set_figure_title', lambda title, fig=None: mocks.titles.append(title))
    monkeypatch.setattr(display, 'wait_for_user_action',
                        lambda fig=None, show_reassign_button=False: mocks.actions.pop(0))

    def fake_assign_gates_randomly(netlist):
        mocks.assign_gates_randomly(netlist)
        netlist.nodes[2]['gate'] = 'OR'
    monkeypatch.setattr(display.circuit_generator, 'assign_gates_randomly', fake_assign_gates_randomly)
    return mocks


@pytest.mark.parametrize('action, expected', [(display.NEXT, False), (display.QUIT, True)])
def test_show_netlist_returns_whether_user_quit(mocks, action, expected):
    mocks.actions = [action]

    assert display.show_netlist(_and_circuit()) is expected
    mocks.assign_gates_randomly.assert_not_called()
    assert mocks.display_circuit.call_count == 1


def test_show_netlist_reassigns_and_redraws_until_user_moves_on(mocks):
    mocks.actions = [display.REASSIGN_GATES, display.REASSIGN_GATES, display.NEXT]
    netlist = _and_circuit()

    assert display.show_netlist(netlist) is False

    assert mocks.assign_gates_randomly.call_count == 2
    assert all(call.args[0] is netlist for call in mocks.assign_gates_randomly.call_args_list)
    assert mocks.display_circuit.call_count == 3


def test_show_netlist_title_tracks_reassigned_gates(mocks):
    mocks.actions = [display.REASSIGN_GATES, display.NEXT]

    display.show_netlist(_and_circuit(), title_prefix='id=7')

    assert len(mocks.titles) == 2
    assert all(title.startswith('id=7: ') for title in mocks.titles)
    assert mocks.titles[0] != mocks.titles[1]  # AND, then OR after reassigning


def test_show_netlist_title_uses_overlines_for_not(mocks):
    mocks.actions = [display.NEXT]
    netlist = nx.DiGraph([(0, 3), (3, 2), (1, 2)])
    nx.set_node_attributes(netlist, {0: {'gate': 'src'}, 1: {'gate': 'src'},
                                     2: {'gate': 'AND'}, 3: {'gate': 'NOT'}})

    display.show_netlist(netlist, title_prefix='id=7')

    # Wrapped in $...$ so matplotlib renders it as mathtext, with a bar over the negated term.
    assert mocks.titles == ['id=7: $\\overline{a}b$']


# ── test wait_for_user_action: buttons from earlier waits ──────────────────────────────────

def test_repeated_reassign_clicks_with_old_buttons_still_alive(monkeypatch):
    # fig.clf() removes a wait's buttons from the figure, but if the Button objects themselves
    # outlive the wait (an uncollected reference cycle, or the GUI backend holding them), they
    # used to keep handling presses at their old position. Clicking the new button in that spot
    # then raised "Another Axes already grabs mouse input".
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseEvent

    kept_alive = []

    class KeptAliveButton(display.Button):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            kept_alive.append(self)

    monkeypatch.setattr(display, 'Button', KeptAliveButton)
    fig = plt.figure()

    def click_reassign_button(timeout):
        ax = next(a for a in fig.axes if any(t.get_text() == 'Reassign Gates' for t in a.texts))
        x, y = ax.bbox.x0 + ax.bbox.width / 2, ax.bbox.y0 + ax.bbox.height / 2
        MouseEvent('button_press_event', fig.canvas, x, y, button=1)._process()
        MouseEvent('button_release_event', fig.canvas, x, y, button=1)._process()
    monkeypatch.setattr(fig.canvas, 'start_event_loop', click_reassign_button)

    try:
        for _ in range(3):
            assert display.wait_for_user_action(fig, show_reassign_button=True) == display.REASSIGN_GATES
            assert fig.canvas.mouse_grabber is None
    finally:
        plt.close(fig)
