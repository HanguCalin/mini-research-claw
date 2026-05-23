"""End-to-End wiring test for the LangGraph pipeline.

Mocks all 14 node functions to ensure the graph logic (edges and conditional routing)
works as expected without making real API calls.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import END

# Ensure we can import from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.graph import get_graph
from backend.state import AutoResearchState
import backend.graph as _graph_module


@pytest.fixture(autouse=True)
def reset_graph_singleton():
    """Reset the compiled graph singleton before each test so patches are applied fresh."""
    _graph_module._compiled_graph = None
    yield
    _graph_module._compiled_graph = None

# ─── Mock Node Responses ──────────────────────────────────────────────────────

def mock_node_factory(name, return_updates):
    def _mock_node(state):
        # Merge existing state with updates
        return return_updates
    return _mock_node

# ─── Test Case ────────────────────────────────────────────────────────────────

@patch("backend.graph.arxiv_retriever")
@patch("backend.graph.kg_extractor")
@patch("backend.graph.hypothesis_generator")
@patch("backend.graph.hitl_gate")
@patch("backend.graph.experiment_designer")
@patch("backend.graph.hitl_experiment_gate")
@patch("backend.graph.ml_coder")
@patch("backend.graph.dependency_resolver")
@patch("backend.graph.executor")
@patch("backend.graph.claim_ledger_builder")
@patch("backend.graph.academic_writer")
@patch("backend.graph.deterministic_linter")
@patch("backend.graph.critique_panel")
@patch("backend.graph.critique_aggregator")
@patch("backend.graph.latex_compiler")
def test_full_graph_execution_path(
    m_latex, m_agg, m_panel, m_lint, m_writer, m_ledger, 
    m_exec, m_dep, m_coder, m_hitl_exp, m_designer, m_hitl, 
    m_hypo, m_kg, m_arxiv
):
    """Verify that a successful run traverses the expected nodes."""
    
    # Setup happy path mocks
    m_arxiv.side_effect = mock_node_factory("arxiv", {"arxiv_papers_full_text": [{"id": "1"}], "retrieval_round": 1})
    m_kg.side_effect = mock_node_factory("kg", {"kg_entities": [{"id": "e1"}], "kg_edges": []})
    m_hypo.side_effect = mock_node_factory("hypo", {"hypothesis": "H1", "novelty_passed": True, "kg_valid": True})
    m_hitl.side_effect = mock_node_factory("hitl", {"hitl_approved": True})
    m_designer.side_effect = mock_node_factory("design", {"experiment_spec": {}})
    m_hitl_exp.side_effect = mock_node_factory("hitl_exp", {"hitl_experiment_approved": True})
    m_coder.side_effect = mock_node_factory("coder", {"python_code": "print(1)", "code_retry_count": 0})
    m_dep.side_effect = mock_node_factory("dep", {"resolved_dependencies": []})
    m_exec.side_effect = mock_node_factory("exec", {"execution_success": True, "metrics_json": {}})
    m_ledger.side_effect = mock_node_factory("ledger", {"claim_ledger": [], "pipeline_status": "drafting"})
    
    # Academic writer needs to handle two passes
    m_writer.side_effect = [
        {"latex_draft": "v1", "revision_pass_done": False}, # Pass 1
        {"latex_draft": "v2", "revision_pass_done": True},  # Pass 2
    ]
    
    m_lint.side_effect = mock_node_factory("lint", {"critique_warnings": []})
    m_panel.side_effect = mock_node_factory("panel", {"debate_log": [], "surviving_critiques": []})
    m_agg.side_effect = mock_node_factory("agg", {"aggregated_critique": "Better"})
    m_latex.side_effect = mock_node_factory("latex", {"final_pdf_path": "/tmp/paper.pdf", "pipeline_status": "success"})

    # Initialize state
    initial_state: AutoResearchState = {
        "topic": "test topic",
        "run_id": "test_run",
        "retrieval_round": 0,
        "code_retry_count": 0,
    }

    # Execute
    graph = get_graph()
    final_state = graph.invoke(initial_state)

    # Assertions
    assert final_state["pipeline_status"] == "success"
    assert final_state["final_pdf_path"] == "/tmp/paper.pdf"
    assert m_arxiv.called
    assert m_writer.call_count == 2 # Verify revision loop
    assert m_latex.called


def test_graph_fails_on_low_novelty():
    """Verify that the graph terminates if novelty is not passed."""
    with patch("backend.graph.arxiv_retriever") as m_arxiv, \
         patch("backend.graph.kg_extractor") as m_kg, \
         patch("backend.graph.hypothesis_generator") as m_hypo:
        
        m_arxiv.return_value = {"arxiv_papers_full_text": [], "retrieval_round": 1}
        m_kg.return_value = {"kg_entities": [], "kg_edges": []}
        m_hypo.return_value = {"novelty_passed": False, "pipeline_status": "failed_novelty"}

        initial_state = {"topic": "boring topic", "retrieval_round": 0}
        graph = get_graph()
        final_state = graph.invoke(initial_state)

        assert final_state["pipeline_status"] == "failed_novelty"
        # Ensure it didn't proceed to hitl_gate
        # We can't easily check 'hitl_gate.called' here without patching it too, 
        # but the state check is sufficient.
