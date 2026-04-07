from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TrajectoryStep:
    step_number: int
    node_name: str
    action: str
    input_summary: str
    output_summary: str
    timestamp: str


@dataclass
class GraphPath:
    nodes: List[str]
    relationships: List[str]
    evidence: List[str]
    path_length: int


@dataclass
class AgentState:
    question: str = ""
    strategy: Literal["vector_only", "graph_only", "hybrid"] = "hybrid"
    plan: List[str] = field(default_factory=list)
    vector_context: List[Dict] = field(default_factory=list)
    graph_paths: List[GraphPath] = field(default_factory=list)
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    token_usage: int = 0
    synthesis: str = ""
    confidence: float = 0.0
    citations: List[str] = field(default_factory=list)

    def add_trajectory(self, node: str, action: str, input_s: str, output_s: str):
        self.trajectory.append(TrajectoryStep(
            step_number=len(self.trajectory) + 1,
            node_name=node,
            action=action,
            input_summary=input_s[:100],
            output_summary=output_s[:100],
            timestamp=datetime.now().isoformat()
        ))

    def to_dict(self) -> Dict:
        return {
            'question': self.question,
            'strategy': self.strategy,
            'plan': self.plan,
            'vector_context': self.vector_context,
            'graph_paths': [{'nodes': p.nodes, 'relationships': p.relationships, 'path_length': p.path_length} for p in self.graph_paths],
            'trajectory': [{'step': t.step_number, 'node': t.node_name, 'action': t.action} for t in self.trajectory],
            'token_usage': self.token_usage,
            'synthesis': self.synthesis,
            'confidence': self.confidence,
            'citations': self.citations
        }