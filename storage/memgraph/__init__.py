"""Совместимость импортов для Memgraph хранилища."""

from src.database.graph_store import (
    GraphStore,
    create_graph_store,
    initialize_graph_store,
    graph_store,
)

__all__ = [
    "GraphStore",
    "create_graph_store",
    "initialize_graph_store",
    "graph_store",
]
