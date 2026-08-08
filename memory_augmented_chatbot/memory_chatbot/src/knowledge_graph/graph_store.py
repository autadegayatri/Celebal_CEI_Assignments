"""
Knowledge Graph storage layer.

Two backends behind a common interface:

- NetworkXGraphStore (default): in-process directed graph, persisted to
  disk with pickle. Zero setup -- this is what the system uses out of the
  box so the KG layer works anywhere, including sandboxes with no external
  services.
- Neo4jGraphStore (optional): talks to a real Neo4j instance via the
  official `neo4j` Python driver, exactly as the problem statement's tech
  stack specifies. Enable by installing `neo4j`, running a Neo4j instance,
  and setting GRAPH_BACKEND=neo4j (+ NEO4J_URI/USER/PASSWORD) in the
  environment.

Both backends implement: 'add_triple', 'add_entity', 'neighbors',
'find_paths', 'search_entities', 'save'/'load' (networkx only -- Neo4j is
already persistent by nature).
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx

from src import config
from src.knowledge_graph.entity_extractor import Triple


class GraphStore(ABC):
    @abstractmethod
    def add_entity(self, name: str, **attrs) -> None: ...

    @abstractmethod
    def add_triple(self, triple: Triple) -> None: ...

    @abstractmethod
    def neighbors(self, entity: str, max_hops: int = 1) -> list[dict]: ...

    @abstractmethod
    def search_entities(self, query: str, limit: int = 5) -> list[str]: ...


class NetworkXGraphStore(GraphStore):
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_entity(self, name: str, **attrs) -> None:
        if name not in self.graph:
            self.graph.add_node(name, **attrs)
        else:
            self.graph.nodes[name].update(attrs)

    def add_triple(self, triple: Triple) -> None:
        self.add_entity(triple.subject)
        self.add_entity(triple.obj)
        self.graph.add_edge(
            triple.subject,
            triple.obj,
            relation=triple.relation,
            sentence=triple.sentence,
        )

    def neighbors(self, entity: str, max_hops: int = 1) -> list[dict]:
        if entity not in self.graph:
            # fall back to a case-insensitive fuzzy match
            matches = self.search_entities(entity, limit=1)
            if not matches:
                return []
            entity = matches[0]

        results = []
        visited = {entity}
        frontier = [entity]
        for hop in range(max_hops):
            next_frontier = []
            for node in frontier:
                for _, target, data in self.graph.out_edges(node, data=True):
                    results.append(
                        {
                            "subject": node,
                            "relation": data.get("relation"),
                            "object": target,
                            "hop": hop + 1,
                        }
                    )
                    if target not in visited:
                        visited.add(target)
                        next_frontier.append(target)
                for source, _, data in self.graph.in_edges(node, data=True):
                    results.append(
                        {
                            "subject": source,
                            "relation": data.get("relation"),
                            "object": node,
                            "hop": hop + 1,
                        }
                    )
                    if source not in visited:
                        visited.add(source)
                        next_frontier.append(source)
            frontier = next_frontier
        return results

    def find_paths(self, source: str, target: str, max_length: int = 3) -> list[list[str]]:
        if source not in self.graph or target not in self.graph:
            return []
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
            return paths
        except nx.NetworkXNoPath:
            return []

    def search_entities(self, query: str, limit: int = 5) -> list[str]:
        query_lower = query.lower()
        exact = [n for n in self.graph.nodes if n.lower() == query_lower]
        if exact:
            return exact
        partial = [n for n in self.graph.nodes if query_lower in n.lower() or n.lower() in query_lower]
        return partial[:limit]

    def stats(self) -> dict:
        return {
            "num_entities": self.graph.number_of_nodes(),
            "num_relations": self.graph.number_of_edges(),
        }

    def save(self, path: Path = config.GRAPH_STORE_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)

    def load(self, path: Path = config.GRAPH_STORE_PATH) -> None:
        with open(path, "rb") as f:
            self.graph = pickle.load(f)


class Neo4jGraphStore(GraphStore):
    """Optional Neo4j-backed implementation. Requires `pip install neo4j`."""

    def __init__(
        self,
        uri: str = config.NEO4J_URI,
        user: str = config.NEO4J_USER,
        password: str = config.NEO4J_PASSWORD,
    ):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ImportError("Run `pip install neo4j` to use the Neo4j graph backend.") from exc
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def stats(self) -> dict:
        """Return basic counts for UI/diagnostics compatibility with NetworkX store."""
        with self.driver.session() as session:
            node_res = session.run("MATCH (n) RETURN count(n) AS c").single()
            rel_res = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
            return {"num_entities": node_res["c"], "num_relations": rel_res["c"]}

    def add_entity(self, name: str, **attrs) -> None:
        with self.driver.session() as session:
            session.run("MERGE (e:Entity {name: $name})", name=name)

    def add_triple(self, triple: Triple) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MERGE (s:Entity {name: $subject})
                MERGE (o:Entity {name: $object})
                MERGE (s)-[r:RELATION {type: $relation}]->(o)
                SET r.sentence = $sentence
                """,
                subject=triple.subject,
                object=triple.obj,
                relation=triple.relation,
                sentence=triple.sentence,
            )

    def neighbors(self, entity: str, max_hops: int = 1) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (e:Entity {{name: $name}})-[r:RELATION*1..{max_hops}]-(n)
                RETURN e.name AS subject, n.name AS object, r
                LIMIT 50
                """,
                name=entity,
            )
            return [dict(record) for record in result]

    def search_entities(self, query: str, limit: int = 5) -> list[str]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($q) RETURN e.name AS name LIMIT $limit",
                q=query,
                limit=limit,
            )
            return [record["name"] for record in result]


def get_graph_store(backend: str | None = None) -> GraphStore:
    backend = backend or config.GRAPH_BACKEND
    if backend == "neo4j":
        return Neo4jGraphStore()
    return NetworkXGraphStore()
