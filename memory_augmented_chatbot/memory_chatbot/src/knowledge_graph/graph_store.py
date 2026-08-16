"""
Knowledge Graph storage layer -- Neo4j only.

Talks to a real Neo4j instance via the official `neo4j` Python driver, as
specified in the problem statement's tech stack ("Graph DB: Neo4j"). Every
entity is stored as an `(:Entity {name})` node; every extracted relation
is stored as a `[:RELATION {type}]` edge between two entities.

Requires a running Neo4j instance (see docker-compose.yml) and
NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in the environment.
"""

from __future__ import annotations

from memory_augmented_chatbot.memory_chatbot.src import config
from memory_augmented_chatbot.memory_chatbot.src.knowledge_graph.entity_extractor import Triple


class Neo4jGraphStore:
    def __init__(
        self,
        uri: str = config.NEO4J_URI,
        user: str = config.NEO4J_USER,
        password: str = config.NEO4J_PASSWORD,
    ):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        """Unique constraint on Entity.name -- also creates a backing index,
        which is what keeps add_entity/search_entities/neighbors fast."""
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

    def close(self) -> None:
        self.driver.close()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add_entity(self, name: str) -> None:
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

    def clear(self) -> None:
        """Wipe the whole graph -- used before a fresh knowledge-base build
        so re-running the build script doesn't duplicate/stack old data."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def neighbors(self, entity: str, max_hops: int = 2) -> list[dict]:
        """
        Return every (subject, relation, object) edge within `max_hops` of
        `entity`, in either direction, flattened into plain dicts -- this
        is the shape `graph_query.format_neighbors` expects.
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH path = (e:Entity {{name: $name}})-[:RELATION*1..{int(max_hops)}]-(:Entity)
                UNWIND relationships(path) AS rel
                WITH DISTINCT startNode(rel) AS s, rel, endNode(rel) AS o
                RETURN s.name AS subject, rel.type AS relation, o.name AS object
                LIMIT 50
                """,
                name=entity,
            )
            return [{"subject": r["subject"], "relation": r["relation"], "object": r["object"]} for r in result]

    def search_entities(self, query: str, limit: int = 5) -> list[str]:
        with self.driver.session() as session:
            exact = session.run(
                "MATCH (e:Entity) WHERE toLower(e.name) = toLower($q) RETURN e.name AS name LIMIT $limit",
                q=query,
                limit=limit,
            )
            names = [r["name"] for r in exact]
            if names:
                return names

            partial = session.run(
                "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($q) RETURN e.name AS name LIMIT $limit",
                q=query,
                limit=limit,
            )
            return [r["name"] for r in partial]

    def stats(self) -> dict:
        with self.driver.session() as session:
            node_count = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS c").single()["c"]
            return {"num_entities": node_count, "num_relations": rel_count}


def get_graph_store() -> Neo4jGraphStore:
    return Neo4jGraphStore()
