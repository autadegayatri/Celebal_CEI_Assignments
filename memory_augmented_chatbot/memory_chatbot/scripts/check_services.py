"""
Verify PostgreSQL and Neo4j connectivity before running the rest of the
system -- run this first after `docker-compose up -d`.

Usage:
    python -m scripts.check_services
"""

from __future__ import annotations

import sys

from memory_augmented_chatbot.memory_chatbot.src import config


def check_postgres() -> bool:
    try:
        import psycopg2
    except ImportError:
        print("[postgres] FAILED -- `psycopg2` is not installed. Run `pip install psycopg2-binary`.")
        return False

    dsn = config.DATABASE_URL or (
        f"host={config.POSTGRES_HOST} port={config.POSTGRES_PORT} "
        f"dbname={config.POSTGRES_DB} user={config.POSTGRES_USER} "
        f"password={config.POSTGRES_PASSWORD}"
    )
    try:
        conn = psycopg2.connect(dsn)
        conn.close()
        print(f"[postgres] OK -- connected to {config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}")
        return True
    except Exception as exc:
        print(f"[postgres] FAILED -- {exc}")
        print(
            "  Make sure PostgreSQL is running (`docker-compose up -d postgres`) "
            "and DATABASE_URL / POSTGRES_* in .env are correct."
        )
        return False


def check_neo4j() -> bool:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[neo4j] FAILED -- `neo4j` driver is not installed. Run `pip install neo4j`.")
        return False

    try:
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        driver.verify_connectivity()
        driver.close()
        print(f"[neo4j]    OK -- connected to {config.NEO4J_URI}")
        return True
    except Exception as exc:
        print(f"[neo4j]    FAILED -- {exc}")
        print(
            "  Make sure Neo4j is running (`docker-compose up -d neo4j`) "
            "and NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in .env are correct."
        )
        return False


def check_groq() -> bool:
    if not config.GROQ_API_KEY:
        print("[groq]     FAILED -- GROQ_API_KEY is not set in .env.")
        return False
    print("[groq]     OK -- GROQ_API_KEY is set (not verified with a live call).")
    return True


def main():
    print("Checking required services...\n")
    results = [check_postgres(), check_neo4j(), check_groq()]
    print()
    if all(results):
        print("All services are reachable. You're ready to run:")
        print("  python -m scripts.build_knowledge_base")
    else:
        print("One or more services are not reachable -- fix the issues above before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
