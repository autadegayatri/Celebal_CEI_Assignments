"""
Verify Postgres connectivity and create the configured database if missing.

Usage:
    python -m scripts.verify_postgres
"""
from __future__ import annotations

import sys
from src import config

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except Exception as exc:
    print("psycopg2 is not installed or failed to import:", exc)
    sys.exit(1)


def main():
    host = config.POSTGRES_HOST
    port = config.POSTGRES_PORT
    user = config.POSTGRES_USER
    password = config.POSTGRES_PASSWORD
    target_db = config.POSTGRES_DB

    print(f"Connecting to Postgres at {host}:{port} as {user}...")
    try:
        conn = psycopg2.connect(dbname="postgres", user=user, password=password, host=host, port=port)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    except Exception as exc:
        print("Failed to connect to Postgres 'postgres' database:", exc)
        sys.exit(2)

    cur = conn.cursor()
    # Check if target_db exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
    exists = cur.fetchone() is not None
    if exists:
        print(f"Database '{target_db}' already exists.")
    else:
        try:
            print(f"Creating database '{target_db}'...")
            cur.execute(f"CREATE DATABASE {psycopg2.extensions.quote_ident(target_db, cur)}")
            print("Database created.")
        except Exception as exc:
            # fallback: use parameterized creation
            try:
                cur.execute("CREATE DATABASE %s" % psycopg2.extensions.QuotedString(target_db).getquoted().decode())
                print("Database created (fallback path).")
            except Exception as exc2:
                print("Failed to create database:", exc2)
                sys.exit(3)

    cur.close()
    conn.close()
    print("Postgres verification complete.")


if __name__ == '__main__':
    main()
