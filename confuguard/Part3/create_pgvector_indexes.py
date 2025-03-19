import os
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Index, inspect, MetaData, text, Table
from sqlalchemy.exc import NoSuchTableError
from pgvector.sqlalchemy import Vector
from google.cloud.sql.connector import Connector
from loguru import logger

from config import (
    HF_GCP_PROJECT_ID,
    HF_GCP_REGION,
    HF_GCP_INSTANCE_NAME,
    HF_DB_USER,
    HF_DB_PASS,
    HF_DB_PORT
)

################################################################################
# The three databases you want to index (quantized models)
################################################################################
DATABASES = [
    # "typosquat_float32",
    "typosquat_float16",
    "typosquat_int8"
]

################################################################################
# The registry tables you have in each database
################################################################################
REGISTRIES = [
    "npm",
    "maven",
    "pypi",
    "ruby",
    "golang",
    "hf"
]

################################################################################
# Helper function for connecting to Cloud SQL Postgres with pg8000 + Connector.
################################################################################
def init_connection_engine(project, region, instance_name, user, password, database):
    """
    Initialize SQLAlchemy engine for Google Cloud SQL (PostgreSQL).
    """
    logger.debug(
        f"Initializing connection engine for project: {project}, "
        f"region: {region}, instance: {instance_name}, user: {user}, "
        f"database: {database}"
    )

    def getconn():
        with Connector() as connector:
            conn = connector.connect(
                f"{project}:{region}:{instance_name}",
                "pg8000",
                user=user,
                password=password,
                db=database
            )
            return conn

    engine = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        pool_size=5,
        max_overflow=2,
        pool_timeout=30,
        pool_recycle=1800,
        echo=False  # Set to True if you want SQL echo logs
    )
    return engine

################################################################################
# Utility to get the size of a table
################################################################################
def get_table_size(engine, table_name):
    """Get the size of a table (and related indexes) in both pretty and raw bytes."""
    with engine.connect() as conn:
        size_query = text(f"""
            SELECT
                pg_size_pretty(pg_total_relation_size('{table_name}')) AS total_size,
                pg_size_pretty(pg_relation_size('{table_name}')) AS table_size,
                pg_size_pretty(pg_indexes_size('{table_name}')) AS index_size,
                pg_total_relation_size('{table_name}') AS total_bytes
            FROM pg_catalog.pg_tables
            WHERE tablename = '{table_name}';
        """)
        result = conn.execute(size_query).fetchone()
        return result if result else None

################################################################################
# Main logic to create HNSW indexes for each embedding column found.
################################################################################
def create_vector_indexes(engine, table_name):
    """
    For a given Postgres engine and table name, automatically detect
    the embedding columns and create an approximate HNSW index for each one.
    Returns timing and size information.
    """
    # Get size before indexing
    size_before = get_table_size(engine, table_name)
    start_time = time.time()

    # Use inspector to get table information
    insp = inspect(engine)
    if not insp.has_table(table_name):
        logger.warning(f"Table {table_name} does not exist.")
        return {
            'duration': 0,
            'size_before': None,
            'size_after': None
        }

    columns_info = insp.get_columns(table_name)
    columns = [col['name'] for col in columns_info]

    # Potential embedding columns to look for
    embedding_cols = []
    if "embedding" in columns:
        embedding_cols.append("embedding")
    if "author_embedding" in columns:
        embedding_cols.append("author_embedding")
    if "package_embedding" in columns:
        embedding_cols.append("package_embedding")

    if not embedding_cols:
        logger.info(f"No embedding columns found in table {table_name}, skipping.")
        return {
            'duration': 0,
            'size_before': None,
            'size_after': None
        }

    # Reflect the table to get SQLAlchemy Table object
    metadata = MetaData()
    try:
        table = Table(table_name, metadata, autoload_with=engine)
    except NoSuchTableError:
        logger.warning(f"Table {table_name} does not exist in the database.")
        return {
            'duration': 0,
            'size_before': None,
            'size_after': None
        }
    except Exception as e:
        logger.error(f"Error reflecting table {table_name}: {e}")
        return {
            'duration': 0,
            'size_before': None,
            'size_after': None
        }

    # Create HNSW indexes
    with engine.begin() as conn:
        for col in embedding_cols:
            index_name = f"{table_name}_{col}_hnsw_idx"

            # Define the Index using pgvector's Vector
            index = Index(
                index_name,
                table.c[col],
                postgresql_using='hnsw',
                postgresql_with={'m': 16, 'ef_construction': 64},
                postgresql_ops={col: 'vector_cosine_ops'}
            )

            # Create the index if it doesn't exist
            try:
                index.create(bind=engine, checkfirst=True)
                logger.info(f"Created HNSW index on {table_name}.{col} (if not exists).")
            except Exception as e:
                logger.error(f"Failed to create index {index_name} on {table_name}.{col}: {e}")

        # Optionally, ANALYZE table after creating indexes
        try:
            conn.execute(text(f"ANALYZE {table_name};"))
            logger.info(f"Analyzed table {table_name}.\n")
        except Exception as e:
            logger.error(f"Failed to analyze table {table_name}: {e}")

    end_time = time.time()
    size_after = get_table_size(engine, table_name)

    return {
        'duration': end_time - start_time,
        'size_before': size_before,
        'size_after': size_after
    }

################################################################################
# Main script entry point
################################################################################
def main():
    # Configure loguru to log to a file as well as the console
    logger.add("create_pgvector_indexes.log", rotation="1 MB", retention="10 days")

    # Create output directory if it doesn't exist
    output_dir = Path("./eval/EQ1-EmbeddingModel")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"indexing_stats_{timestamp}.txt"

    with open(output_file, "w") as f:
        f.write("PGVector Indexing Statistics (HNSW)\n")
        f.write("=" * 50 + "\n\n")

        for db_name in DATABASES:
            f.write(f"\nDatabase: {db_name}\n")
            f.write("-" * 50 + "\n")

            engine = init_connection_engine(
                HF_GCP_PROJECT_ID,
                HF_GCP_REGION,
                HF_GCP_INSTANCE_NAME,
                HF_DB_USER,
                HF_DB_PASS,
                db_name
            )

            # Ensure the 'vector' extension is enabled
            with engine.connect() as conn:
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    logger.info(f"Ensured 'vector' extension is enabled for database {db_name}")
                except Exception as e:
                    logger.error(f"Failed to create 'vector' extension in database {db_name}: {e}")
                    f.write(f"{db_name}: Failed to create 'vector' extension: {e}\n")
                    engine.dispose()
                    continue

            # For each known registry
            for registry in REGISTRIES:
                table_name = f"typosquat_{registry}_embeddings"

                # Check if table exists
                if not inspect(engine).has_table(table_name):
                    f.write(f"{table_name}: Table does not exist\n")
                    logger.info(f"Table {table_name} does not exist in database {db_name}, skipping.")
                    continue

                logger.info(f"Creating indexes for table: {table_name}")
                stats = create_vector_indexes(engine, table_name)

                f.write(f"\nTable: {table_name}\n")
                f.write(f"Indexing Duration: {stats['duration']:.2f} seconds\n")
                if stats["size_before"] and stats["size_after"]:
                    # size_before = (pretty_total, pretty_table, pretty_idx, raw_bytes)
                    f.write(f"Size Before: {stats['size_before'][0]}\n")  # e.g., '123 MB'
                    f.write(f"Size After: {stats['size_after'][0]}\n")    # e.g., '150 MB'
                    f.write(f"Index Size: {stats['size_after'][2]}\n")   # e.g., 'X MB'
                    size_increase_mb = (stats['size_after'][3] - stats['size_before'][3]) / (1024 * 1024)
                    f.write(f"Size Increase: {size_increase_mb:.2f} MB\n")
                else:
                    f.write("Size information not available.\n")
                f.write("\n")

            engine.dispose()
            f.write(f"Finished indexing for {db_name}.\n\n")

################################################################################
# Entry point
################################################################################
if __name__ == "__main__":
    main()
