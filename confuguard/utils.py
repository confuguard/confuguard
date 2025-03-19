import os
import re
import json
import sqlalchemy
from loguru import logger
from google.cloud.sql.connector import Connector

import psycopg2

def clean_postgres(engine, table_name='embeddings_table'):
    try:
        with engine.connect() as conn:
            # Drop the table if it exists
            conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS {table_name};"))
            logger.info(f"Table {table_name} has been dropped (if it existed).")

            conn.commit()

    except Exception as e:
        logger.error(f"Failed to drop PostgreSQL table: {e}")
        raise

def init_connection_engine(db_user, db_pass, db_name, db_port="5433", is_local=False):
    def getconn():
        try:
            # Ensure db_pass is set to "postgres" for local connections if not provided
            password = "postgres" if is_local and not db_pass else db_pass

            # Both local and cloud proxy connections can use 127.0.0.1
            logger.debug(f"Connecting to PostgreSQL database -- user: {db_user}, dbname: {db_name}, port: {db_port}...")
            conn = psycopg2.connect(
                host="127.0.0.1",
                port=db_port,
                user=db_user,
                password=password,
                dbname=db_name
            )
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    return sqlalchemy.create_engine(
        "postgresql+psycopg2://",
        creator=getconn,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True
    )

