# Standard library imports
import os
import json
import re
import time
import argparse
import multiprocessing
from typing import List
from datetime import timedelta
from functools import partial
from multiprocessing import Pool, cpu_count, Manager

# Third-party imports
import struct
import numpy as np
import pandas as pd
import openai
import psycopg2
import sqlalchemy
from loguru import logger
from tqdm import tqdm
from filelock import FileLock
from dotenv import load_dotenv
from gensim.models import FastText
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import (
    Table,
    Column,
    Integer,
    String,
    MetaData,
    DateTime,
    func,
    Text,
    event
)
from sqlalchemy.dialects.postgresql import insert
from pgvector.sqlalchemy import Vector
import backoff
import tenacity
from openai import RateLimitError, APIError
from openai import OpenAI
import sqlite_vec

# Local imports
try:
    from python.typosquat.config import MODEL_PATH
    from python.typosquat.utils import init_connection_engine, clean_postgres
except:
    from config import MODEL_PATH
    from confuguard.utils import init_connection_engine, clean_postgres

load_dotenv('.env')

# Configuration
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
GCP_INSTANCE_NAME = os.getenv("GCP_INSTANCE_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

# Update SQLite database paths to include directory
SQLITE_DB_DIR = os.path.abspath('sqlite_dbs')
os.makedirs(SQLITE_DB_DIR, exist_ok=True)
SQLITE_BINARY_DB_PATH = os.path.join(SQLITE_DB_DIR, 'local_embeddings_binary.db')


BATCH_SIZE = 100  # Adjust batch size as needed
RESUME_FILE = './resume_state.json'  # File to save the resume state

# Constants
DELIMITERS = ('-', '_', ' ', '.', '~', '@', '/', ':')
DELIMITER_PATTERN = re.compile(f'[{"".join(DELIMITERS)}]+')

DATA_PATH = 'typosquat-data/typosquat-lfs/all_pkgs'

# Add these constants
OPENAI_RATE_LIMIT_RPM = 500  # Requests per minute for text-embedding-3-small
OPENAI_BATCH_SIZE = 2048  # Maximum tokens per request
OPENAI_MAX_RETRIES = 5
OPENAI_MIN_SECONDS = 0.1  # Minimum time between API calls

################################################################################
# Create a timestamp for the log file
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = f"./logs/embedding_run_{timestamp}.txt"


################################################################################
# Global variables for multiprocessing
model = None
preprocessor = None
engines = None  # Declare engines as a global variable

# Add OpenAI import and API key setup at the top
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client globally
client = OpenAI()

@backoff.on_exception(
    backoff.expo,
    (RateLimitError, APIError),
    max_tries=OPENAI_MAX_RETRIES
)
def get_openai_embeddings(texts):
    """
    Get embeddings from OpenAI with rate limiting and retries.
    """
    try:
        # logger.debug(f"Getting OpenAI embeddings for '{texts}'")
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        time.sleep(OPENAI_MIN_SECONDS)  # Basic rate limiting
        return response
    except Exception as e:
        logger.error(f"Error getting OpenAI embeddings: {e}")
        raise

class Preprocessor:
    def __init__(self, model, embedding_model="local"):
        self.delimiters = DELIMITERS
        self.delimiter_pattern = DELIMITER_PATTERN
        self.model = model
        self.embedding_model = embedding_model
        if embedding_model == "openai":
            self.client = OpenAI()  # Initialize OpenAI client

    def replace_delimiters(self, target: str, replacement: str) -> str:
        target = target.lower()
        delim_pass = re.sub(self.delimiter_pattern, replacement, target)
        num_pass = re.sub(r'[0-9]+', lambda match: match.group() + ' ', delim_pass)
        return num_pass

    def to_sequence(self, target: str) -> list:
        return self.replace_delimiters(target, ' ').split()

    def __call__(self, registry: str, package_names: list) -> dict:
        if registry == 'maven':
            # Maven package names are in the format group_id:artifact_id
            full_ids = package_names
            group_ids = [pkg.split(':')[0] for pkg in package_names]
            artifact_ids = [pkg.split(':')[1] for pkg in package_names]
            return {'full_ids': full_ids, 'group_ids': group_ids, 'artifact_ids': artifact_ids}
        elif registry == 'golang':
            # Golang package names are in the format namespace/package or namespace/author/package
            full_ids = package_names
            namespace_ids = []
            author_ids = []
            package_ids = []

            for pkg in package_names:
                parts = pkg.split('/')
                if len(parts) == 1:
                    namespace_ids.append(parts[0])
                    author_ids.append(parts[0])  # author ID is the same as namespace
                    package_ids.append('')
                elif len(parts) == 2:
                    namespace_ids.append(parts[0])
                    author_ids.append(parts[0])  # author ID is the same as namespace
                    package_ids.append(parts[1])
                else:
                    namespace_ids.append(parts[0])
                    author_ids.append(parts[1])
                    package_ids.append('/'.join(parts[2:]))

            return {'full_ids': full_ids, 'namespace_ids': namespace_ids, 'author_ids': author_ids, 'package_ids': package_ids}
        elif registry == 'hf':
            # HuggingFace package names are in the format author/package
            full_ids = package_names
            author_ids = []
            package_ids = []
            for pkg in package_names:
                parts = pkg.split('/')
                if len(parts) == 1:
                    author_ids.append('')
                    package_ids.append(parts[0])
                else:
                    author_ids.append(parts[0])
                    package_ids.append('/'.join(parts[1:]))
            return {'full_ids': full_ids, 'author_ids': author_ids, 'package_ids': package_ids}
        else:
            # For other registries, just return the full package names
            return {'full_ids': package_names}

    def get_embedding(self, name: str) -> np.ndarray:
        if self.embedding_model == "openai":
            name = name.replace("\n", " ")  # Clean text
            response = self.client.embeddings.create(
                input=[name],
                model="text-embedding-3-small"
            )
            return np.array(response.data[0].embedding).astype(np.float32)
        else:
            # Existing FastText embedding logic
            tokens = self.to_sequence(name)
            token_embeddings = []
            for token in tokens:
                if token in self.model.wv:
                    token_embeddings.append(self.model.wv[token])
                else:
                    token_embeddings.append(np.zeros(self.model.vector_size))
            return np.mean(token_embeddings, axis=0) if token_embeddings else np.zeros(self.model.vector_size)

    def get_embeddings(self, registry: str, components: dict) -> dict:
        embeddings = {}
        if self.embedding_model == "openai":
            try:
                # Clean text and batch process embeddings through OpenAI API with rate limiting
                if 'full_ids' in components:
                    full_texts = [text.replace("\n", " ") for text in components['full_ids']]
                    full_resp = get_openai_embeddings(full_texts)
                    embeddings['full_embeddings'] = np.array([d.embedding for d in full_resp.data])

                # For maven/golang/hf, also get secondary embeddings
                author_ids = components.get('author_ids') or components.get('group_ids')
                if author_ids:
                    auth_texts = [text.replace("\n", " ") for text in author_ids]
                    auth_resp = get_openai_embeddings(auth_texts)
                    embeddings['author_embeddings'] = np.array([d.embedding for d in auth_resp.data]).astype(np.float32)

                package_ids = components.get('package_ids') or components.get('artifact_ids')
                if package_ids:
                    pack_texts = [text.replace("\n", " ") for text in package_ids]
                    pack_resp = get_openai_embeddings(pack_texts)
                    embeddings['package_embeddings'] = np.array([d.embedding for d in pack_resp.data]).astype(np.float32)

            except Exception as e:
                logger.error(f"Failed to get OpenAI embeddings: {e}")
                raise
        else:
            # Existing FastText embedding logic
            if registry in ['maven', 'golang', 'hf']:
                full_ids = components['full_ids']
                embeddings['full_embeddings'] = np.array([self.get_embedding(name) for name in full_ids]).astype(np.float32)

                author_ids = components.get('author_ids') or components.get('group_ids')
                embeddings['author_embeddings'] = np.array([self.get_embedding(name) for name in author_ids]).astype(np.float32)

                package_ids = components.get('package_ids') or components.get('artifact_ids')
                embeddings['package_embeddings'] = np.array([self.get_embedding(name) for name in package_ids]).astype(np.float32)
            else:
                full_ids = components['full_ids']
                embeddings['full_embeddings'] = np.array([self.get_embedding(name) for name in full_ids]).astype(np.float32)

        return embeddings

    def load_package_names(self, file_path: str) -> list:
        data = pd.read_csv(file_path)

        if 'package_name' in data.columns:
            return data['package_name'].tolist()
        elif 'group_id' in data.columns and 'artifact_id' in data.columns:  # Maven packages
            return (data['group_id'] + ":" + data['artifact_id']).tolist()
        elif 'context_id' in data.columns:  # HF packages
            return data['context_id'].tolist()
        else:
            # If no valid columns are found, raise an error
            logger.error(f"None of the expected columns found in {file_path}. Available columns: {data.columns}")
            raise KeyError(f"'package_name' or 'group_id' and 'artifact_id' columns not found in the file: {file_path}")


def extract_source_from_filename(file_path):
    filename = os.path.basename(file_path)
    if 'npm' in filename:
        return 'npm'
    elif 'maven' in filename:
        return 'maven'
    elif 'pypi' in filename:
        return 'pypi'
    elif 'ruby' in filename:
        return 'ruby'
    elif 'golang' in filename:
        return 'golang'
    elif 'hf' in filename:
        return 'hf'
    elif 'nuget' in filename:
        return 'nuget'
    else:
        return 'unknown'


def save_resume_state(current_index, file_name, is_completed=False):
    """
    Saves the current processing state to a JSON file for resuming later.
    """
    lock = FileLock(RESUME_FILE + ".lock")
    with lock:
        try:
            # Attempt to load existing state
            state = {}
            if os.path.exists(RESUME_FILE):
                with open(RESUME_FILE, 'r') as f:
                    try:
                        state = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error loading resume state from {RESUME_FILE}: {e}")
                        logger.info("Resume state file is corrupted. Starting with an empty state.")
            else:
                logger.info("No existing resume state file found. Creating a new one.")

            # Update the state
            completed_flag = 'yes' if is_completed else 'no'
            state[file_name] = {
                "last_processed_index": current_index,
                'completed?': completed_flag
            }

            # Save the updated state
            with open(RESUME_FILE, 'w') as f:
                json.dump(state, f, indent=4)

        except Exception as e:
            logger.error(f"Error saving resume state: {e}")


def load_resume_state(file_name):
    """
    Loads the processing state from a JSON file to resume processing.
    """
    lock = FileLock(RESUME_FILE + ".lock")
    with lock:
        if os.path.exists(RESUME_FILE):
            try:
                with open(RESUME_FILE, 'r') as f:
                    state = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error loading resume state from {RESUME_FILE}: {e}")
                logger.info("Deleting corrupted resume state file and starting from index 0.")
                os.remove(RESUME_FILE)
                return 0

            if file_name in state:
                last_processed_index = state[file_name].get("last_processed_index", 0)
                completed = state[file_name].get('completed?', 'no')

                if completed == 'yes':
                    logger.info(f"{file_name} has already been processed completely.")
                    return -1  # Indicates processing is complete

                logger.info(f"Loaded resume state for {file_name}. Last processed index = {last_processed_index}")
                return last_processed_index
            else:
                logger.info(f"No resume state found for {file_name}. Starting from index 0.")
                return 0
        else:
            logger.info("No resume state file found. Starting from index 0.")
            return 0


def init_sqlite_engine(db_path):
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        logger.info(f"Created directory for SQLite database: {db_dir}")

    engine = sqlalchemy.create_engine(
        f'sqlite:///{db_path}',
        connect_args={"check_same_thread": False, "timeout": 30}
    )

    @event.listens_for(engine, "connect")
    def load_sqlite_vec(dbapi_connection, connection_record):
        try:
            dbapi_connection.enable_load_extension(True)
            sqlite_vec.load(dbapi_connection)
            dbapi_connection.enable_load_extension(False)
            # Set WAL mode
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            journal_mode = cursor.fetchone()
            logger.info(f"Set journal_mode to {journal_mode[0]}")
        except Exception as e:
            logger.error(f"Failed to load sqlite-vec extension on connection: {e}")
            raise

    logger.info(f"Initialized SQLite engine at {db_path}")
    return engine


def create_vector_table(engine, table_name, vector_dim, author_dim=None, package_dim=None):
    """
    Creates a virtual table for vector data using sqlite-vec with cosine similarity.
    """
    # Determine database format (for logging purposes)
    db_format = "unknown"
    if str(engine.url).endswith('binary.db'):
        db_format = "binary format"
    elif str(engine.url).endswith('float.db'):
        db_format = "float format"

    if engine.dialect.name == 'sqlite':
        vec_table = f"vec_{table_name}"
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS {vec_table}"))
            conn.commit()

            # Create the virtual table.
            # For the main embedding, we change the column name to embedding and specify cosine similarity.
            if author_dim is not None and package_dim is not None:
                conn.execute(sqlalchemy.text(f"""
                    CREATE VIRTUAL TABLE {vec_table}
                    USING vec0(
                        id INTEGER PRIMARY KEY,
                        embedding FLOAT[{vector_dim}] distance_metric=cosine,
                        author_embedding FLOAT[{vector_dim}],
                        package_embedding FLOAT[{vector_dim}]
                    );
                """))
            else:
                conn.execute(sqlalchemy.text(f"""
                    CREATE VIRTUAL TABLE {vec_table}
                    USING vec0(
                        id INTEGER PRIMARY KEY,
                        embedding FLOAT[{vector_dim}] distance_metric=cosine
                    );
                """))
            conn.commit()

            logger.info(f"Created virtual table {vec_table} in {db_format} database with cosine similarity (dimension: {vector_dim}).")


def init_sqlite_engines(binary_db_path):
    """Initialize binary format SQLite engine"""
    engines = {}

    # Create parent directory if it doesn't exist
    db_dir = os.path.dirname(binary_db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        logger.info(f"Created directory for SQLite database: {db_dir}")

    # Initialize binary format engine
    engines['sqlite_binary'] = sqlalchemy.create_engine(
        f'sqlite:///{binary_db_path}',
        connect_args={"check_same_thread": False, "timeout": 30}
    )
    logger.info(f"Initialized binary format SQLite engine at {binary_db_path}")

    # Set up engine with sqlite-vec extension
    @event.listens_for(engines['sqlite_binary'], "connect")
    def load_sqlite_vec(dbapi_connection, connection_record):
        try:
            dbapi_connection.enable_load_extension(True)
            sqlite_vec.load(dbapi_connection)
            dbapi_connection.enable_load_extension(False)
            # Set WAL mode
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            journal_mode = cursor.fetchone()
            logger.info(f"Set journal_mode to {journal_mode[0]} for sqlite_binary")
        except Exception as e:
            logger.error(f"Failed to load sqlite-vec extension on sqlite_binary connection: {e}")
            raise

    return engines

def create_table(engine, table_name, vector_dim, author_dim=None, package_dim=None):
    """
    Creates a table with the specified dimensions for embeddings in the given engine.
    """
    metadata = MetaData()

    # Determine the data types for embeddings based on the database
    if engine.dialect.name == 'postgresql':
        # Use Vector type for PostgreSQL
        embedding_type = Vector(vector_dim)
        author_type = Vector(author_dim) if author_dim else None
        package_type = Vector(package_dim) if package_dim else None
    elif engine.dialect.name == 'sqlite':
        # Use Text type to store JSON strings in SQLite
        embedding_type = Text
        author_type = Text if author_dim else None
        package_type = Text if package_dim else None
        logger.info(f"Using Text type for SQLite embeddings in table {table_name}")
    else:
        raise NotImplementedError(f"Unsupported database dialect: {engine.dialect.name}")

    # Create table if it doesn't exist
    if not engine.dialect.has_table(engine.connect(), table_name):
        columns = [
            Column('id', Integer, primary_key=True),
            Column('package_name', String, unique=True),
            Column('embedding', embedding_type),
            Column('created_at', DateTime, server_default=func.now()),
            Column('updated_at', DateTime, server_default=func.now(), onupdate=func.now())
        ]

        if author_dim:
            columns.append(Column('author_embedding', author_type))
        if package_dim:
            columns.append(Column('package_embedding', package_type))

        table = Table(table_name, metadata, *columns)
        metadata.create_all(engine)
        logger.info(f"Created table {table_name} in {engine.dialect.name} database")
    else:
        logger.info(f"Table {table_name} already exists in {engine.dialect.name} database")


def create_database(db_user, db_pass, db_name, db_port, is_local):
    """
    Creates a PostgreSQL database if it doesn't exist.

    Args:
        db_user (str): Database username.
        db_pass (str): Database password.
        db_name (str): Name of the database to create.
        db_port (str): Database port.
        is_local (bool): Whether to use a local connection or a cloud host.
    """
    host = "localhost" if is_local else "127.0.0.1"

    try:
        # Connect to the default 'postgres' database
        conn = psycopg2.connect(
            dbname="postgres",
            user=db_user,
            password=db_pass,
            host=host,
            port=db_port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Check if the database already exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone()

        if not exists:
            logger.info(f"Database {db_name} does not exist, creating database {db_name}")
            cur.execute(f"CREATE DATABASE {db_name}")
            logger.info(f"Created database: {db_name}")
        else:
            logger.info(f"Database {db_name} already exists")

        cur.close()
        conn.close()

    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise


def upload_embeddings_to_database(engine, components, embeddings, table_name='embeddings_table', id_offset=0):
    try:
        # Add debug logging for vector dimensions
        logger.info(f"Embedding shapes:")
        logger.info(f"- full_embeddings shape: {embeddings['full_embeddings'].shape}")
        if 'author_embeddings' in embeddings:
            logger.info(f"- author_embeddings shape: {embeddings['author_embeddings'].shape}")
        if 'package_embeddings' in embeddings:
            logger.info(f"- package_embeddings shape: {embeddings['package_embeddings'].shape}")

        # Determine which database format is being used
        db_format = "unknown"
        if str(engine.url).endswith('binary.db'):
            db_format = "binary format"

        logger.info(f"Uploading embeddings to {db_format} SQLite database for table {table_name}")

        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine)

        # Check if vec table exists before trying to autoload it
        with engine.connect() as conn:
            vec_table_exists = engine.dialect.has_table(conn, f"vec_{table_name}")

        if vec_table_exists:
            vec_table = Table(f"vec_{table_name}", metadata, autoload_with=engine)
        else:
            logger.warning(f"Vector table vec_{table_name} does not exist")
            return 0

        # For SQLite, wrap the connection in a FileLock to serialize writes.
        if engine.dialect.name == 'sqlite':
            lock_file = engine.url.database + ".lock"
            with FileLock(lock_file):
                with engine.connect() as conn:
                    total_inserted = 0
                    trans = conn.begin()
                    try:
                        batch_data = []
                        vec_batch_data = []

                        for idx, package_name in enumerate(components['full_ids']):
                            if not package_name or pd.isna(package_name):
                                continue

                            # Main table data
                            data_dict = {
                                'package_name': package_name,
                                'created_at': func.now(),
                                'updated_at': func.now(),
                            }
                            batch_data.append(data_dict)

                            # Vector table data with binary serialization for sqlite-vec
                            vec_dict = {
                                'id': idx + id_offset,
                                'embedding': sqlite_vec.serialize_float32(embeddings['full_embeddings'][idx]),
                            }

                            # Only add author and package embeddings if they exist in the embeddings dict
                            # AND if the corresponding columns exist in the vector table
                            has_author_embedding = 'author_embeddings' in embeddings
                            has_package_embedding = 'package_embeddings' in embeddings

                            if has_author_embedding:
                                vec_dict['author_embedding'] = sqlite_vec.serialize_float32(embeddings['author_embeddings'][idx])
                            if has_package_embedding:
                                vec_dict['package_embedding'] = sqlite_vec.serialize_float32(embeddings['package_embeddings'][idx])

                            vec_batch_data.append(vec_dict)

                        if batch_data:
                            result = conn.execute(
                                insert(table).values(batch_data).on_conflict_do_update(
                                    index_elements=['package_name'],
                                    set_=dict(updated_at=func.now())
                                )
                            )

                            for vec_data in vec_batch_data:
                                # Dynamically build the SQL statement based on available columns
                                columns = ['id', 'embedding']
                                placeholders = [':id', ':embedding']

                                if 'author_embedding' in vec_data:
                                    columns.append('author_embedding')
                                    placeholders.append(':author_embedding')

                                if 'package_embedding' in vec_data:
                                    columns.append('package_embedding')
                                    placeholders.append(':package_embedding')

                                # Build the SQL statement
                                sql = f"""INSERT OR REPLACE INTO vec_{table_name}
                                         ({', '.join(columns)})
                                         VALUES ({', '.join(placeholders)})"""

                                conn.execute(sqlalchemy.text(sql), vec_data)

                            total_inserted = result.rowcount

                        trans.commit()
                        logger.info(f"Successfully inserted/updated {total_inserted} rows in {table_name} ({db_format})")
                        return total_inserted

                    except Exception as e:
                        trans.rollback()
                        logger.error(f"Transaction failed for {db_format} database: {e}")
                        raise
        else:
            # For PostgreSQL, no additional lock is necessary.
            with engine.connect() as conn:
                total_inserted = 0
                trans = conn.begin()
                try:
                    batch_data = []
                    vec_batch_data = []
                    for idx, package_name in enumerate(components['full_ids']):
                        if not package_name or pd.isna(package_name):
                            continue
                        data_dict = {
                            'package_name': package_name,
                            'created_at': func.now(),
                            'updated_at': func.now(),
                            'embedding': embeddings['full_embeddings'][idx].tolist(),
                        }
                        if 'author_embedding' in embeddings:
                            data_dict['author_embedding'] = embeddings['author_embeddings'][idx].tolist()
                        if 'package_embedding' in embeddings:
                            data_dict['package_embedding'] = embeddings['package_embeddings'][idx].tolist()
                        batch_data.append(data_dict)
                    if batch_data:
                        insert_stmt = insert(table)
                        update_dict = {
                            'embedding': insert_stmt.excluded.embedding,
                            'updated_at': insert_stmt.excluded.updated_at
                        }
                        if 'author_embedding' in table.c:
                            update_dict['author_embedding'] = insert_stmt.excluded.author_embedding
                        if 'package_embedding' in table.c:
                            update_dict['package_embedding'] = insert_stmt.excluded.package_embedding
                        upsert_stmt = insert_stmt.on_conflict_do_update(
                            index_elements=['package_name'],
                            set_=update_dict
                        )
                        result = conn.execute(upsert_stmt, batch_data)
                        total_inserted = result.rowcount

                    trans.commit()
                    logger.info(f"Successfully inserted/updated {total_inserted} rows in {table_name} ({db_format})")
                    return total_inserted

                except Exception as e:
                    trans.rollback()
                    logger.error(f"Transaction failed for {db_format} database: {e}")
                    raise

    except Exception as e:
        logger.error(f"Failed to upload embeddings to database: {e}")
        raise


def init_worker(save_to, postgres_params, sqlite_binary_db_path, model_path=None, embedding_model="local"):
    global preprocessor
    global engines
    global model

    logger.info(f"Worker process {multiprocessing.current_process().name} starting.")

    # Load FastText model in each worker if using local embeddings
    if embedding_model == "local":
        try:
            logger.info(f"Loading FastText model from {model_path} in worker process")
            model = FastText.load(model_path)
            preprocessor = Preprocessor(model, embedding_model="local")
        except Exception as e:
            logger.error(f"Failed to load FastText model in worker process: {e}")
            raise
    else:
        # For OpenAI embeddings
        preprocessor = Preprocessor(None, embedding_model="openai")

    engines = {}

    if save_to in ['postgres', 'both']:
        engines['postgres'] = init_connection_engine(*postgres_params)
        logger.info(f"Worker process {multiprocessing.current_process().name} initialized with PostgreSQL engine.")
    if save_to in ['sqlite', 'both']:
        engines['sqlite_binary'] = init_sqlite_engine(sqlite_binary_db_path)
        logger.info(f"Worker process {multiprocessing.current_process().name} initialized with SQLite binary engine.")


def process_batch(args, lock, save_to):
    batch_packages, start_idx, end_idx, table_name, file_name = args

    batch_start_time = time.time()

    try:
        global preprocessor
        global engines

        # Process components and get embeddings
        components = preprocessor(table_name.split('_')[1], batch_packages)
        embeddings = preprocessor.get_embeddings(table_name.split('_')[1], components)

        if start_idx == 0:
            logger.debug(f"Embedding example for {table_name}: {embeddings}")

        # Upload to the specified databases
        if save_to in ['postgres', 'both']:
            upload_embeddings_to_database(engines['postgres'], components, embeddings, table_name=table_name)
        if save_to in ['sqlite', 'both']:
            upload_embeddings_to_database(engines['sqlite_binary'], components, embeddings, table_name=table_name, id_offset=start_idx)

        # Update resume state
        with lock:
            save_resume_state(end_idx, file_name)

        batch_end_time = time.time()
        batch_duration = timedelta(seconds=batch_end_time - batch_start_time)
        return True  # Return success status

    except Exception as e:
        logger.error(f"Error processing batch starting at index {start_idx}: {e}")
        return False


def process_in_batches(packages, source, file_name, batch_size, save_to, postgres_params,
                       sqlite_binary_db_path, embedding_model="local"):
    global engines

    # Initialize engines dictionary if it doesn't exist
    if engines is None:
        engines = {}

    # Adjust batch size for OpenAI to respect rate limits and token limits
    if embedding_model == "openai":
        # Calculate approximate tokens per package name (rough estimate)
        avg_tokens_per_pkg = sum(len(pkg.split()) for pkg in packages[:100]) / min(100, len(packages))
        max_pkgs_per_batch = min(
            OPENAI_BATCH_SIZE // int(avg_tokens_per_pkg),  # Token limit
            (OPENAI_RATE_LIMIT_RPM // 60) * 3  # Rate limit (3 embeddings per package for maven/golang/hf)
        )
        batch_size = min(batch_size, max_pkgs_per_batch)
        logger.info(f"Adjusted batch size for OpenAI: {batch_size}")

    total_packages = len(packages)
    logger.info(f"Total packages in {source}: {total_packages}")

    start_index = load_resume_state(file_name)
    if start_index == -1:
        logger.info(f"{file_name} has been processed completely")
        return

    table_name = f"typosquat_{source}_embeddings"

    # Prepare batches
    batches = []
    for start_idx in range(start_index, total_packages, batch_size):
        end_idx = min(start_idx + batch_size, total_packages)
        batch_packages = [pkg for pkg in packages[start_idx:end_idx] if pd.notna(pkg)]
        batches.append((batch_packages, start_idx, end_idx, table_name, file_name))

    if not batches:
        logger.info("No batches to process.")
        return

    file_start_time = time.time()

    # Use multiprocessing Pool to process batches in parallel
    num_processes = min(cpu_count() // 2, len(batches))
    logger.info(f"Using {num_processes} processes for parallel processing")

    manager = Manager()
    lock = manager.Lock()

    # Initialize counters for progress tracking
    total_batches = len(batches)
    successful_batches = 0
    failed_batches = 0

    # Initialize engines with logging
    if save_to in ['postgres', 'both']:
        engines['postgres'] = init_connection_engine(*postgres_params)
        logger.info(f"Using PostgreSQL database for processing {source}")

    if save_to in ['sqlite', 'both']:
        engines['sqlite_binary'] = init_sqlite_engine(sqlite_binary_db_path)
        logger.info(f"Using binary format SQLite database at {sqlite_binary_db_path} for {source}")

    with Pool(processes=num_processes, initializer=init_worker,
              initargs=(save_to, postgres_params, sqlite_binary_db_path,
                       MODEL_PATH, embedding_model)) as pool:
        # Create progress bar for overall progress
        with tqdm(total=total_batches, desc=f"Processing {source}",
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} batches '
                 '[{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:

            process_batch_with_lock = partial(process_batch, lock=lock, save_to=save_to)

            # Process batches and update progress
            for result in pool.imap_unordered(process_batch_with_lock, batches):
                if result:
                    successful_batches += 1
                else:
                    failed_batches += 1

                # Update progress bar with success/failure stats
                pbar.set_postfix({
                    'success': successful_batches,
                    'failed': failed_batches,
                    'success_rate': f"{(successful_batches/total_batches)*100:.1f}%"
                })
                pbar.update(1)

    file_end_time = time.time()
    file_duration = timedelta(seconds=file_end_time - file_start_time)
    logger.info(f"Finished processing {source}:")
    logger.info(f"- Total batches: {total_batches}")
    logger.info(f"- Successful: {successful_batches}")
    logger.info(f"- Failed: {failed_batches}")
    logger.info(f"- Total time: {file_duration}")


def ensure_pgvector_extension(engine):
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    logger.info("Ensured pgvector extension is installed and enabled.")


def clean_sqlite(engine, table_name='embeddings_table'):
    """
    Drops the specified table from the SQLite database.
    """
    metadata = MetaData()
    with engine.connect() as conn:
        if engine.dialect.has_table(conn, table_name):
            table = Table(table_name, metadata, autoload_with=engine)
            table.drop(engine)
            logger.info(f"Table {table_name} dropped from SQLite database")
        else:
            logger.info(f"Table {table_name} does not exist in SQLite database")

def log_run_metrics(file_path, source, packages_processed, start_time, end_time):
    duration = end_time - start_time
    throughput = packages_processed / duration if duration > 0 else 0
    # Convert latency to milliseconds (multiply by 1000)
    latency_ms = (duration / packages_processed * 1000) if packages_processed > 0 else 0

    metrics = {
        'ecosystem': source,
        'packages_processed': packages_processed,
        'duration_seconds': duration,
        'throughput_packages_per_second': throughput,
        'latency_ms_per_package': latency_ms  # Changed to milliseconds
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Upload FastText/OpenAI embeddings to databases")
    parser.add_argument('--clean', action='store_true', help="Clean the database tables before uploading")
    parser.add_argument('--save_to', choices=['postgres', 'sqlite', 'both'], default='postgres',
                        help="Choose which database(s) to save to: 'postgres', 'sqlite', or 'both'")
    parser.add_argument('--embedding_model', choices=['local', 'openai'], default='local',
                        help="Embedding model to use: 'local' (FastText) or 'openai'")
    parser.add_argument('--is_local', action='store_true', help="Use local PostgreSQL connection")

    args = parser.parse_args()

    # Add file handler to logger
    file_handler = logger.add(log_file, format="{time} | {level} | {message}", level="INFO")

    # Log initial parameters
    logger.info("=== Embedding Creation Run Configuration ===")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Parameters:")
    logger.info(f"  - Save to: {args.save_to}")
    logger.info(f"  - Is local: {args.is_local}")
    logger.info(f"  - Embedding model: {args.embedding_model}")
    logger.info(f"  - Clean database: {args.clean}")
    logger.info("=======================================")

    total_start_time = time.time()
    ecosystem_metrics = []

    try:
        engines = {}
        if not args.is_local:
            # Use cloud connection
            db_name = f"{DB_NAME}_openai" if args.embedding_model == "openai" else DB_NAME
            postgres_params = (DB_USER, DB_PASS, db_name, "5433", False)
            logger.info(f"Using cloud PostgreSQL database: {db_name}")
        else:
            # Use local connection
            db_name = "typosquat_embeddings_openai" if args.embedding_model == "openai" else "typosquat_embeddings_local"
            postgres_params = ('postgres', 'postgres', db_name, "5432", True)
            logger.info(f"Using local {'PostgreSQL' if args.save_to == 'postgres' else 'SQLite'} database: {db_name}")

        # Initialize engines based on save_to parameter
        if args.save_to in ['postgres', 'both']:
            try:
                engines['postgres'] = init_connection_engine(*postgres_params)
                # Try to ensure pgvector extension
                try:
                    ensure_pgvector_extension(engines['postgres'])
                except Exception as e:
                    if "does not exist" in str(e):
                        logger.warning(f"Database {postgres_params[2]} does not exist, creating database...")
                        engines['postgres'].dispose()
                        create_database(*postgres_params)
                        engines['postgres'] = init_connection_engine(*postgres_params)
                        ensure_pgvector_extension(engines['postgres'])
                        logger.success(f"Successfully created and connected to database {postgres_params[2]}")
                    else:
                        raise e
                logger.success(f"Connected to PostgreSQL instance {GCP_INSTANCE_NAME if not args.is_local else 'local'}")
            except Exception as e:
                logger.error(f"Failed to setup PostgreSQL connection: {e}")
                raise

        if args.save_to in ['sqlite', 'both']:
            try:
                # Modify SQLite database path to include embedding model type
                if args.embedding_model == "openai":
                    sqlite_binary_db_path = os.path.join(SQLITE_DB_DIR, 'local_embeddings_openai_binary.db')
                else:
                    sqlite_binary_db_path = SQLITE_BINARY_DB_PATH

                # Setup SQLite binary engine
                logger.info(f"Initializing binary format SQLite database at {sqlite_binary_db_path}")
                engines['sqlite_binary'] = init_sqlite_engine(sqlite_binary_db_path)
                logger.success(f"Connected to SQLite binary format database at {sqlite_binary_db_path}")
            except Exception as e:
                logger.error(f"Failed to setup SQLite connection: {e}")
                raise

        files = ['hf_packages.csv', 'maven_packages.csv', 'golang_packages.csv', 'npm_packages.csv', 'pypi_packages.csv', 'ruby_packages.csv', 'nuget_packages.csv']
        file_paths = [os.path.expanduser(os.path.join(DATA_PATH, file)) for file in files]

        # Update model loading logic
        global model
        if args.embedding_model == "local":
            logger.info(f"Loading FastText model from {MODEL_PATH}")
            model = FastText.load(MODEL_PATH)
            vector_dim = model.vector_size
            logger.info(f"FastText model loaded with vector size: {vector_dim}")
        else:
            model = None
            vector_dim = 1536  # OpenAI Ada embedding size
            logger.info("Using OpenAI embedding model (vector size 1536)")

        # Set the multiprocessing start method to 'fork' (only for Unix-based systems)
        multiprocessing.set_start_method('fork', force=True)

        # Update preprocessor initialization with correct embedding model choice
        preprocessor = Preprocessor(model, embedding_model=args.embedding_model)

        # Add overall progress bar for files
        with tqdm(total=len(file_paths), desc="Processing ecosystems",
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ecosystems '
                 '[{elapsed}<{remaining}]') as file_pbar:

            for file_path in file_paths:
                file_start_time = time.time()
                source = extract_source_from_filename(file_path)

                # Update description to show current ecosystem
                file_pbar.set_description(f"Processing {source}")

                # Load packages
                preprocessor_main = Preprocessor(None)
                packages = preprocessor_main.load_package_names(file_path)
                num_packages = len(packages)

                # Always clean tables when switching between embedding models to avoid dimension mismatch
                if args.clean or args.embedding_model == "openai":
                    if args.save_to in ['postgres', 'both']:
                        clean_postgres(engines['postgres'], table_name=f"typosquat_{source}_embeddings")
                    if args.save_to in ['sqlite', 'both']:
                        clean_sqlite(engines['sqlite_binary'], table_name=f"typosquat_{source}_embeddings")
                        # Also clean the vector table
                        with engines['sqlite_binary'].connect() as conn:
                            conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS vec_typosquat_{source}_embeddings"))
                            conn.commit()
                            logger.info(f"Dropped vector table vec_typosquat_{source}_embeddings")

                # Ensure that tables exist in the specified databases
                if args.save_to in ['postgres', 'both']:
                    with engines['postgres'].connect() as conn:
                        if not engines['postgres'].dialect.has_table(conn, f"typosquat_{source}_embeddings"):
                            if source in ['maven', 'golang', 'hf']:
                                author_dim = vector_dim
                                package_dim = vector_dim
                            else:
                                author_dim = None
                                package_dim = None
                            logger.warning(f"Table typosquat_{source}_embeddings not found, creating table typosquat_{source}_embeddings in PostgreSQL")
                            create_table(engines['postgres'], f"typosquat_{source}_embeddings", vector_dim, author_dim, package_dim)

                if args.save_to in ['sqlite', 'both']:
                    with engines['sqlite_binary'].connect() as conn:
                        main_table = f"typosquat_{source}_embeddings"
                        # Create the main table if it doesn't exist
                        if not engines['sqlite_binary'].dialect.has_table(conn, main_table):
                            if source in ['maven', 'golang', 'hf']:
                                author_dim = vector_dim
                                package_dim = vector_dim
                            else:
                                author_dim = None
                                package_dim = None
                            create_table(engines['sqlite_binary'], main_table, vector_dim, author_dim, package_dim)

                        # Ensure the corresponding vector table exists with all needed columns
                        vec_table = f"vec_{main_table}"
                        if not engines['sqlite_binary'].dialect.has_table(conn, vec_table):
                            if source in ['maven', 'golang', 'hf']:
                                create_vector_table(engines['sqlite_binary'], main_table, vector_dim, vector_dim, vector_dim)
                            else:
                                create_vector_table(engines['sqlite_binary'], main_table, vector_dim)

                # Pass the modified SQLite path to process_in_batches
                sqlite_path = sqlite_binary_db_path if args.embedding_model == "openai" else SQLITE_BINARY_DB_PATH

                process_in_batches(packages, source, file_path, batch_size=BATCH_SIZE,
                                  save_to=args.save_to,
                                  postgres_params=postgres_params,
                                  sqlite_binary_db_path=sqlite_path,
                                  embedding_model=args.embedding_model)

                file_end_time = time.time()

                # Calculate and store metrics for this ecosystem
                metrics = log_run_metrics(file_path, source, num_packages, file_start_time, file_end_time)
                ecosystem_metrics.append(metrics)

                logger.info(f"=== Metrics for {metrics['ecosystem']} ===")
                logger.info(f"Packages processed: {metrics['packages_processed']:,}")
                logger.info(f"Processing time: {metrics['duration_seconds']:.2f} seconds")
                logger.info(f"Throughput: {metrics['throughput_packages_per_second']:.2f} packages/second")
                logger.info(f"Latency: {metrics['latency_ms_per_package']:.2f} ms/package")
                logger.info("=======================================")

                file_pbar.update(1)

        # Dispose engines
        for eng in engines.values():
            eng.dispose()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception("Stack trace:")

    finally:
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # Log final summary
        logger.info("\n=== Final Run Summary ===")
        logger.info(f"Total execution time: {timedelta(seconds=total_duration)}")

        # Calculate aggregate metrics only if we have data
        if ecosystem_metrics:
            total_packages = sum(m['packages_processed'] for m in ecosystem_metrics)
            avg_throughput = sum(m['throughput_packages_per_second'] for m in ecosystem_metrics) / len(ecosystem_metrics)
            avg_latency = sum(m['latency_ms_per_package'] for m in ecosystem_metrics) / len(ecosystem_metrics)

            logger.info(f"Total packages processed: {total_packages:,}")
            logger.info(f"Average throughput: {avg_throughput:.2f} packages/second")
            logger.info(f"Average latency: {avg_latency:.2f} ms/package")
            logger.info("\nPer-ecosystem metrics:")

            # Create a table-like format for ecosystem metrics
            logger.info(f"{'Ecosystem':<12} {'Packages':>10} {'Duration(s)':>12} {'Throughput':>12} {'Latency(ms)':>12}")
            logger.info("-" * 60)
            for m in ecosystem_metrics:
                logger.info(f"{m['ecosystem']:<12} {m['packages_processed']:>10,} {m['duration_seconds']:>12.2f} "
                           f"{m['throughput_packages_per_second']:>12.2f} {m['latency_ms_per_package']:>12.2f}")
        else:
            logger.info("No ecosystem metrics collected during this run.")

        logger.info("=======================================")

        # Dispose engines
        for eng in engines.values():
            eng.dispose()

        # Remove the file handler
        logger.remove(file_handler)


if __name__ == "__main__":
    # Set the start method before calling main()
    if multiprocessing.get_start_method() != 'fork':
        multiprocessing.set_start_method('fork', force=True)
    main()
