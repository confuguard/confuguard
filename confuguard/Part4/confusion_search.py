import os
import re
import sys
import csv
import json
import time
from collections import Counter
from functools import lru_cache, wraps
from typing import List, Dict, Tuple, Any, Set
from threading import Lock

import struct
import numpy as np
from flask import Flask, request, jsonify, Response, stream_with_context
from google.cloud.sql.connector import Connector
from jellyfish import soundex, metaphone
from loguru import logger
from pgvector.sqlalchemy import Vector
from pyxdameraulevenshtein import damerau_levenshtein_distance
from sqlalchemy import create_engine, Table, MetaData, inspect, text, func, event
from sqlalchemy.dialects.postgresql import insert
from gensim.models import FastText
from rapidfuzz import fuzz
import psycopg2
import psutil
from huggingface_hub import HfApi, hf_hub_download
import sqlite_vec
import datetime



from config import (
    DB_PORT, DB_USER, DB_PASS, DB_NAME, DELIMITER_PATTERN,
    TYPOSQUAT_BEARER_TOKEN, MODEL_PATH, ECOSYSTEMS_MAPPING,
    OPENAI_API_KEY, NAMESPACE_ALLOWLIST, REGISTRIES,
    HF_DB_PORT, HF_DB_USER, HF_DB_PASS, HF_DB_NAME,
    IS_LOCAL, IS_OPENAI, IS_SQLITE, SQLITE_DB_PATH
)
from Part2.get_legit_packages import get_all_legit_packages
from Part2.create_embedding import Preprocessor, init_sqlite_engine
from Part5.benignity_check import FPVerifier
from utils import init_connection_engine


from urllib.parse import urlparse
from functools import wraps
from difflib import SequenceMatcher
from datetime import datetime, timezone

# Extract host and port from METADATA_ORIGIN
TYPOSQUAT_SERVICE_URL = os.environ.get('TYPOSQUAT_SERVICE_URL', 'http://localhost:5444')  # Ensure METADATA_ORIGIN is defined
parsed_url = urlparse(TYPOSQUAT_SERVICE_URL)

HOST = parsed_url.hostname
PORT = parsed_url.port or 5444

if IS_LOCAL:
    EMBEDDINGS_DB_USER = "postgres"
    EMBEDDINGS_DB_PASS = "postgres"
    EMBEDDINGS_DB_PORT = 5432
    EMBEDDINGS_DB_NAME = "typosquat_embeddings_local" if not IS_OPENAI else "typosquat_embeddings_openai"
else:
    EMBEDDINGS_DB_USER = DB_USER
    EMBEDDINGS_DB_PASS = DB_PASS
    EMBEDDINGS_DB_PORT = DB_PORT
    EMBEDDINGS_DB_NAME = DB_NAME

# Global caches
pop_packages_cache: Dict[str, List[Tuple[str, int]]] = {}
all_packages_cache: Dict[str, List[str]] = {}
command_cache: Dict[Tuple[str, str], Tuple[bool, Dict[str, float], str]] = {}
initial_memory = None


def register_sqlite_functions(db_connection):
    """Register custom SQLite functions."""
    def edit_distance(s1: str, s2: str) -> int:
        return damerau_levenshtein_distance(s1, s2)

    db_connection.create_function("edit_distance", 2, edit_distance)

class DatabaseManager:
    def __init__(self, embeddings_db_name: str = None):
        # Initialize connection settings based on environment
        self._init_connection_settings(embeddings_db_name)

        # Initialize database engines
        self.pg_engine = self._init_pg_engine()
        self.hf_engine = self._init_hf_engine()
        self.embeddings_engine = self._init_embeddings_engine()

    def _init_connection_settings(self, embeddings_db_name: str = None):
        """Initialize database connection settings based on environment"""
        if IS_LOCAL:
            self.embeddings_settings = {
                'user': "postgres",
                'pass': "postgres",
                'port': 5432,
                'name': "typosquat_embeddings_local" if not IS_OPENAI else "typosquat_embeddings_openai"
            }
        else:
            self.embeddings_settings = {
                'user': DB_USER,
                'pass': DB_PASS,
                'port': DB_PORT,
                'name': embeddings_db_name if embeddings_db_name else DB_NAME
            }

    def _init_pg_engine(self):
        """Initialize main PostgreSQL engine"""
        logger.info(f"Connecting to cloud postgres database with {DB_USER}, {DB_NAME}, {DB_PORT}")
        return init_connection_engine(
            db_user=DB_USER,
            db_pass=DB_PASS,
            db_name=DB_NAME,
            db_port=DB_PORT
        )

    def _init_hf_engine(self):
        """Initialize HuggingFace database engine"""
        logger.info(f"Connecting to cloud huggingface database with {HF_DB_USER}, {HF_DB_NAME}, {HF_DB_PORT}")
        return init_connection_engine(
            db_user=HF_DB_USER,
            db_pass=HF_DB_PASS,
            db_name=HF_DB_NAME,
            db_port=HF_DB_PORT
        )

    def _init_embeddings_engine(self):
        """Initialize embeddings database engine (SQLite or PostgreSQL)"""
        if IS_SQLITE:
            logger.info(f"Connecting to local SQLite embeddings database at {SQLITE_DB_PATH}")
            abs_db_path = os.path.abspath(SQLITE_DB_PATH)
            logger.info(f"Absolute database path: {abs_db_path}")
            engine = create_engine(
                f'sqlite:///{abs_db_path}',
                connect_args={"check_same_thread": False, "timeout": 30}
            )
            # Attach event listener to load the sqlite-vec extension on every connection.
            @event.listens_for(engine, "connect")
            def on_connect(dbapi_connection, connection_record):
                try:
                    dbapi_connection.enable_load_extension(True)
                    sqlite_vec.load(dbapi_connection)
                    dbapi_connection.enable_load_extension(False)
                    register_sqlite_functions(dbapi_connection)
                    logger.info("Loaded sqlite-vec extension and custom functions")
                except Exception as e:
                    logger.error(f"Failed to load extensions: {e}")
                    raise
            return engine
        else:
            logger.info(
                f"Connecting to {'local' if IS_LOCAL else 'cloud'} embeddings database with "
                f"{self.embeddings_settings['user']}, {self.embeddings_settings['name']}, {self.embeddings_settings['port']}"
            )
            return init_connection_engine(
                db_user=self.embeddings_settings['user'],
                db_pass=self.embeddings_settings['pass'],
                db_name=self.embeddings_settings['name'],
                db_port=self.embeddings_settings['port'],
                is_local=IS_LOCAL
            )

    def ensure_pg_extensions(self):
        # Skip extension creation for SQLite
        if IS_SQLITE:
            logger.debug("Skipping PostgreSQL extensions for SQLite database")
            return

        with self.embeddings_engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

    def verify_extension_loaded(self, conn):
        """Verify that sqlite-vec extension is properly loaded"""
        try:
            # Try to call a vector function to verify it exists
            version = conn.execute(text("SELECT vec_version()")).fetchone()
            # logger.info(f"SQLite-Vec extension loaded correctly. Version: {version[0]}")
            return True
        except Exception as e:
            logger.error(f"SQLite-Vec functions not available: {e}")
            return False

    def get_embeddings_table(self, registry: str) -> Table:
        metadata = MetaData()
        table_name = f"typosquat_{registry}_embeddings"

        try:
            if IS_SQLITE:
                # First verify the table exists
                with self.embeddings_engine.connect() as conn:
                    tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name;"),
                                        {"table_name": table_name}).fetchone()
                    if not tables:
                        raise ValueError(f"Table '{table_name}' does not exist in SQLite database")

                    # Get table info for debugging
                    columns = conn.execute(text(f"PRAGMA table_info({table_name});")).fetchall()

            # Try to load the table
            table = Table(table_name, metadata, autoload_with=self.embeddings_engine)
            return table
        except Exception as e:
            logger.error(f"Error accessing table '{table_name}': {str(e)}")
            raise Exception(f"Table '{table_name}' could not be accessed: {str(e)}")

    def get_embeddings(self, package_name: str, registry: str) -> np.ndarray:
        """Get full embedding for a package from either SQLite or PostgreSQL"""
        table = self.get_embeddings_table(registry)
        with self.embeddings_engine.connect() as conn:
            if IS_SQLITE:
                return self._get_sqlite_embeddings(conn, table, package_name)
            else:
                return self._get_postgres_embeddings(conn, table, package_name)

    def _get_sqlite_embeddings(self, conn, table, package_name):
        """Retrieve full embeddings from SQLite by joining the main and virtual table."""
        # Get rowid from main table
        main_query = text(f"SELECT rowid FROM {table.name} WHERE package_name = :package_name")
        rowid_result = conn.execute(main_query, {"package_name": package_name}).fetchone()
        if rowid_result:
            # Query virtual table for full embedding bytes
            vec_table = f"vec_{table.name}"
            vec_query = text(f"SELECT embedding FROM {vec_table} WHERE id = :rowid")
            result = conn.execute(vec_query, {"rowid": rowid_result[0]}).fetchone()
            if result and result[0]:
                # Properly deserialize the embedding from binary format
                vector_dim = len(result[0]) // 4  # 4 bytes per float
                embedding_list = list(struct.unpack(f'{vector_dim}f', result[0]))
                return np.array(embedding_list, dtype=float)
        logger.warning(f"No full embedding found for package '{package_name}' in SQLite database")
        return None

    def _get_postgres_embeddings(self, conn, table, package_name):
        """Retrieve full embeddings from PostgreSQL (assumed stored as JSON)"""
        query = text(f"SELECT embedding FROM {table.name} WHERE package_name = :package_name")
        result = conn.execute(query, {"package_name": package_name}).fetchone()
        if result and result[0]:
            embedding_list = json.loads(result[0])
            return np.array(embedding_list, dtype=float)
        logger.warning(f"No full embedding found for package '{package_name}' in PostgreSQL database")
        return None

    def get_author_embedding(self, package_name: str, registry: str) -> np.ndarray:
        """Retrieve the author embedding for a package."""
        table = self.get_embeddings_table(registry)
        with self.embeddings_engine.connect() as conn:
            if IS_SQLITE:
                # Get the rowid from the main table
                main_query = text(f"SELECT rowid FROM {table.name} WHERE package_name = :package_name")
                rowid_result = conn.execute(main_query, {"package_name": package_name}).fetchone()
                if rowid_result:
                    vec_table = f"vec_{table.name}"

                    # First check if the author_embedding column exists
                    columns_query = text(f"PRAGMA table_info({vec_table});")
                    columns = conn.execute(columns_query).fetchall()
                    column_names = [col[1] for col in columns]

                    if 'author_embedding' not in column_names:
                        logger.warning(f"Column 'author_embedding' not found in table '{vec_table}'")
                        return None

                    query = text(f"SELECT author_embedding FROM {vec_table} WHERE id = :rowid")
                    result = conn.execute(query, {"rowid": rowid_result[0]}).fetchone()
                    if result and result[0]:
                        # Properly deserialize the binary blob
                        vector_blob = result[0]
                        vector_dim = len(vector_blob) // 4  # 4 bytes per float
                        embedding_list = list(struct.unpack(f'{vector_dim}f', vector_blob))
                        return np.array(embedding_list, dtype=float)
            else:
                # PostgreSQL version (unchanged)
                query = text(f"SELECT author_embedding FROM {table.name} WHERE package_name = :package_name")
                result = conn.execute(query, {"package_name": package_name}).fetchone()
                if result and result[0]:
                    try:
                        return np.array(json.loads(result[0]), dtype=float)
                    except Exception as e:
                        logger.error(f"Error deserializing author_embedding for package '{package_name}': {e}")
                        return None
            logger.warning(f"No author embedding found for package '{package_name}'")
            return None

    def get_package_embedding(self, package_name: str, registry: str) -> np.ndarray:
        """Retrieve the package-specific embedding for a package."""
        table = self.get_embeddings_table(registry)
        with self.embeddings_engine.connect() as conn:
            if IS_SQLITE:
                main_query = text(f"SELECT rowid FROM {table.name} WHERE package_name = :package_name")
                rowid_result = conn.execute(main_query, {"package_name": package_name}).fetchone()
                if rowid_result:
                    vec_table = f"vec_{table.name}"

                    # First check if the package_embedding column exists
                    columns_query = text(f"PRAGMA table_info({vec_table});")
                    columns = conn.execute(columns_query).fetchall()
                    column_names = [col[1] for col in columns]

                    if 'package_embedding' not in column_names:
                        logger.warning(f"Column 'package_embedding' not found in table '{vec_table}'")
                        return None

                    query = text(f"SELECT package_embedding FROM {vec_table} WHERE id = :rowid")
                    result = conn.execute(query, {"rowid": rowid_result[0]}).fetchone()
                    if result and result[0]:
                        # Properly deserialize the binary blob
                        vector_blob = result[0]
                        vector_dim = len(vector_blob) // 4  # 4 bytes per float
                        embedding_list = list(struct.unpack(f'{vector_dim}f', vector_blob))
                        return np.array(embedding_list, dtype=float)
            else:
                # PostgreSQL version (unchanged)
                query = text(f"SELECT package_embedding FROM {table.name} WHERE package_name = :package_name")
                result = conn.execute(query, {"package_name": package_name}).fetchone()
                if result and result[0]:
                    try:
                        return np.array(json.loads(result[0]), dtype=float)
                    except Exception as e:
                        logger.error(f"Error deserializing package_embedding for package '{package_name}': {e}")
                        return None
            logger.warning(f"No package-specific embedding found for package '{package_name}'")
            return None


    @lru_cache(maxsize=128)
    def get_pkg_metadata(self, pkg_name: str, registry: str) -> dict:
        registry_mapped = ECOSYSTEMS_MAPPING.get(registry, registry)
        if registry_mapped == 'npm':
            query = """
            SELECT package_name, doc
            FROM public.npm_packages
            WHERE LOWER(package_name) = LOWER(:pkg_name)
            """
        elif registry_mapped == 'pypi.org':
            query = """
            SELECT
                pp.package_name,
                pp.payload as doc,
                array_agg(DISTINCT ppm.user) as maintainers
            FROM public.pypi_packages pp
            LEFT JOIN public.pypi_package_maintainers ppm
                ON pp.id = ppm.package_id
                AND ppm.removed_at IS NULL
            WHERE pp.normalized_package_name = LOWER(REGEXP_REPLACE(:pkg_name, '[-_.]+', '-', 'g'))
            GROUP BY pp.package_name, pp.payload
            """
        elif registry_mapped == 'repo1.maven.org':
            try:
                group_id, artifact_id = pkg_name.split(':', 1)
            except ValueError:
                logger.error(f"Invalid Maven package name format: '{pkg_name}'. Expected 'group_id:artifact_id'.")
                return {}

            ecosystems_query = """
            SELECT doc
            FROM public.ecosystems_packages
            WHERE LOWER(package_name) = LOWER(:pkg_name)
              AND registry = :registry
            """

            maven_query = """
            SELECT ma.raw_index AS description
            FROM public.maven_packages AS mp
            INNER JOIN public.maven_artifacts AS ma ON mp.id = ma.id
            WHERE LOWER(mp.group_id) = LOWER(:group_id)
              AND LOWER(mp.artifact_id) = LOWER(:artifact_id)
            """

            stmt_ecosystems = text(ecosystems_query)
            stmt_maven = text(maven_query)

            parameters_ecosystems = {"pkg_name": pkg_name, "registry": registry_mapped}
            parameters_maven = {"group_id": group_id, "artifact_id": artifact_id}

            try:
                with self.pg_engine.connect() as conn:
                    # Execute ecosystems_packages query
                    result_ecosystems = conn.execute(stmt_ecosystems, parameters=parameters_ecosystems).fetchone()

                    # Execute maven_packages and maven_artifacts query
                    result_maven = conn.execute(stmt_maven, parameters=parameters_maven).fetchone()
                    if result_ecosystems or result_maven:
                        metadata = {}
                        if result_ecosystems or result_maven:
                            metadata["doc"] = (result_ecosystems.doc if result_ecosystems else None) or (result_maven.description if result_maven else None)
                            metadata["name"] = pkg_name
                        return metadata
                    else:
                        logger.warning(f"Metadata not found for package '{pkg_name}' in registry '{registry}'.")
                        return {}
            except Exception as e:
                logger.error(f"Error fetching metadata for Maven package '{pkg_name}' from registry '{registry}': {e}")
                return {}
            return {}
        elif registry_mapped == 'huggingface.co':
            # TODO: Update this query to get the correct metadata
            query = """
            SELECT context_id as package_name, original_data
            FROM public.metadata
            WHERE LOWER(context_id) = LOWER(:pkg_name)
            """
            api = HfApi()
            model_info = api.model_info(pkg_name)
            # logger.debug(f"Model info: {model_info}")
            stmt = text(query)
            parameters = {"pkg_name": pkg_name}

            try:
                # Download the README.md file
                readme_path = hf_hub_download(repo_id=pkg_name, filename="README.md", repo_type="model")

                # Read the README content
                with open(readme_path, "r", encoding="utf-8") as file:
                    readme = file.read()
            except Exception as e:
                logger.warning(f"Error fetching README for {pkg_name}: {e}")
                readme = model_info.readme if model_info and model_info.readme else ""

            with self.hf_engine.connect() as conn:
                result = conn.execute(stmt, parameters=parameters).fetchone()

            metadata = {}
            metadata["name"] = pkg_name
            metadata["doc"] = result.original_data if result else {}
            metadata["doc"].update({"readme": readme})
            # logger.debug(f"Metadata for {pkg_name}: {metadata}")
            if metadata:
                return metadata
            else:
                return {}
        elif registry_mapped == 'nuget':
            query = """
            SELECT pp.package_name, pp.raw_metadata as doc, np.description as description, np.summary as summary, np.owners as owners
            FROM public.partitioned_packages pp
            LEFT JOIN public.nuget_packages np ON pp.package_name = np.package_name
            WHERE pp.ecosystem = 'nuget'
              AND LOWER(pp.package_name) = LOWER(:pkg_name)
            """
            stmt = text(query)
            parameters = {"pkg_name": pkg_name}

            try:
                with self.pg_engine.connect() as conn:
                    result = conn.execute(stmt, parameters=parameters).fetchone()
                    if result:
                        metadata = {
                            "name": result.package_name,
                            # "doc": result.doc,
                            "description": result.description,
                            "summary": result.summary,
                            "owners": result.owners
                        }
                        return metadata
                    else:
                        logger.warning(f"Metadata not found for package '{pkg_name}' in registry '{registry}'")
                        return {}
            except Exception as e:
                logger.error(f"Error fetching metadata for {pkg_name} from {registry}: {e}")
                return {}
        else:
            query = """
            SELECT package_name, doc
            FROM public.ecosystems_packages
            WHERE LOWER(package_name) = LOWER(:pkg_name)
              AND registry = :registry
            """

        stmt = text(query)
        parameters = {"pkg_name": pkg_name, "registry": registry_mapped}

        try:

            with self.pg_engine.connect() as conn:
                result = conn.execute(stmt, parameters=parameters).fetchone()
                if result:
                    metadata = result.doc
                    if registry_mapped == 'pypi.org':
                        metadata['maintainers'] = result.maintainers if result.maintainers else []
                    return metadata
                else:
                    logger.warning(f"Metadata not found for package '{pkg_name}' in registry '{registry}'")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching metadata for {pkg_name} from {registry}: {e}")
            return {}

    def get_typosquat_tables(self):
        """Get all table names that start with 'typosquat'"""
        with self.embeddings_engine.connect() as conn:
            query = text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name LIKE 'typosquat%'
                AND table_schema = 'public'
            """)
            results = conn.execute(query).fetchall()
            return [row[0] for row in results]

class TypoSim:
    def __init__(self):
        self.max_sim = 0

    def __call__(self, name1: str, name2: str) -> Tuple[float, Dict[str, float]]:
        similarities = {
            "normalize_damerau_levenshtein": self._normalize_damerau_levenshtein(name1, name2),
            "n_gram_similarity": self._n_gram_similarity(name1, name2),
            "phonetic_similarity": self._phonetic_similarity(name1, name2),
            "typosquat_similarity": self._typosquat_similarity(name1, name2),
            "substring_similarity": self._substring_similarity(name1, name2),
            "fuzzy_partial_ratio": self._fuzzy_partial_ratio(name1, name2)
        }
        self.max_sim = max(similarities.values())
        return self.max_sim, similarities

    def _normalize_damerau_levenshtein(self, name1: str, name2: str) -> float:
        # Handle empty strings
        if not name1 or not name2:
            return 0.0

        max_len = max(len(name1), len(name2))

        if max_len == 0:
            return 0.0  # Return 0 similarity for empty strings

        distance = damerau_levenshtein_distance(name1, name2)
        if distance < 3:
            return 1
        else:
            return 1 - distance / max_len

    def _n_gram_similarity(self, s1: str, s2: str, n: int = 2) -> float:
        # Handle empty strings or strings shorter than n
        if not s1 or not s2 or len(s1) < n or len(s2) < n:
            return 0.0

        def get_ngrams(s: str, n: int) -> List[str]:
            return [''.join(gram) for gram in zip(*[s[i:] for i in range(n)])]

        s1_ngrams = Counter(get_ngrams(s1.lower(), n))
        s2_ngrams = Counter(get_ngrams(s2.lower(), n))

        intersection = sum((s1_ngrams & s2_ngrams).values())
        union = sum((s1_ngrams | s2_ngrams).values())

        return intersection / union if union > 0 else 0.0

    def _phonetic_similarity(self, s1: str, s2: str) -> float:
        # Handle empty strings
        if not s1 or not s2:
            return 0.0

        try:
            soundex_sim = 1.0 if soundex(s1) == soundex(s2) else 0.0
            metaphone_sim = 1.0 if metaphone(s1) == metaphone(s2) else 0.0
            return (soundex_sim + metaphone_sim) / 2
        except Exception:
            return 0.0  # Return 0 for any phonetic encoding errors

    def _typosquat_similarity(self, s1: str, s2: str) -> float:
        s1, s2 = s1.lower(), s2.lower()

        substitutions = {
            '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '6': 'b', '7': 't',
            'rn': 'm', 'vv': 'w', 'cl': 'd', 'i': 'l', 'l': 'i'
        }

        for char, replacement in substitutions.items():
            if char in s1 and s1.replace(char, replacement) == s2:
                return 1.0
            if char in s2 and s2.replace(char, replacement) == s1:
                return 1.0

        # handle prefix/suffix single-char additions
        if s1.startswith(s2) or s2.startswith(s1):
            return 1 - (len(s1) - len(s2)) / 100

        if len(s1) == len(s2):
            diffs = sum(c1 != c2 for c1, c2 in zip(s1, s2))
            return 1 - diffs / 100

        return 0.0


    def _fuzzy_partial_ratio(self, s1: str, s2: str) -> float:
        # Handle empty strings
        if not s1 or not s2:
            return 0.0
        return fuzz.partial_ratio(s1, s2) / 100.0

    def _partial_overlap_score(self, s1: str, s2: str) -> float:
        """Returns a fractional overlap score between 0.0 and 1.0"""
        # Handle empty strings
        if not s1 or not s2:
            return 0.0

        def length_of_longest_common_substring(str1: str, str2: str) -> int:
            str1, str2 = str1.lower(), str2.lower()
            m, n = len(str1), len(str2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            max_length = 0

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if str1[i-1] == str2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                        max_length = max(max_length, dp[i][j])

            return max_length

        overlap_length = length_of_longest_common_substring(s1, s2)
        return overlap_length / max(len(s1), len(s2))

    def _substring_similarity(self, s1: str, s2: str) -> float:
        # Handle empty strings
        if not s1 or not s2:
            return 0.0

        s1_lower = s1.lower()
        s2_lower = s2.lower()

        # Split strings into components
        s1_parts = DELIMITER_PATTERN.split(s1_lower)
        s2_parts = DELIMITER_PATTERN.split(s2_lower)

        # Handle empty parts after splitting
        s1_parts = [p for p in s1_parts if p]
        s2_parts = [p for p in s2_parts if p]

        if not s1_parts or not s2_parts:
            return 0.0

        # Function to find all overlapping substrings between two lists of parts
        def find_overlaps(parts1: List[str], parts2: List[str]) -> List[Tuple[str, str, float]]:
            overlaps = []
            for part1 in parts1:
                for part2 in parts2:
                    score = self._partial_overlap_score(part1, part2)
                    if score > 0.5:  # Substantial overlap
                        overlaps.append((part1, part2, score))
            return overlaps

        common_parts = find_overlaps(s1_parts, s2_parts)

        if common_parts:
            # Calculate similarity based on matched components and their scores
            total_matched_length = sum(max(len(p1), len(p2)) * score
                                    for p1, p2, score in common_parts)
            total_length = sum(len(p) for p in s1_parts + s2_parts)

            component_similarity = total_matched_length / total_length

            # Increase similarity based on the number of overlapping substrings
            overlap_count = len(common_parts)
            max_possible_overlaps = min(len(s1_parts), len(s2_parts))
            overlap_ratio = overlap_count / max_possible_overlaps
            component_similarity *= (1 + overlap_ratio * 0.3)  # Up to 30% boost based on overlap count

            # Boost score if matching components are in the same order
            if len(common_parts) > 1:
                # Check if the order of common parts is consistent
                s1_indices = [s1_parts.index(p1) for p1, p2, _ in common_parts]
                s2_indices = [s2_parts.index(p2) for p1, p2, _ in common_parts]
                if s1_indices == sorted(s1_indices) and s2_indices == sorted(s2_indices):
                    order_bonus = 0.2
                    component_similarity += order_bonus

            # Additional boost for prefix matches
            if (s1_parts[0] in s2_parts[0]) or (s2_parts[0] in s1_parts[0]):
                component_similarity += 0.1

            # Additional boost for suffix matches
            if (s1_parts[-1] in s2_parts[-1]) or (s2_parts[-1] in s1_parts[-1]):
                component_similarity += 0.1

            return min(1.0, component_similarity)

        # If no component matches, fall back to direct substring comparison
        if s1_lower in s2_lower or s2_lower in s1_lower:
            shorter = min(len(s1_lower), len(s2_lower))
            longer = max(len(s1_lower), len(s2_lower))
            return 0.7 + 0.3 * (shorter / longer)

        return 0.0

class NamespaceAnalyzer:
    def __init__(self, preprocessor, logger):
        self.preprocessor = preprocessor
        self.logger = logger

    def _extract_namespace(self, name: str, registry: str) -> str:
        """Extract namespace based on registry-specific package name format."""
        if not name:
            return None

        try:
            if registry == 'npm':
                return name.split('/')[0][1:] if name.startswith('@') else None
            elif registry == 'maven':
                components = name.split(':')
                return components[0] if len(components) == 2 else None  # group_id
            elif registry == 'golang':
                components = name.split('/')
                return components[0] if len(components) > 1 else None  # domain
            elif registry == 'hf':
                components = name.split('/')
                return components[0] if len(components) == 2 else None  # author_name
            return None
        except Exception as e:
            self.logger.warning(f"Error extracting namespace from {name}: {str(e)}")
            return None

    def _compute_namespace_similarity(self, pkg_namespace: str, neighbor_namespace: str) -> float:
        """Compute similarity between two namespaces using edit distance."""
        if not pkg_namespace or not neighbor_namespace:
            return 0.0

        try:
            # Calculate edit distance
            distance = damerau_levenshtein_distance(pkg_namespace.lower(), neighbor_namespace.lower())

            # Return 1.0 if edit distance is less than 3, otherwise 0.0
            return 1.0 if distance < 3 else 0.0

        except Exception as e:
            self.logger.warning(f"Error computing namespace similarity: {str(e)}")
            return 0.0

    def has_suspicious_namespace(self, package_name: str, neighbor_name: str, registry: str) -> bool:
        """Check if two packages have different but suspiciously similar namespaces."""
        pkg_namespace = self._extract_namespace(package_name, registry)
        neighbor_namespace = self._extract_namespace(neighbor_name, registry)

        # If either namespace is missing, consider it suspicious
        if not pkg_namespace or not neighbor_namespace:
            return True

        # Check allowlist first (if applicable)
        if registry in NAMESPACE_ALLOWLIST:
            if pkg_namespace in NAMESPACE_ALLOWLIST[registry] or neighbor_namespace in NAMESPACE_ALLOWLIST[registry]:
                self.logger.debug(f"Namespace {pkg_namespace} or {neighbor_namespace} is allowlisted")
                return False

        # If namespaces are identical, it's not suspicious
        if pkg_namespace == neighbor_namespace:
            return False

        # Check if namespaces are suspiciously similar
        namespace_similarity = self._compute_namespace_similarity(pkg_namespace, neighbor_namespace)
        if namespace_similarity >= 0.8:  # High similarity threshold
            self.logger.debug(
                f"Suspicious namespace similarity ({namespace_similarity:.2f}) "
                f"between {package_name} ({pkg_namespace}) and {neighbor_name} ({neighbor_namespace})"
            )
            return True

        return False


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

def replace_delimiters(target: str, replacement: str) -> str:
    target = target.lower()
    delim_pass = re.sub(DELIMITER_PATTERN, replacement, target)
    num_pass = re.sub(r'[0-9]+', lambda match: match.group() + ' ', delim_pass)
    return num_pass

def to_sequence(target: str) -> List[str]:
    return replace_delimiters(target, ' ').split()

def split_package_name(package_name, registry):
    try:
        if registry == 'maven':
            components = package_name.split(':')
            if len(components) != 2:
                raise ValueError(f"Invalid Maven package name format: {package_name}")
            group_id, artifact_id = components
            group_domain = '.'.join(group_id.split('.')[:2])  # Extract domain part, e.g., 'com.example'
            return {
                'group_id': group_id,
                'group_domain': group_domain,
                'artifact_id': artifact_id,
                'package_component': 'artifact_id',
                'other_component': 'group_id',
                'delimiter': ':',
                'package_component_index': 2,
                'other_component_index': 1
            }
        elif registry == 'golang':
            # Golang package name format: domain.com/namespace/package
            # For GitHub packages, the namespace is github.com/username/repo
            if package_name.startswith('github.com/'):
                components = package_name.split('/')
                if len(components) < 3:
                    raise ValueError("Invalid Golang GitHub package name format.")
                domain = components[0]
                username = components[1].lower()  # Convert username to lowercase
                repo = components[2]
                subpath = '/'.join(components[3:]) if len(components) > 3 else ''
                return {
                    'domain': domain,
                    'username': username,
                    'repo': repo,
                    'subpath': subpath,
                    'package_component': 'repo',
                    'other_component': 'username',
                    'delimiter': '/',
                    'package_component_index': 3,
                    'other_component_index': 2,
                    'is_github': True
                }
            else:
                # Handle other domains
                components = package_name.split('/')
                if len(components) < 2:
                    raise ValueError("Invalid Golang package name format.")
                domain = components[0]
                namespace = components[1]
                subpath = '/'.join(components[2:]) if len(components) > 2 else ''
                return {
                    'domain': domain,
                    'namespace': namespace,
                    'subpath': subpath,
                    'package_component': 'namespace',
                    'other_component': 'domain',
                    'delimiter': '/',
                    'package_component_index': 2,
                    'other_component_index': 1,
                    'is_github': False
                }
        elif registry == 'npm':
            # NPM package names can be scoped (@scope/package) or unscoped
            if package_name.startswith('@'):
                scope, package = package_name.split('/')
                return {
                    'scope': scope,
                    'package': package,
                    'package_component': 'package',
                    'other_component': 'scope',
                    'delimiter': '/',
                    'package_component_index': 2,
                    'other_component_index': 1
                }
            else:
                return {
                    'package': package_name,
                    'package_component': 'package',
                    'other_component': '',
                    'delimiter': '',
                    'package_component_index': 1,
                    'other_component_index': 0
                }
        elif registry == 'hf':
            # Parse HuggingFace package name (format: author_name/model_name)
            components = package_name.split('/')
            if len(components) != 2:
                raise ValueError(f"Invalid HuggingFace package name format: {package_name}")

            return {
                'author_name': components[0],
                'model_name': components[1],
                'package_component': 'model_name',
                'other_component': 'author_name',
                'delimiter': '/',
                'package_component_index': 2,
                'other_component_index': 1
            }
        else:
            raise ValueError(f"Unsupported registry {registry}")
    except ValueError as e:
        logger.error(f"Invalid package format: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing package '{package_name}': {str(e)}")

    return None

def edit_distance(s1: str, s2: str) -> int:
    """Compute Damerau-Levenshtein edit distance between two strings."""
    return damerau_levenshtein_distance(s1.lower(), s2.lower())

def add_package_to_sqlite(package_name: str, registry: str, db_manager: DatabaseManager, embeddings: dict) -> bool:
    """
    Add a package and its embeddings to the SQLite database.
    This handles the specialized author_embedding and package_embedding fields when available.
    """
    try:
        # Get the table name
        table_name = f"typosquat_{registry}_embeddings"
        vec_table = f"vec_{table_name}"

        # Use Python's datetime instead of SQLAlchemy func.now()
        current_time = datetime.now()

        with db_manager.embeddings_engine.connect() as conn:
            # Start a transaction
            trans = conn.begin()
            try:
                # Insert into main table to get the rowid
                conn.execute(
                    text(f"INSERT OR REPLACE INTO {table_name} (package_name, created_at, updated_at) VALUES (:name, :created, :updated)"),
                    {"name": package_name, "created": current_time, "updated": current_time}
                )

                # Get the rowid
                result = conn.execute(
                    text(f"SELECT rowid FROM {table_name} WHERE package_name = :name"),
                    {"name": package_name}
                ).fetchone()
                rowid = result[0]

                # First, check if a record already exists in the vec table and delete it to avoid constraint error
                conn.execute(
                    text(f"DELETE FROM {vec_table} WHERE id = :id"),
                    {"id": rowid}
                )

                # Prepare vector data with binary serialization
                vec_data = {
                    'id': rowid,
                    'embedding': sqlite_vec.serialize_float32(embeddings['full_embeddings'][0].tolist()),
                }

                # Add author and package embeddings if they exist
                if 'author_embeddings' in embeddings and embeddings['author_embeddings'][0] is not None:
                    vec_data['author_embedding'] = sqlite_vec.serialize_float32(embeddings['author_embeddings'][0].tolist())

                if 'package_embeddings' in embeddings and embeddings['package_embeddings'][0] is not None:
                    vec_data['package_embedding'] = sqlite_vec.serialize_float32(embeddings['package_embeddings'][0].tolist())

                # Dynamically build the SQL statement based on available columns
                columns = ['id', 'embedding']
                placeholders = [':id', ':embedding']

                if 'author_embedding' in vec_data:
                    columns.append('author_embedding')
                    placeholders.append(':author_embedding')

                if 'package_embedding' in vec_data:
                    columns.append('package_embedding')
                    placeholders.append(':package_embedding')

                # Build and execute the SQL statement
                sql = f"""INSERT INTO {vec_table}
                         ({', '.join(columns)})
                         VALUES ({', '.join(placeholders)})"""

                conn.execute(text(sql), vec_data)

                # Commit the transaction
                trans.commit()

                logger.info(f"Successfully inserted package '{package_name}' into SQLite tables ({table_name} and {vec_table})")
                return True

            except Exception as e:
                # Rollback the transaction on error
                trans.rollback()
                logger.error(f"SQLite transaction failed for package '{package_name}': {str(e)}")
                raise
    except Exception as e:
        logger.error(f"Failed to add package '{package_name}' to SQLite database: {str(e)}")
        return False

def add_package_to_db(package_name: str, registry: str, db_manager: DatabaseManager, preprocessor: Preprocessor, embedding_model: str = "local") -> bool:
    try:
        # Get the table name
        table_name = f"typosquat_{registry}_embeddings"
        # Get the table
        table = db_manager.get_embeddings_table(registry)
        # Use preprocessor to compute the embeddings
        components = preprocessor(registry, [package_name])
        embeddings = preprocessor.get_embeddings(registry, components)

        # Validate embedding dimensions based on preprocessor type
        is_openai = embedding_model == "openai"
        if is_openai and len(embeddings['full_embeddings'][0]) != 1536:
            raise ValueError(f"OpenAI embeddings must be 1536-dimensional, got {len(embeddings['full_embeddings'][0])}")
        elif not is_openai and len(embeddings['full_embeddings'][0]) != 300:
            raise ValueError(f"FastText embeddings must be 300-dimensional, got {len(embeddings['full_embeddings'][0])}")

        with db_manager.embeddings_engine.connect() as conn:
            if IS_SQLITE:
                # Use the dedicated SQLite function to handle vector tables properly
                return add_package_to_sqlite(package_name, registry, db_manager, embeddings)
            else:
                # For PostgreSQL, use the existing upsert logic
                data = {
                    'package_name': package_name,
                    'created_at': func.now(),
                    'updated_at': func.now(),
                    'embedding': embeddings['full_embeddings'][0].tolist()
                }
                if 'author_embeddings' in embeddings and embeddings['author_embeddings'][0] is not None:
                    data['author_embedding'] = embeddings['author_embeddings'][0].tolist()
                if 'package_embeddings' in embeddings and embeddings['package_embeddings'][0] is not None:
                    data['package_embedding'] = embeddings['package_embeddings'][0].tolist()

                insert_stmt = insert(table).values([data])
                update_dict = {
                    'embedding': insert_stmt.excluded.embedding,
                    'updated_at': insert_stmt.excluded.updated_at
                }
                if 'author_embedding' in data:
                    update_dict['author_embedding'] = insert_stmt.excluded.author_embedding
                if 'package_embedding' in data:
                    update_dict['package_embedding'] = insert_stmt.excluded.package_embedding

                upsert_stmt = insert_stmt.on_conflict_do_update(
                    index_elements=['package_name'],
                    set_=update_dict
                )
                conn.execute(upsert_stmt)

            conn.commit()

        logger.info(f"Inserted package '{package_name}' into table '{table_name}' in {'SQLite' if IS_SQLITE else 'PostgreSQL'} database.")
        return True
    except Exception as e:
        logger.error(f"Failed to add package '{package_name}' to database: {str(e)}")
        return False

def check_command_squatting(pkg_name: str, registry: str) -> dict:
    """
    Load the list of known commands for the registry and use the FPVerifier's
    verify_command_squatting function to decide if pkg_name is suspicious.
    Returns a dict with candidate info if a match is found; otherwise, None.
    """
    global command_cache
    commands = command_cache.get(registry, [])
    if not commands:
        return None

    for cmd in commands:
        if edit_distance(pkg_name, cmd) < 2:
            return {
                'package_name': pkg_name,
                'legit_command': cmd,
                'similarity': 1.1,  # Higher than other neighbors to prioritize command squatting
                'details': {'edit_distance': edit_distance(pkg_name, cmd)},
                'typo_category': 'Command Squatting',
                'explanation': f"The package name '{pkg_name}' is suspiciously similar to the command '{cmd}'."
            }
    return None


def get_neighbors(package_name: str,
                  registry: str,
                  db_manager: DatabaseManager,
                  preprocessor: Preprocessor,
                  similarity_threshold: float = 0.85,
                  all_packages: List[str] = None,
                  target_packages: List[str] = None,
                  is_sqlite: bool = IS_SQLITE) -> List[Dict[str, Any]]:
    """
    Retrieve neighbor packages for a given package based on embedding similarity.

    When is_sqlite is True, a unified nearest-neighbor search is performed using the
    sqlite-vec virtual table (and the MATCH operator). Otherwise, registry-specific queries
    (designed for PostgreSQL) are used.
    """
    typo_sim = TypoSim()
    neighbors = []

    # Check for command squatting (common to both modes)
    cmd_neighbors = check_command_squatting(package_name, registry)
    if cmd_neighbors is not None:
        neighbors.append(cmd_neighbors)

    # Determine comparison packages from popularity file (or provided target_packages)
    pop_packages = load_pop_packages(registry) if target_packages is None else target_packages
    comparison_packages = target_packages if target_packages is not None else [pkg[0] for pkg in pop_packages]
    if not comparison_packages:
        logger.warning(f"No comparison packages available for {package_name} in registry {registry}")
        return []

    # Retrieve the full embedding for the package (this method works for both backends)
    package_embedding = db_manager.get_embeddings(package_name, registry)
    if package_embedding is None:
        logger.warning(f"No embedding found for package '{package_name}' in registry '{registry}'")
        return []

    # For registry-specific embeddings, retrieve additional embeddings if needed
    provided_author_embedding = None
    provided_package_embedding = None
    if registry in ['maven', 'hf', 'golang']:
        provided_author_embedding = db_manager.get_author_embedding(package_name, registry)
        provided_package_embedding = db_manager.get_package_embedding(package_name, registry)

    # ---------- SQLite branch using unified vector search ----------
    if is_sqlite:
        serialized_emb = sqlite_vec.serialize_float32(package_embedding.tolist())
        # logger.debug(f"Serialized embedding: {serialized_emb}")
        main_table = f"typosquat_{registry}_embeddings"
        vec_table = f"vec_{main_table}"

        # Use registry-specific queries, just like in the PostgreSQL branch
        with db_manager.embeddings_engine.connect() as conn:
            # Verify that sqlite-vec extension is properly loaded
            if not db_manager.verify_extension_loaded(conn):
                raise ValueError("SQLite-Vec extension not loaded")

            if registry == 'maven':
                components = split_package_name(package_name, registry)
                if not components:
                    return []
                artifact_id_value = components['artifact_id']
                group_id_value = components['group_id']
                group_domain_value = components['group_domain']

                # First filter step: find packages with same artifact_id but different group_id
                filter_query = text(f"""
                    SELECT m.rowid, m.package_name,
                           LOWER(SUBSTR(m.package_name, 1, INSTR(m.package_name, ':') - 1)) AS neighbor_group_id,
                           edit_distance(
                               LOWER(SUBSTR(m.package_name, 1, INSTR(m.package_name, ':') - 1)),
                               LOWER(:group_id_value)
                           ) AS group_edit_dist
                    FROM {main_table} m
                    WHERE m.package_name != :package_name
                      AND m.package_name IN ({','.join([':pkg_' + str(i) for i in range(len(comparison_packages))])})
                      AND LOWER(SUBSTR(m.package_name, INSTR(m.package_name, ':') + 1)) = LOWER(:artifact_id_value)
                      AND LOWER(SUBSTR(m.package_name, 1, INSTR(m.package_name, ':') - 1)) != LOWER(:group_id_value)
                """)

                # Prepare parameters dictionary for filter query
                filter_params = {
                    "package_name": package_name,
                    "artifact_id_value": artifact_id_value,
                    "group_id_value": group_id_value
                }
                # Add comparison packages to parameters
                for i, pkg in enumerate(comparison_packages):
                    filter_params[f"pkg_{i}"] = pkg

                filtered_results = conn.execute(filter_query, filter_params).fetchall()
                logger.debug(f"Maven filtered {len(filtered_results)} results: {filtered_results[:10]}")

                if filtered_results:
                    # Get rowids for the second query
                    rowids = [row[0] for row in filtered_results]
                    rowid_placeholders = [f":rowid_{i}" for i in range(len(rowids))]

                    # Second step: similarity calculation
                    similarity_query = text(f"""
                        SELECT m.package_name,
                               vec_distance_cosine(v.author_embedding, :author_emb) as author_similarity,
                               vec_distance_cosine(v.package_embedding, :package_emb) as package_similarity,
                               vec_distance_cosine(v.embedding, :emb) as vector_similarity,
                               edit_distance(
                                   LOWER(SUBSTR(m.package_name, 1, INSTR(m.package_name, ':') - 1)),
                                   LOWER(:group_id_value)
                               ) AS group_edit_dist
                        FROM {vec_table} v
                        JOIN {main_table} m ON m.rowid = v.id
                        WHERE v.id IN ({','.join(rowid_placeholders)})
                        AND (
                            ((author_similarity > 0.9) AND (package_similarity > 0.99) AND group_edit_dist < :edit_dist_threshold)
                            OR (group_edit_dist < 4 AND group_edit_dist < :edit_dist_threshold)
                        )
                        ORDER BY group_edit_dist ASC, package_similarity DESC, author_similarity DESC
                    """)

                    # Prepare parameters dictionary for similarity query
                    similarity_params = {
                        "emb": serialized_emb,
                        "author_emb": sqlite_vec.serialize_float32(provided_author_embedding.tolist()) if provided_author_embedding is not None else None,
                        "package_emb": sqlite_vec.serialize_float32(provided_package_embedding.tolist()) if provided_package_embedding is not None else None,
                        "group_id_value": group_id_value,
                        "edit_dist_threshold": 2*len(group_id_value) // 3
                    }
                    # Add rowids to parameters
                    for i, rowid in enumerate(rowids):
                        similarity_params[f"rowid_{i}"] = rowid

                    # Execute the main query with thresholds
                    results = conn.execute(similarity_query, similarity_params).fetchall()

                    # Filter results by domain (similar to PostgreSQL version)
                    filtered_results = []
                    for row in results:
                        neighbor_components = split_package_name(row[0], registry)
                        if neighbor_components and neighbor_components['group_domain'] != group_domain_value:
                            filtered_results.append(row)
                    results = filtered_results
                else:
                    results = []

            elif registry == 'golang':
                components = split_package_name(package_name, registry)
                if not components:
                    return []
                package_component_value = components[components['package_component']]
                other_component_value = components[components['other_component']].lower()

                # First filter step
                filter_query = text(f"""
                    SELECT m.rowid, m.package_name,
                           LOWER(SUBSTR(m.package_name,
                                       INSTR(m.package_name, '/') + 1,
                                       INSTR(SUBSTR(m.package_name, INSTR(m.package_name, '/') + 1), '/') - 1)) AS neighbor_other_component,
                           edit_distance(
                               LOWER(SUBSTR(m.package_name,
                                          INSTR(m.package_name, '/') + INSTR(SUBSTR(m.package_name, INSTR(m.package_name, '/') + 1), '/') + 1)),
                               LOWER(:package_component_value)
                           ) AS package_edit_dist,
                           edit_distance(
                               LOWER(SUBSTR(m.package_name,
                                          INSTR(m.package_name, '/') + 1,
                                          INSTR(SUBSTR(m.package_name, INSTR(m.package_name, '/') + 1), '/') - 1)),
                               LOWER(:other_component_value)
                           ) AS other_edit_dist
                    FROM {main_table} m
                    WHERE m.package_name != :package_name
                      AND m.package_name IN ({','.join([':pkg_' + str(i) for i in range(len(comparison_packages))])})
                      AND LOWER(SUBSTR(m.package_name,
                                     INSTR(m.package_name, '/') + INSTR(SUBSTR(m.package_name, INSTR(m.package_name, '/') + 1), '/') + 1)) = LOWER(:package_component_value)
                      AND LOWER(SUBSTR(m.package_name,
                                     INSTR(m.package_name, '/') + 1,
                                     INSTR(SUBSTR(m.package_name, INSTR(m.package_name, '/') + 1), '/') - 1)) != LOWER(:other_component_value)
                """)

                # Prepare parameters dictionary for filter query
                filter_params = {
                    "package_name": package_name,
                    "package_component_value": package_component_value,
                    "other_component_value": other_component_value
                }
                # Add comparison packages to parameters
                for i, pkg in enumerate(comparison_packages):
                    filter_params[f"pkg_{i}"] = pkg

                golang_filtered_results = conn.execute(filter_query, filter_params).fetchall()
                logger.debug(f"Golang filtered {len(golang_filtered_results)} results: {golang_filtered_results[:10]}")

                if golang_filtered_results:
                    # Get rowids for the second query
                    rowids = [row[0] for row in golang_filtered_results]
                    rowid_placeholders = [f":rowid_{i}" for i in range(len(rowids))]

                    # Second step: similarity calculation
                    similarity_query = text(f"""
                        SELECT m.package_name,
                               vec_distance_cosine(v.author_embedding, :author_emb) as author_similarity,
                               vec_distance_cosine(v.package_embedding, :package_emb) as package_similarity,
                               vec_distance_cosine(v.embedding, :emb) as vector_similarity,
                               edit_distance(
                                   LOWER(SUBSTR(m.package_name,
                                             INSTR(m.package_name, '/') + 1,
                                             INSTR(SUBSTR(m.package_name, INSTR(m.package_name, '/') + 1), '/') - 1)),
                                   LOWER(:other_component_value)
                               ) AS other_edit_dist
                        FROM {vec_table} v
                        JOIN {main_table} m ON m.rowid = v.id
                        WHERE v.id IN ({','.join(rowid_placeholders)})
                        AND (
                            ((author_similarity > 0.9) AND (package_similarity > 0.99))
                            OR other_edit_dist < 4
                        )
                        ORDER BY other_edit_dist ASC, package_similarity DESC, author_similarity DESC
                    """)

                    # Prepare parameters dictionary for similarity query
                    similarity_params = {
                        "emb": serialized_emb,
                        "author_emb": sqlite_vec.serialize_float32(provided_author_embedding.tolist()) if provided_author_embedding is not None else None,
                        "package_emb": sqlite_vec.serialize_float32(provided_package_embedding.tolist()) if provided_package_embedding is not None else None,
                        "other_component_value": other_component_value
                    }
                    # Add rowids to parameters
                    for i, rowid in enumerate(rowids):
                        similarity_params[f"rowid_{i}"] = rowid

                    results = conn.execute(similarity_query, similarity_params).fetchall()
                else:
                    results = []

            elif registry == 'hf':
                components = split_package_name(package_name, registry)
                if not components:
                    return []
                author_name = components['author_name']
                model_name = components['model_name']

                # First filter: find packages with same model_name but different author
                filter_query = text(f"""
                    SELECT m.rowid, m.package_name,
                           LOWER(SUBSTR(m.package_name, 1, INSTR(m.package_name, '/') - 1)) AS neighbor_author_name,
                           edit_distance(
                               LOWER(SUBSTR(m.package_name, 1, INSTR(m.package_name, '/') - 1)),
                               LOWER(:author_name)
                           ) AS author_edit_dist
                    FROM {main_table} m
                    WHERE m.package_name != :package_name
                      AND m.package_name IN ({','.join([':pkg_' + str(i) for i in range(len(comparison_packages))])})
                      AND LOWER(SUBSTR(m.package_name, INSTR(m.package_name, '/') + 1)) = LOWER(:model_name)
                      AND LOWER(SUBSTR(m.package_name, 1, INSTR(m.package_name, '/') - 1)) != LOWER(:author_name)
                """)

                # Prepare parameters dictionary for filter query
                filter_params = {
                    "package_name": package_name,
                    "author_name": author_name,
                    "model_name": model_name
                }
                # Add comparison packages to parameters
                for i, pkg in enumerate(comparison_packages):
                    filter_params[f"pkg_{i}"] = pkg

                hf_filtered_results = conn.execute(filter_query, filter_params).fetchall()
                logger.debug(f"HuggingFace filtered {len(hf_filtered_results)} results: {hf_filtered_results[:10]}")

                if hf_filtered_results:
                    # Get rowids for the second query
                    rowids = [row[0] for row in hf_filtered_results]
                    rowid_placeholders = [f":rowid_{i}" for i in range(len(rowids))]

                    # Second step: similarity calculation
                    similarity_query = text(f"""
                        SELECT m.package_name,
                               vec_distance_cosine(v.author_embedding, :author_emb) as author_similarity,
                               vec_distance_cosine(v.package_embedding, :package_emb) as package_similarity,
                               vec_distance_cosine(v.embedding, :emb) as vector_similarity,
                               edit_distance(
                                   LOWER(SUBSTR(m.package_name, 1, INSTR(m.package_name, '/') - 1)),
                                   LOWER(:author_name)
                               ) AS author_edit_dist
                        FROM {vec_table} v
                        JOIN {main_table} m ON m.rowid = v.id
                        WHERE v.id IN ({','.join(rowid_placeholders)})
                        AND (
                            ((author_similarity > 0.9) AND (package_similarity > 0.99) AND author_edit_dist < :edit_dist_threshold)
                            OR (author_edit_dist < 4 AND author_edit_dist < :edit_dist_threshold)
                        )
                        ORDER BY author_edit_dist ASC, package_similarity DESC, author_similarity DESC
                    """)

                    # Prepare parameters dictionary for similarity query
                    similarity_params = {
                        "emb": serialized_emb,
                        "author_emb": sqlite_vec.serialize_float32(provided_author_embedding.tolist()) if provided_author_embedding is not None else None,
                        "package_emb": sqlite_vec.serialize_float32(provided_package_embedding.tolist()) if provided_package_embedding is not None else None,
                        "author_name": author_name,
                        "edit_dist_threshold": 2*len(author_name) // 3
                    }

                    # Add rowids to parameters
                    for i, rowid in enumerate(rowids):
                        similarity_params[f"rowid_{i}"] = rowid

                    # Debug query: Output all author similarity calculations before filtering
                    debug_query = text(f"""
                        SELECT
                            m.package_name,
                            vec_distance_cosine(v.author_embedding, :author_emb) as author_similarity,
                            vec_distance_cosine(v.package_embedding, :package_emb) as package_similarity,
                            vec_distance_cosine(v.embedding, :emb) as vector_similarity,
                            edit_distance(
                                LOWER(SUBSTR(m.package_name, 1, INSTR(m.package_name, '/') - 1)),
                                LOWER(:author_name)
                            ) AS author_edit_dist
                        FROM {vec_table} v
                        JOIN {main_table} m ON m.rowid = v.id
                        WHERE v.id IN ({','.join(rowid_placeholders)})
                        ORDER BY author_similarity DESC
                    """)

                    debug_results = conn.execute(debug_query, similarity_params).fetchall()
                    logger.debug(f"HuggingFace author similarities (before threshold filtering):")
                    for row in debug_results[:10]:  # Limit debug output to first 10 rows
                        logger.debug(f"Package: {row[0]}, Author similarity: {row[1]}, Package similarity: {row[2]}, "
                                    f"Vector similarity: {row[3]}, Author edit distance: {row[4]}")

                    results = conn.execute(similarity_query, similarity_params).fetchall()

                    results = conn.execute(similarity_query, similarity_params).fetchall()
                else:
                    results = []

            elif registry in ['pypi', 'npm', 'ruby', 'nuget']:
                # For these registries, use a more general approach similar to PostgreSQL
                query = text(f"""
                    SELECT m.package_name,
                           edit_distance(LOWER(m.package_name), LOWER(:pkg_name)) AS lev_distance,
                           1 - vec_distance_cosine(v.embedding, :emb) AS vector_similarity
                    FROM {vec_table} v
                    JOIN {main_table} m ON m.rowid = v.id
                    WHERE
                        LOWER(m.package_name) != LOWER(:pkg_name)
                        AND m.package_name IN ({','.join([':pkg_' + str(i) for i in range(len(comparison_packages))])})
                        AND v.embedding IS NOT NULL
                    AND (
                        (vector_similarity > 0.9 AND lev_distance < :edit_dist_threshold)
                        OR (lev_distance < 3 AND lev_distance < :edit_dist_threshold)
                    )
                    ORDER BY
                        lev_distance < 3 DESC,
                        lev_distance ASC,
                        vector_similarity DESC
                    LIMIT 10
                """)

                # Prepare parameters dictionary
                params = {
                    "emb": serialized_emb,
                    "pkg_name": package_name,
                    "edit_dist_threshold": 2*len(package_name) / 3
                }
                # Add comparison packages to parameters
                for i, pkg in enumerate(comparison_packages):
                    params[f"pkg_{i}"] = pkg

                results = conn.execute(query, params).fetchall()
            else:
                results = []
                return []

        logger.debug(f"SQLite results: {results[:4]}")

        # Process results to calculate overall similarity using both vector and name-based scores.
        final_neighbors = []
        for row in results:
            neighbor_name = row[0]
            if neighbor_name == package_name:
                continue

            # Extract the vector similarity from the query results
            if registry in ['maven', 'golang', 'hf']:
                vector_similarity = float(row[1])  # author_similarity
            else:
                vector_similarity = float(row[2])  # vector_similarity

            # Calculate name-based similarity using typo_sim
            name_similarity, name_similarity_details = typo_sim(package_name, neighbor_name)
            name_similarity_details.pop('typosquat_similarity', None)

            # Combine all similarity scores
            similarities = [vector_similarity, name_similarity] + list(name_similarity_details.values())
            similarities = [s for s in similarities if not np.isnan(s)]
            if not similarities:
                continue
            max_similarity = max(similarities)

            # Add to final neighbors if it meets the threshold
            if max_similarity >= similarity_threshold:
                final_neighbors.append({
                    'package_name': neighbor_name,
                    'similarity': max_similarity,
                    'details': {
                        'vector_similarity': vector_similarity,
                        **{k: v for k, v in name_similarity_details.items() if not np.isnan(v)}
                    }
                })

        # Append popularity information and sort neighbors
        for neighbor in final_neighbors:
            popularity = next((pkg[1] for pkg in pop_packages_cache[registry] if pkg[0] == neighbor['package_name']), 0)
            neighbor['popularity'] = popularity

        if registry in ['maven', 'golang']:
            final_neighbors.sort(key=lambda x: (x['similarity'], -x['popularity']), reverse=True)
        else:
            final_neighbors.sort(key=lambda x: (x['similarity'], x['popularity']), reverse=True)
        return final_neighbors

    # ---------- PostgreSQL/Non-SQLite branch: registry-specific queries ----------
    else:
        table = db_manager.get_embeddings_table(registry)
        # For these queries we also build a string version of the full embedding
        package_embedding_str = '[' + ','.join(str(x) for x in package_embedding.tolist()) + ']'

        # Prepare string versions of specialized embeddings if available
        author_embedding_str = None
        package_embedding_str_specific = None
        if registry in ['maven', 'hf', 'golang'] and provided_author_embedding is not None:
            author_embedding_str = '[' + ','.join(str(x) for x in provided_author_embedding.tolist()) + ']'
        if registry in ['maven', 'hf', 'golang'] and provided_package_embedding is not None:
            package_embedding_str_specific = '[' + ','.join(str(x) for x in provided_package_embedding.tolist()) + ']'

        with db_manager.embeddings_engine.connect() as conn:
            if registry == 'maven':
                components = split_package_name(package_name, registry)
                if not components:
                    return []
                artifact_id_value = components['artifact_id']
                group_id_value = components['group_id']
                group_domain_value = components['group_domain']
                # --- Maven Query: Name Filtering ---
                maven_name_query = text(f"""
                    WITH computed AS (
                        SELECT
                            n.package_name,
                            LOWER(split_part(n.package_name, ':', 1)) AS neighbor_group_id,
                            levenshtein(
                                LOWER(split_part(n.package_name, ':', 1)),
                                LOWER(:group_id_value)
                            ) AS group_edit_dist
                        FROM {table.name} n
                        WHERE n.package_name != :package_name
                          AND n.package_name = ANY(:comparison_packages)
                          AND LOWER(split_part(n.package_name, ':', 2)) = LOWER(:artifact_id_value)
                          AND LOWER(split_part(n.package_name, ':', 1)) != LOWER(:group_id_value)
                    )
                    SELECT *
                    FROM computed
                """)
                maven_name_params = {
                    "package_name": package_name,
                    "artifact_id_value": artifact_id_value,
                    "group_id_value": group_id_value,
                    "comparison_packages": list(comparison_packages)
                }
                maven_filtered_results = conn.execute(maven_name_query, maven_name_params).fetchall()
                logger.debug(f"Maven filtered {len(maven_filtered_results)} results: {maven_filtered_results[:10]}")
                if maven_filtered_results:
                    filtered_packages = [row.package_name for row in maven_filtered_results]
                    maven_similarity_query = text(f"""
                        SELECT *
                        FROM (
                            SELECT
                                n.package_name,
                                CASE
                                    WHEN author_embedding IS NULL OR :author_embedding IS NULL THEN 0
                                    ELSE 1 - (author_embedding <=> CAST(:author_embedding AS vector))
                                END AS author_similarity,
                                CASE
                                    WHEN package_embedding IS NULL OR :package_embedding IS NULL THEN 0
                                    ELSE 1 - (package_embedding <=> CAST(:package_embedding AS vector))
                                END AS artifact_similarity,
                                levenshtein(
                                    LOWER(split_part(n.package_name, ':', 1)),
                                    LOWER(:group_id_value)
                                ) AS group_edit_dist
                            FROM {table.name} n
                            WHERE n.package_name = ANY(:filtered_packages)
                        ) sub
                        WHERE (author_similarity > 0.9 AND artifact_similarity > 0.99 AND group_edit_dist < :edit_dist_threshold)
                           OR (group_edit_dist < 4 AND group_edit_dist < :edit_dist_threshold)
                        ORDER BY group_edit_dist ASC,
                                 artifact_similarity DESC,
                                 author_similarity DESC
                    """)
                    maven_similarity_params = {
                        "author_embedding": author_embedding_str,
                        "package_embedding": package_embedding_str_specific,
                        "group_id_value": group_id_value,
                        "filtered_packages": filtered_packages,
                        "edit_dist_threshold": 2*len(group_id_value) // 3
                    }
                    results = conn.execute(maven_similarity_query, maven_similarity_params).fetchall()
                    results = [
                        row for row in results
                        if (neighbor_components := split_package_name(row.package_name, registry))
                        and neighbor_components['group_domain'] != group_domain_value
                    ]
                else:
                    results = []
            elif registry == 'golang':
                components = split_package_name(package_name, registry)
                if not components:
                    return []
                package_component_value = components[components['package_component']]
                other_component_value = components[components['other_component']].lower()
                name_filter_query = text("""
                    WITH computed AS (
                        SELECT
                            n.package_name,
                            LOWER(split_part(n.package_name, '/', 2)) AS neighbor_other_component,
                            levenshtein(
                                LOWER(split_part(n.package_name, '/', 3)),
                                LOWER(:package_component_value)
                            ) AS package_edit_dist,
                            levenshtein(
                                LOWER(split_part(n.package_name, '/', 2)),
                                LOWER(:other_component_value)
                            ) AS other_edit_dist
                        FROM typosquat_golang_embeddings n
                        WHERE n.package_name != :package_name
                          AND n.package_name = ANY(:comparison_packages)
                          AND LOWER(split_part(n.package_name, '/', 2)) != LOWER(:other_component_value)
                          AND LOWER(split_part(n.package_name, '/', 3)) = LOWER(:package_component_value)
                    )
                    SELECT *
                    FROM computed
                """)
                name_filter_params = {
                    "package_component_value": package_component_value,
                    "other_component_value": other_component_value,
                    "package_name": package_name,
                    "comparison_packages": list(comparison_packages)
                }
                golang_filtered_results = conn.execute(name_filter_query, name_filter_params).fetchall()
                logger.debug(f"Filtered {len(golang_filtered_results)} results: {golang_filtered_results[:10]}")
                if golang_filtered_results:
                    filtered_packages = [row.package_name for row in golang_filtered_results]
                    similarity_query = text("""
                        SELECT *
                        FROM (
                            SELECT
                                n.package_name,
                                CASE
                                    WHEN author_embedding IS NULL OR :author_embedding IS NULL THEN 0
                                    ELSE 1 - (author_embedding <=> CAST(:author_embedding AS vector))
                                END AS author_similarity,
                                CASE
                                    WHEN package_embedding IS NULL OR :package_embedding IS NULL THEN 0
                                    ELSE 1 - (package_embedding <=> CAST(:package_embedding AS vector))
                                END AS package_similarity,
                                levenshtein(
                                    LOWER(split_part(n.package_name, '/', 2)),
                                    LOWER(:other_component_value)
                                ) AS other_edit_dist
                            FROM typosquat_golang_embeddings n
                            WHERE n.package_name = ANY(:filtered_packages)
                        ) sub
                        WHERE (author_similarity > 0.9 AND package_similarity > 0.99)
                           OR other_edit_dist < 4
                        ORDER BY other_edit_dist ASC,
                                 package_similarity DESC,
                                 author_similarity DESC
                    """) # Intentionally not using edit_dist_threshold here due to namespace squatting
                    similarity_params = {
                        "author_embedding": author_embedding_str,
                        "package_embedding": package_embedding_str_specific,
                        "other_component_value": other_component_value,
                        "filtered_packages": filtered_packages
                    }
                    results = conn.execute(similarity_query, similarity_params).fetchall()
                else:
                    results = []
            elif registry == 'hf':
                components = split_package_name(package_name, registry)
                if not components:
                    return []
                author_name = components['author_name']
                model_name = components['model_name']
                hf_name_query = text(f"""
                    WITH computed AS (
                        SELECT
                            n.package_name,
                            LOWER(split_part(n.package_name, '/', 1)) AS neighbor_author_name,
                            levenshtein(
                                LOWER(split_part(n.package_name, '/', 1)),
                                LOWER(:author_name)
                            ) AS author_edit_dist
                        FROM {table.name} n
                        WHERE n.package_name != :package_name
                          AND n.package_name = ANY(:comparison_packages)
                          AND LOWER(split_part(n.package_name, '/', 2)) = LOWER(:model_name)
                          AND LOWER(split_part(n.package_name, '/', 1)) != LOWER(:author_name)
                    )
                    SELECT *
                    FROM computed
                """)
                hf_name_params = {
                    "package_name": package_name,
                    "author_name": author_name,
                    "model_name": model_name,
                    "comparison_packages": list(comparison_packages)
                }
                hf_filtered_results = conn.execute(hf_name_query, hf_name_params).fetchall()
                logger.debug(f"Huggingface filtered {len(hf_filtered_results)} results: {hf_filtered_results[:10]}")
                if hf_filtered_results:
                    filtered_packages = [row.package_name for row in hf_filtered_results]
                    hf_similarity_query = text(f"""
                        SELECT *
                        FROM (
                            SELECT
                                n.package_name,
                                CASE
                                    WHEN author_embedding IS NULL OR :author_embedding IS NULL THEN 0
                                    ELSE 1 - (author_embedding <=> CAST(:author_embedding AS vector))
                                END AS author_similarity,
                                CASE
                                    WHEN package_embedding IS NULL OR :package_embedding IS NULL THEN 0
                                    ELSE 1 - (package_embedding <=> CAST(:package_embedding AS vector))
                                END AS model_similarity,
                                levenshtein(
                                    LOWER(split_part(n.package_name, '/', 1)),
                                    LOWER(:author_name)
                                ) AS author_edit_dist
                            FROM {table.name} n
                            WHERE n.package_name = ANY(:filtered_packages)
                        ) sub
                        WHERE (author_similarity > 0.9 AND model_similarity > 0.99 AND author_edit_dist < :edit_dist_threshold)
                           OR (author_edit_dist < 4 AND author_edit_dist < :edit_dist_threshold)
                        ORDER BY author_edit_dist ASC,
                                 model_similarity DESC,
                                 author_similarity DESC
                    """)
                    hf_similarity_params = {
                        "author_embedding": author_embedding_str,
                        "package_embedding": package_embedding_str_specific,
                        "author_name": author_name,
                        "filtered_packages": filtered_packages,
                        "edit_dist_threshold": 2*len(author_name) // 3
                    }
                    results = conn.execute(hf_similarity_query, hf_similarity_params).fetchall()
                else:
                    results = []
            elif registry in ['pypi', 'npm', 'ruby', 'nuget']:
                package_embedding = db_manager.get_embeddings(package_name, registry)
                if package_embedding is None:
                    logger.warning(f"No embedding found for package '{package_name}' in registry '{registry}'")
                    return []
                package_embedding_str = '[' + ','.join(str(x) for x in package_embedding.tolist()) + ']'
                query = text(f"""
                    WITH cte AS (
                        SELECT
                            package_name,
                            levenshtein(LOWER(package_name), LOWER(:pkg_name)) AS lev_distance,
                            CASE
                                WHEN embedding IS NULL OR embedding::text ~ 'NaN|Infinity' THEN 0
                                ELSE 1 - (embedding <=> CAST(:embedding AS vector))
                            END AS vector_similarity
                        FROM {table.name}
                        WHERE
                            LOWER(package_name) != LOWER(:pkg_name)
                            AND package_name = ANY(:comparison_packages)
                            AND embedding IS NOT NULL
                    )
                    SELECT
                        package_name,
                        lev_distance,
                        vector_similarity
                    FROM cte
                    WHERE
                        (vector_similarity > 0.8 AND lev_distance < :edit_dist_threshold)
                        OR (lev_distance < 3 AND lev_distance < :edit_dist_threshold)
                    ORDER BY
                        lev_distance < 3 DESC,
                        CASE WHEN lev_distance < 3 THEN lev_distance END ASC NULLS LAST,
                        vector_similarity DESC NULLS LAST
                    LIMIT 10;
                """)
                params = {
                    "embedding": package_embedding_str,
                    "pkg_name": package_name,
                    "comparison_packages": list(comparison_packages),
                    "edit_dist_threshold": 2*len(package_name) // 3
                }
                # try:
                #     query_str = str(query.compile(compile_kwargs={'literal_binds': True}))
                #     params_size = sum(len(str(v)) for v in params.values())
                #     total_size = len(query_str) + params_size
                #     logger.debug(f"Query size: {len(query_str)} bytes, Params size: {params_size} bytes, Total size: {total_size} bytes")
                # except Exception as e:
                #     logger.warning(f"Could not compile query for size check: {e}")
                results = conn.execute(query, params).fetchall()
                logger.debug(f"Results: {results}")
            else:
                results = []
                return []

        # Process results to calculate overall similarity using both vector and name-based scores.
        final_neighbors = []
        for row in results:
            neighbor_name = row.package_name
            if registry in ['maven', 'golang', 'hf']:
                vector_similarity = float(row.author_similarity)
            else:
                vector_similarity = float(row.vector_similarity)
            name_similarity, name_similarity_details = typo_sim(package_name, neighbor_name)
            name_similarity_details.pop('typosquat_similarity', None)
            similarities = [vector_similarity, name_similarity] + list(name_similarity_details.values())
            similarities = [s for s in similarities if not np.isnan(s)]
            if not similarities:
                continue
            max_similarity = max(similarities)
            if max_similarity >= similarity_threshold:
                final_neighbors.append({
                    'package_name': neighbor_name,
                    'similarity': max_similarity,
                    'details': {
                        'vector_similarity': vector_similarity,
                        **{k: v for k, v in name_similarity_details.items() if not np.isnan(v)}
                    }
                })

        # Append popularity info
        for neighbor in final_neighbors:
            neighbor_name = neighbor['package_name']
            popularity = next((pkg[1] for pkg in pop_packages_cache[registry] if pkg[0] == neighbor_name), 0)
            neighbor['popularity'] = popularity

        # Sort neighbors based on similarity (and popularity if applicable)
        if registry in ['maven', 'golang']:
            final_neighbors.sort(key=lambda x: (x['similarity'], -x['popularity']), reverse=True)
        else:
            final_neighbors.sort(key=lambda x: (x['similarity'], x['popularity']), reverse=True)

        return final_neighbors



def load_pop_packages(registry: str) -> List[Tuple[str, int]]:
    global pop_packages_cache
    if registry in pop_packages_cache:
        return pop_packages_cache[registry]

    before_load = get_memory_usage()

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    legit_packages_dir = os.path.join(current_dir, 'legit_packages')
    filename = os.path.join(legit_packages_dir, f'{registry}_legit_packages.csv')

    if not os.path.isfile(filename):
        logger.error(f"Popular packages file not found for registry '{registry}': {filename}")
        pop_packages_cache[registry] = []
        return pop_packages_cache[registry]

    pop_packages = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            if row:
                package_name = row[0].strip()
                popularity = float(row[1].strip())
                # logger.debug(f"Package: {package_name}, Downloads: {popularity}")
                pop_packages.append((package_name, popularity))

    # Sort by downloads in descending order
    pop_packages.sort(key=lambda x: x[1], reverse=True)
    pop_packages_cache[registry] = pop_packages

    after_load = get_memory_usage()
    memory_delta = after_load - before_load
    logger.info(f"Loaded {len(pop_packages)} popular packages for registry '{registry}' from file '{filename}'. " \
               f"Memory usage: {after_load:.2f} MB (+{memory_delta:.2f} MB)")
    return pop_packages

def load_all_packages(registry: str) -> List[str]:
    global all_packages_cache, db_manager
    if registry in all_packages_cache:
        return all_packages_cache[registry]

    before_load = get_memory_usage()
    logger.debug(f"Memory usage before loading all packages for {registry}: {before_load:.2f} MB")

    table = db_manager.get_embeddings_table(registry)
    with db_manager.pg_engine.connect() as conn:
        query = text(f"SELECT DISTINCT package_name FROM {table.name}")
        results = conn.execute(query).fetchall()
        all_packages = [row.package_name for row in results]
        all_packages_cache[registry] = all_packages

        after_load = get_memory_usage()
        memory_delta = after_load - before_load
        logger.info(f"Loaded {len(all_packages)} all packages for registry '{registry}'. " \
                   f"Memory usage: {after_load:.2f} MB (+{memory_delta:.2f} MB)")
        return all_packages

def remove_from_legit_packages(package_name: str, popularity: int, registry: str) -> bool:
    try:
        legit_packages_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'legit_packages')
        csv_path = os.path.join(legit_packages_dir, f'{registry}_legit_packages.csv')
        temp_path = csv_path + '.tmp'
        removed = False

        with open(csv_path, 'r', encoding='utf-8') as input_file, \
             open(temp_path, 'w', encoding='utf-8', newline='') as output_file:
            writer = csv.writer(output_file)
            for row in csv.reader(input_file):
                if row and row[0].strip() != package_name:
                    writer.writerow(row)
                else:
                    removed = True

        # Replace the original file with the filtered one
        if removed:
            os.replace(temp_path, csv_path)
            # Update the cache
            if registry in pop_packages_cache:
                try:
                    pkg_data = (package_name, popularity)
                    pop_packages_cache[registry].remove(pkg_data)
                except ValueError:
                    logger.warning(f"Package {package_name} not found in cache for registry {registry}")
            # Reload the updated packages into memory
            load_pop_packages(registry)
            logger.info(f"Removed {package_name} from {registry} legit packages list")
            return True
        else:
            os.remove(temp_path)
            logger.warning(f"Package {package_name} not found in {registry} legit packages list")
            return False

    except Exception as e:
        logger.error(f"Error removing {package_name} from legit packages: {str(e)}")
        return False


def is_false_positive(typo_doc: dict, legit_doc: dict, registry: str, fp_verifier: FPVerifier) -> Tuple[bool, dict, str, str]:
    return fp_verifier.verify(typo_doc, legit_doc, registry)


def initialize_service(timeout=30):
    try:
        db_manager = DatabaseManager()
        db_manager.ensure_pg_extensions()

        if IS_OPENAI:
            # No need to load FastText model for OpenAI embeddings
            model = None
            preprocessor = Preprocessor(model=None, embedding_model="openai")
        else:
            logger.info(f"Loading FastText model from {MODEL_PATH}")
            # model = FastText.load(MODEL_PATH)
            model = None # TODO: Test only
            preprocessor = Preprocessor(model=model, embedding_model="local")

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'registry_commands.json'), 'r', encoding='utf-8') as f:
            global command_cache
            command_cache = json.load(f)

        fp_verifier = FPVerifier(OPENAI_API_KEY)
        initialized = True
        for registry in REGISTRIES:
            load_pop_packages(registry)
    except Exception as e:
        initialized = False
        logger.error(f"Failed to initialize services: {e}")
        raise
    return db_manager, preprocessor, fp_verifier, initialized


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def deserialize(blob: bytes) -> np.ndarray:
    """Converts a binary blob (from sqlite-vec) into a NumPy array of floats."""
    vector_dim = len(blob) // 4  # 4 bytes per float
    return np.array(struct.unpack(f'{vector_dim}f', blob), dtype=float)
