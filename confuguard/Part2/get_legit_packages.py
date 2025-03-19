import os
import csv
import argparse
from tqdm import tqdm
from sqlalchemy.sql import text
from loguru import logger

try:
  from config import (
      GCP_PROJECT_ID, GCP_REGION, GCP_INSTANCE_NAME, DB_USER, DB_PASS, DB_NAME, DB_PORT,
      REGISTRIES, ECOSYSTEMS_MAPPING, POP_THRESHOLD,
      HF_GCP_PROJECT_ID, HF_GCP_REGION, HF_GCP_INSTANCE_NAME, HF_DB_USER, HF_DB_PASS, HF_DB_NAME, HF_DB_PORT,
  )
  from utils import init_connection_engine
except:
  from python.typosquat.config import (
      GCP_PROJECT_ID, GCP_REGION, GCP_INSTANCE_NAME, DB_USER, DB_PASS, DB_NAME, DB_PORT,
      REGISTRIES, ECOSYSTEMS_MAPPING, POP_THRESHOLD,
      HF_GCP_PROJECT_ID, HF_GCP_REGION, HF_GCP_INSTANCE_NAME, HF_DB_USER, HF_DB_PASS, HF_DB_NAME, HF_DB_PORT,
  )
  from python.typosquat.utils import init_connection_engine

class QueryLegitPackages:
    def __init__(self):
        # self.hf_engine = init_connection_engine(HF_DB_USER, HF_DB_PASS, HF_DB_NAME, HF_DB_PORT)
        self.engine = init_connection_engine(DB_USER, DB_PASS, DB_NAME, DB_PORT)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.legit_packages_path = os.path.join(current_dir, 'legit_packages')

    def get_legit_packages(self, ecosystems, top_n=None, by_threshold=False):
        results = {ecosystem: self._get_legit_packages_by_ecosystem(ecosystem, top_n, by_threshold) for ecosystem in ecosystems}
        return results

    def _get_legit_packages_by_ecosystem(self, ecosystem, top_n, by_threshold):
        if ecosystem == "npm":
            return self._get_npm_legit_packages(top_n, by_threshold)
        elif ecosystem == "pypi":
            return self._get_pypi_legit_packages(top_n, by_threshold)
        elif ecosystem == "maven":
            return self._get_maven_legit_packages(top_n, by_threshold)
        elif ecosystem == "ruby":
            return self._get_ruby_legit_packages(top_n, by_threshold)
        elif ecosystem == "golang":
            return self._get_golang_legit_packages(top_n, by_threshold)
        elif ecosystem == "hf":
            return self._get_hf_legit_packages(top_n, by_threshold)
        elif ecosystem == "nuget":
            return self._get_nuget_legit_packages(top_n, by_threshold)
        else:
            raise ValueError(f"Invalid ecosystem: {ecosystem}")

    def save_legit_packages(self, ecosystem, data):
        """
        Save the legit packages to a csv file with package name and popularity
        """
        os.makedirs(self.legit_packages_path, exist_ok=True)
        with open(os.path.join(self.legit_packages_path, f"{ecosystem}_legit_packages.csv"), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["package_name", "popularity"])
            for package in data:
                writer.writerow([package[0], package[1]])


    def _get_npm_legit_packages(self, top_n, by_threshold):
        # First get total count
        count_query = text("SELECT COUNT(*) FROM npm_packages")
        with self.engine.connect() as conn:
            total_count = conn.execute(count_query).scalar()
            logger.info(f"Total npm packages in database: {total_count}")

        if by_threshold:
            query = text("""SELECT package_name, recent_weekly_downloads
                            FROM npm_packages
                            WHERE recent_weekly_downloads >= :threshold
                            ORDER BY recent_weekly_downloads DESC""")
            params = {"threshold": POP_THRESHOLD["npm"]}
        else:
            query = text("""SELECT package_name, recent_weekly_downloads
                            FROM npm_packages
                            ORDER BY recent_weekly_downloads DESC
                            LIMIT :top_n""")
            params = {"top_n": top_n}

        with self.engine.connect() as conn:
            result = conn.execute(query, params)
            results = result.fetchall()
            logger.debug(f"Got {len(results)} npm legit packages")
            return results

    def _get_ecosystem_legit_packages(self, ecosystem, top_n, by_threshold):
        # First get total count
        count_query = text("SELECT COUNT(*) FROM ecosystems_packages WHERE registry = :registry")
        with self.engine.connect() as conn:
            total_count = conn.execute(count_query, {"registry": ECOSYSTEMS_MAPPING[ecosystem]}).scalar()
            logger.info(f"Total {ecosystem} packages in database: {total_count}")

        if by_threshold:
            query = text("""SELECT package_name, average_ranking
                            FROM ecosystems_packages
                            WHERE registry = :registry AND average_ranking <= :threshold
                            ORDER BY average_ranking ASC""")
            params = {"registry": ECOSYSTEMS_MAPPING[ecosystem], "threshold": POP_THRESHOLD[ecosystem]}
        else:
            query = text("""SELECT package_name, average_ranking
                            FROM ecosystems_packages
                            WHERE registry = :registry
                            ORDER BY average_ranking ASC
                            LIMIT :top_n""")
            params = {"registry": ECOSYSTEMS_MAPPING[ecosystem], "top_n": top_n}

        with self.engine.connect() as conn:
            result = conn.execute(query, params)
            logger.debug(f"Got {result.rowcount} {ecosystem} legit packages")
            return result.fetchall()

    def _get_maven_legit_packages(self, top_n, by_threshold):
        return self._get_ecosystem_legit_packages('maven', top_n, by_threshold)

    def _get_golang_legit_packages(self, top_n, by_threshold):
        return self._get_ecosystem_legit_packages('golang', top_n, by_threshold)

    def _get_pypi_legit_packages(self, top_n, by_threshold):
        # First get total count
        count_query = text("SELECT COUNT(*) FROM pypi_packages")
        with self.engine.connect() as conn:
            total_count = conn.execute(count_query).scalar()
            logger.info(f"Total PyPI packages in database: {total_count}")

        if by_threshold:
            query = text("""SELECT package_name, recent_weekly_downloads AS downloads
                            FROM pypi_packages
                            WHERE recent_weekly_downloads >= :threshold
                            ORDER BY downloads DESC""")
            params = {"threshold": POP_THRESHOLD["pypi"]}
        else:
            query = text("""SELECT package_name, recent_weekly_downloads AS downloads
                            FROM pypi_packages
                            ORDER BY recent_weekly_downloads DESC
                            LIMIT :top_n""")
            params = {"top_n": top_n}

        with self.engine.connect() as conn:
            result = conn.execute(query, params)
            logger.debug(f"Got {result.rowcount} pypi legit packages")
            return result.fetchall()

    def _get_ruby_legit_packages(self, top_n, by_threshold):
        # First get total count
        count_query = text("SELECT COUNT(*) FROM rubygems_packages")
        with self.engine.connect() as conn:
            total_count = conn.execute(count_query).scalar()
            logger.info(f"Total Ruby packages in database: {total_count}")

        if by_threshold:
            query = text("""SELECT package_name, downloads
                            FROM rubygems_packages
                            WHERE downloads >= :threshold
                            ORDER BY downloads DESC""")
            params = {"threshold": POP_THRESHOLD["ruby"]}
        else:
            query = text("""SELECT package_name, downloads
                            FROM rubygems_packages
                            ORDER BY downloads DESC
                            LIMIT :top_n""")
            params = {"top_n": top_n}

        with self.engine.connect() as conn:
            result = conn.execute(query, params)
            logger.debug(f"Got {result.rowcount} ruby legit packages")
            return result.fetchall()

    def _get_hf_legit_packages(self, top_n, by_threshold):
        # First get total count
        count_query = text("SELECT COUNT(*) FROM metadata")
        with self.hf_engine.connect() as conn:
            total_count = conn.execute(count_query).scalar()
            logger.info(f"Total HF packages in database: {total_count}")

        if by_threshold:
            query = text("""SELECT
                            context_id AS package_name,
                            (downloads #>> '{}')::integer AS downloads
                        FROM metadata
                        WHERE (downloads #>> '{}')::integer >= :threshold
                        ORDER BY (downloads #>> '{}')::integer DESC""")
            params = {"threshold": POP_THRESHOLD["hf"]}
        else:
            query = text("""SELECT
                            context_id AS package_name,
                            (downloads #>> '{}')::integer AS downloads
                        FROM metadata
                        ORDER BY (downloads #>> '{}')::integer DESC
                        LIMIT :top_n""")
            params = {"top_n": top_n}

        with self.hf_engine.connect() as conn:
            result = conn.execute(query, params)
            logger.debug(f"Got {result.rowcount} hf legit packages")
            return result.fetchall()

    def _get_nuget_legit_packages(self, top_n, by_threshold):
        count_query = text("SELECT COUNT(*) FROM partitioned_packages WHERE ecosystem = 'nuget'")
        with self.engine.connect() as conn:
            total_count = conn.execute(count_query).scalar()
            logger.info(f"Total NuGet packages in database: {total_count}")

        if by_threshold:
            query = text("""SELECT
                            package_name,
                            popularity
                        FROM partitioned_packages
                        WHERE ecosystem='nuget' AND popularity >= :threshold
                        ORDER BY popularity DESC""")
            params = {"threshold": POP_THRESHOLD["nuget"]}
        else:
            query = text("""SELECT
                            package_name,
                            popularity
                        FROM partitioned_packages
                        WHERE ecosystem='nuget'
                        ORDER BY popularity DESC
                        LIMIT :top_n""")
            params = {"top_n": top_n}

        with self.engine.connect() as conn:
            result = conn.execute(query, params)
            logger.debug(f"Got {result.rowcount} nuget legit packages")
            return result.fetchall()

def check_table_exists(engine, ecosystem):
    table_name = f"{ecosystem}_pop_packages"
    try:
        with engine.connect() as conn:
            # Check if the table exists
            result = conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}');"))
            table_exists = result.fetchone()[0]
            if not table_exists:
                logger.info(f"Table {table_name} does not exist. Creating table.")
                # Create the table
                conn.execute(text(f"""
                    CREATE TABLE {table_name} (
                        package_name VARCHAR(255) PRIMARY KEY
                    )
                """))
                conn.commit()
                logger.success(f"Table {table_name} created successfully.")
            else:
                logger.info(f"Table {table_name} already exists.")
    except Exception as e:
        logger.error(f"Error checking table {table_name}: {e}")

def push_pop_packages_to_postgres(engine, packages, ecosystem):
    """
    Push legit packages to PostgreSQL.

    Args:
    - engine: SQLAlchemy engine object.
    - packages: List of tuples (package_name, popularity) to insert.
    - ecosystem: The name of the ecosystem (used for the table name).
    """

    # Ensure the table exists before inserting
    check_table_exists(engine, ecosystem)

    table_name = f"{ecosystem}_pop_packages"

    try:
        with engine.connect() as conn:
            for package in tqdm(packages, desc=f"Inserting {ecosystem} packages"):
                package_name = package[0]  # Extract package name from the tuple
                conn.execute(
                    text(f"INSERT INTO {table_name} (package_name) VALUES (:package_name) ON CONFLICT (package_name) DO NOTHING"),
                    {"package_name": package_name}
                )
            conn.commit()
        logger.success(f"Successfully pushed {len(packages)} packages to {table_name}")
    except Exception as e:
        logger.error(f"Error pushing packages to {table_name}: {e}")

def get_all_legit_packages(
    ecosystems=REGISTRIES,
    # ecosystems=["npm", "pypi", "maven", "golang", "ruby"],
    topn = 10000,
    by_threshold = True,
    push_to_postgres=False
):
    query = QueryLegitPackages()
    # ecosystems = ["npm", "pypi", "maven", "golang", "ruby", "hf"]
    # ecosystems = ["hf"]

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the legit_packages directory
    legit_packages_path = os.path.join(current_dir, 'legit_packages')

    legit_packages = {}
    ecosystems_to_query = [ecosystem for ecosystem in ecosystems if not os.path.exists(os.path.join(legit_packages_path, f"{ecosystem}_legit_packages.csv"))]

    if ecosystems_to_query:
        if by_threshold:
            new_legit_packages = query.get_legit_packages(ecosystems_to_query, topn, by_threshold)
        else:
            new_legit_packages = query.get_legit_packages(ecosystems_to_query, topn)

        for ecosystem in ecosystems_to_query:
            if ecosystem in new_legit_packages:
                legit_packages[ecosystem] = new_legit_packages[ecosystem]

                logger.info(f"Saving the legit packages for {ecosystem}")
                query.save_legit_packages(ecosystem, legit_packages[ecosystem])
                logger.success(f"Successfully saved {len(legit_packages[ecosystem])} {ecosystem} legit packages")

    # Load from local files for ecosystems not queried
    for ecosystem in ecosystems:
        ecosystem_file = os.path.join(legit_packages_path, f"{ecosystem}_legit_packages.csv")
        if ecosystem not in ecosystems_to_query and os.path.exists(ecosystem_file):
            with open(ecosystem_file, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header row
                legit_packages[ecosystem] = [row for row in csv_reader]

    logger.success(f"Successfully got all legit packages from {ecosystems}!")

    if push_to_postgres:
        # Pushing the legit packages to postgres
        logger.info("Pushing the legit packages to postgres...")
        engine = init_connection_engine(DB_USER, DB_PASS, DB_NAME)
        for ecosystem in ecosystems:
            push_pop_packages_to_postgres(engine, legit_packages[ecosystem], ecosystem)

        logger.success("Successfully pushed the legit packages to postgres")
    return legit_packages_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get legit packages and optionally push to Postgres.")
    parser.add_argument("--push_to_postgres", action="store_true", help="Push legit packages to Postgres.")
    args = parser.parse_args()

    if args.push_to_postgres:
      get_all_legit_packages(push_to_postgres=True)
      logger.debug(f"Pushed to GCP_PROJECT_ID: {GCP_PROJECT_ID}, DB_NAME: {DB_NAME}")
    else:
      get_all_legit_packages(push_to_postgres=False)
