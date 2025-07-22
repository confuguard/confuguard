import os
import sys
import time
import numpy as np
from loguru import logger
from sqlalchemy import text
from functools import wraps
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor

sys.path.append('submodules/typomind-release')
from core import detectors


from config import TYPOSQUAT_BEARER_TOKEN, NAMESPACE_ALLOWLIST
from Part4.confusion_search import (
    TypoSim,
    NamespaceAnalyzer,
    add_package_to_db,
    get_neighbors,
    remove_from_legit_packages,
    load_pop_packages,
    load_all_packages,
    is_false_positive,
    initialize_service
)

app = Flask(__name__)

logger.info("Initializing service...")
db_manager, preprocessor, fp_verifier, service_initialized = initialize_service()

if not service_initialized:
    logger.error("Failed to initialize service...")
    raise RuntimeError("Failed to initialize service")

# Create an instance of NamespaceAnalyzer
namespace_analyzer = NamespaceAnalyzer(preprocessor, logger)

logger.success("Service initialized!")
logger.info("Flask app ready to handle requests...")

# Add this after app initialization
# Configure thread pool for handling concurrent requests
executor = ThreadPoolExecutor(max_workers=4)


def require_auth(f):
  @wraps(f)
  def decorated(*args, **kwargs):
      auth_header = request.headers.get('Authorization', '')
      parts = auth_header.split()

      # Check if the Authorization header is in the correct format
      if len(parts) != 2 or parts[0].lower() != 'bearer':
          logger.warning("Invalid Authorization header format.")
          return jsonify({'error': 'Unauthorized'}), 401

      token = parts[1]

      # Validate the token
      if token == TYPOSQUAT_BEARER_TOKEN:
          return f(*args, **kwargs)
      else:
          logger.warning("Invalid token provided.")
          return jsonify({'error': 'Unauthorized'}), 401

  return decorated


@app.route('/', methods=['GET'])
def default_route():
    return jsonify({
        "message": "Welcome to the Typosquat Detection Service",
        "available_routes": [
            "/health",
            "/get_neighbors",
            "/add_package",
            "/similarity",
            "/detect",
            "/detectPop",
            "/get_pop_packages",
            "/fullscan",
            "/verify_fp"
        ]
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not Found",
        "message": "The requested URL was not found on the server.",
        "available_routes": [
            "/health",
            "/get_neighbors",
            "/add_package",
            "/similarity",
            "/detect",
            "/detectPop",
            "/get_pop_packages",
            "/fullscan",
            "/verify_fp"
        ]
    }), 404

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/get_neighbors', methods=['POST'])
@require_auth
def get_neighbors_route():
    data = request.json
    package_name = data.get('package_name')
    registry = data.get('registry')
    similarity_threshold = request.args.get('threshold', default=0.8, type=float)

    if not package_name or not registry:
        return jsonify({"error": "Missing 'package_name' or 'registry' parameter"}), 400

    logger.info(f"Fetching neighbors for {package_name} from {registry}")
    start_time = time.time()

    # Get neighbors
    neighbors = get_neighbors(package_name, registry=registry, db_manager=db_manager,
                            preprocessor=preprocessor, similarity_threshold=similarity_threshold)

    # Filter valid neighbors
    valid_neighbors = []
    for n in neighbors:
        if (not np.isnan(n['similarity'])
            and len(n['package_name']) >= 3
            and (namespace_analyzer.has_suspicious_namespace(package_name, n['package_name'], registry) if registry == 'npm' else True)):
            valid_neighbors.append(n)
            if len(valid_neighbors) == 2:
                break

    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time:.4f} seconds")

    if valid_neighbors:
        return jsonify({
            "package_name": package_name,
            "valid_neighbors": valid_neighbors
        }), 200
    else:
        return jsonify({
            "package_name": package_name,
            "valid_neighbors": []
        }), 200

@app.route('/add_package', methods=['POST'])
@require_auth
def add_package():
    data = request.json
    package_name = data.get('package_name')
    registry = data.get('registry')
    if not package_name or not registry:
        return jsonify({"error": "Missing 'package_name' or 'registry' parameter"}), 400

    logger.info(f"Adding package: {package_name} in {registry}")
    success = add_package_to_db(package_name, registry, db_manager, preprocessor)
    if success:
        return jsonify({"message": f"Package '{package_name}' added"}), 200
    else:
        return jsonify({"error": f"Failed to add '{package_name}'"}), 500

@app.route('/similarity', methods=['POST'])
@require_auth
def similarity():
    data = request.json
    package_name1 = data.get('package_name1')
    package_name2 = data.get('package_name2')
    registry = data.get('registry')
    if not package_name1 or not package_name2 or not registry:
        return jsonify({"error": "Missing parameters"}), 400

    start_time = time.time()
    try:
        typo_sim = TypoSim()
        similarity_score, similarity_details = typo_sim(package_name1, package_name2)

        emd_package1 = db_manager.get_embeddings(package_name1, registry)
        emd_package2 = db_manager.get_embeddings(package_name2, registry)

        if emd_package1 is not None and emd_package2 is not None:
            from service import cosine_similarity_numpy
            vector_similarity = float(cosine_similarity_numpy(emd_package1, emd_package2))
            similarity_details['vector_similarity'] = vector_similarity

        end_time = time.time()
        logger.info(f"Time taken: {end_time - start_time:.4f} seconds")
        return jsonify({"similarity": similarity_score, "details": similarity_details}), 200
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        return jsonify({"error": "Error in similarity"}), 500

@app.route('/verify_fp', methods=['POST'])
@require_auth
def verify_fp():
    data = request.json
    package_name = data.get('package_name')
    registry = data.get('registry')
    neighbor = data.get('neighbor')

    # Validate input format
    if not package_name or not registry or not neighbor:
        return jsonify({"error": "Missing required parameters"}), 400

    if not isinstance(neighbor, dict) or 'package_name' not in neighbor:
        return jsonify({"error": "Invalid neighbor data format"}), 400

    neighbor_name = neighbor.get('package_name')

    # Validate package name formats
    if not isinstance(package_name, str) or not package_name.strip():
        return jsonify({"error": "Invalid package name format"}), 400

    if not isinstance(neighbor_name, str) or not neighbor_name.strip():
        return jsonify({"error": "Invalid neighbor package name format"}), 400

    # Get metadata for both packages
    typo_doc = db_manager.get_pkg_metadata(package_name, registry)
    legit_doc = db_manager.get_pkg_metadata(neighbor_name, registry)

    if not typo_doc or not legit_doc:
        return jsonify({"error": "Missing metadata for packages"}), 400

    # Check if the package is a false positive
    is_fp, metrics, explanation, FP_category = is_false_positive(typo_doc, legit_doc, registry, fp_verifier)

    # Add null check before classification
    try:
        if not is_fp:
            result = detectors.classify_typosquat(package_name, neighbor_name)
            # Handle potential None result
            if result and isinstance(result, dict):
                typo_category = next(iter(result.values()), "Unknown")
            else:
                typo_category = "Unknown"
        else:
            typo_category = None
    except TypeError as e:
        logger.error(f"Classification error: {str(e)}")
        typo_category = "Error"

    # Return the result
    return jsonify({
        "is_false_positive": is_fp,
        "typo_category": typo_category,
        "metrics": metrics,
        "explanation": explanation,
        "FP_category": FP_category
    }), 200

@app.route('/detect', methods=['POST'])
@require_auth
def detect():
    data = request.json
    package_name = data.get('package_name')
    registry = data.get('registry')
    target_packages = data.get('target_packages')

    if not package_name or not registry:
        return jsonify({"error": "Missing parameters"}), 400

    # Submit the detection task to the thread pool

    start_time = time.time()
    future = executor.submit(
        process_detection,
        package_name,
        registry,
        target_packages
    )

    try:
        result = future.result(timeout=60)
        end_time = time.time()
        logger.info(f"Total detection time: {end_time - start_time:.4f} seconds")
        return jsonify(result), 200
    except TimeoutError:
        return jsonify({"error": "Request timed out"}), 408
    except Exception as e:
        logger.error(f"Error processing detection: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Move detection logic to separate function
def process_detection(package_name, registry, target_packages):
    typo_results = []

    # Check if the package's namespace is allowlisted
    if registry in NAMESPACE_ALLOWLIST and NAMESPACE_ALLOWLIST[registry]:
        pkg_namespace = namespace_analyzer._extract_namespace(package_name, registry)
        if pkg_namespace in NAMESPACE_ALLOWLIST[registry]:
            logger.info(f"Namespace {pkg_namespace} is allowlisted. Skipping typosquat detection.")
            return {"typo_results": []}

    # Check if package exists
    table = db_manager.get_embeddings_table(registry)
    with db_manager.embeddings_engine.connect() as conn:
        query = text(f"SELECT 1 FROM {table.name} WHERE package_name = :package_name")
        result = conn.execute(query, {"package_name": package_name}).fetchone()

    if not result:
        logger.info(f"Package '{package_name}' not found. Adding...")
        success = add_package_to_db(package_name, registry, db_manager, preprocessor)
        if not success:
            return {"warning": f"Could not add '{package_name}'"}

    pop_packages = load_pop_packages(registry)
    if target_packages is None:
        pop_packages_list = pop_packages
    else:
        pop_packages_list = target_packages

    is_popular = any(pkg[0] == package_name for pkg in pop_packages)  # Check package name in tuples
    if is_popular and target_packages is None:
        try:
            pkg_index = next(i for i, (pkg, _) in enumerate(pop_packages) if pkg == package_name)
            pkg_popularity = float(pop_packages[pkg_index][1])

            # Different thresholds based on registry
            #   The following block is to remove popular typos from the legit package list.
            if registry in ['npm', 'pypi', 'hf', 'ruby']:
                if pkg_popularity < 10000:
                    # For these registries, use packages with 10x higher popularity
                    target_packages = [pkg[0] for pkg in pop_packages[:pkg_index]
                                     if float(pkg[1]) >= 10 * pkg_popularity]
            elif registry in ['maven', 'golang']:
                if pkg_popularity > 1:
                  # For golang and maven, use packages with 2x higher popularity
                  target_packages = [pkg[0] for pkg in pop_packages[:pkg_index]
                                  if float(pkg[1]) <= pkg_popularity // 2]

            if not target_packages:
                return {"typo_results": typo_results}
        except ValueError:
            return {"typo_results": typo_results}

    start_time = time.time()
    neighbors = get_neighbors(package_name, registry=registry, db_manager=db_manager,
                            preprocessor=preprocessor, target_packages=target_packages)
    end_time = time.time()
    logger.info(f"Time taken to fetch neighbors: {end_time - start_time:.4f} seconds")
    logger.debug(f"Neighbors: {neighbors}")
    # Process regular typosquat candidates
    valid_neighbors = []
    for n in neighbors:
        if (not np.isnan(n['similarity'])
            and len(n['package_name']) >= 3
            and (namespace_analyzer.has_suspicious_namespace(package_name, n['package_name'], registry) if registry == 'npm' else True)):
            valid_neighbors.append(n)
            if len(valid_neighbors) == 2:
                break

    # Check each valid neighbor
    for valid_neighbor in valid_neighbors:
        # Check first neighbor for command squatting
        if valid_neighbor.get('typo_category') == 'Command Squatting':
          try:
              typo_doc = db_manager.get_pkg_metadata(package_name, registry)
              if typo_doc:
                  is_fp, metrics, explanation, FP_category = fp_verifier.verify_command_squatting(
                      package_name,
                      valid_neighbor['legit_command'],
                      typo_doc,
                      registry
                  )
                  if not is_fp:
                      logger.debug(f"Command squatting detected: {package_name} -> {valid_neighbor['legit_command']}")
                      return {"typo_results": [{
                          'metadata_missing': False,
                          'package_name': valid_neighbor['package_name'],
                          'typo_category': 'Command Squatting',
                          'explanation': explanation or valid_neighbor.get('explanation', ''),
                          'FP_category': FP_category
                      }]}
                  else:
                      logger.debug(f"{package_name} is not a command squat.")
                      continue
          except Exception as e:
              logger.error(f"Error processing command squatting candidate: {str(e)}")
              continue


        try:
            typo_category = "Unknown"

            # Try to classify the typosquat
            result = detectors.classify_typosquat(package_name, valid_neighbor['package_name'])
            if result and isinstance(result, dict):
                first_value = next(iter(result.values()), "Unknown")
                typo_category = first_value if first_value else "Unknown"
            else:
                typo_category = "Unknown"

            # Get metadata for FP check
            typo_doc = db_manager.get_pkg_metadata(package_name, registry)
            legit_doc = db_manager.get_pkg_metadata(valid_neighbor['package_name'], registry)

            if not typo_doc or not legit_doc:
                continue  # Skip this neighbor and try next one
            is_fp, metrics, explanation, FP_category = is_false_positive(typo_doc, legit_doc, registry, fp_verifier)
            if not is_fp:
                result_entry = {
                    'metadata_missing': False,
                    'package_name': valid_neighbor['package_name'],
                    'typo_category': str(typo_category),
                    'explanation': explanation,
                    'FP_category': FP_category
                }

                if is_popular:
                    removed = remove_from_legit_packages(package_name, pkg_popularity, registry)
                    result_entry['removed_from_legit'] = removed

                typo_results.append(result_entry)
                logger.info(f"Found typo: {package_name} -> {valid_neighbor['package_name']}")
                return {"typo_results": typo_results}

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing neighbor {valid_neighbor['package_name']}: {error_msg}")
            continue  # Try next neighbor instead of returning error

    return {"typo_results": typo_results}

@app.route('/detectPop', methods=['POST'])
@require_auth
def detect_pop():
    # This endpoint remains as is, or can be simplified.
    data = request.json
    package_name = data.get('package_name')
    registry = data.get('registry')

    if not package_name or not registry:
        return jsonify({"error": "Missing parameters"}), 400

    # Similar logic for ANN search can be placed here if needed.

    return jsonify({"typo_results": []}), 200

@app.route('/get_pop_packages', methods=['GET'])
@require_auth
def get_pop_packages_route():
    registry = request.args.getlist('registry') or ['npm', 'pypi', 'maven', 'golang', 'ruby', 'hf']
    topn = request.args.get('topn', default=10, type=int)
    push_to_postgres = request.args.get('push_to_postgres', default=False, type=bool)

    from service import get_all_legit_packages
    file_paths = get_all_legit_packages(ecosystems=registry, topn=topn, push_to_postgres=push_to_postgres)

    return jsonify({"messages": [f"Success: Top {topn} popular packages saved."]}), 200

@app.route('/fullscan', methods=['POST'])
@require_auth
def fullscan():
    data = request.json
    registry = data.get('registry')
    if not registry:
        return jsonify({"error": "Missing 'registry'" }), 400

    pop_packages = load_pop_packages(registry)
    all_packages = load_all_packages(registry)
    unpop_packages = [pkg for pkg in all_packages if pkg not in pop_packages]

    results = []
    total_packages = len(unpop_packages)
    logger.info(f"Full scan for {registry}, total: {total_packages}")

    from service import get_neighbors
    for i, package in enumerate(unpop_packages):
        if i % 100 == 0:
            logger.info(f"Scanning {i}/{total_packages}")
        neighbors = get_neighbors(package, registry, db_manager, preprocessor)
        for neighbor in neighbors:
            result = detectors.classify_typosquat(package, neighbor['package_name'])
            if result:
                results.append({
                    'package': package,
                    'legit_package': neighbor['package_name'],
                    'similarity': neighbor['similarity'],
                    'typo_result': {str(k): v for k, v in result.items() if v}
                })

    logger.info(f"Full scan done for {registry}, found {len(results)}")
    return jsonify({"message": f"Fullscan done, found {len(results)}", "results": results}), 200

@app.route('/reload_pop_packages', methods=['POST'])
@require_auth
def reload_pop_packages():
    try:
        registries = ['npm', 'pypi', 'maven', 'golang', 'ruby']
        for reg in registries:
            load_pop_packages(reg)
        logger.info("Popular packages reloaded.")
        return jsonify({"message": "Reloaded"}), 200
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": "Failed reload"}), 500

# Add graceful shutdown
def shutdown():
    logger.info("Shutting down thread pool...")
    executor.shutdown(wait=True)

# Development server only
def main():
    logger.info("Starting the development server...")
    # Add command line argument parsing
    port = 5444
    if '--port' in sys.argv:
        try:
            port_index = sys.argv.index('--port') + 1
            port = int(sys.argv[port_index])
        except (IndexError, ValueError):
            logger.warning("Invalid port specified, using default port 5444")

    try:
        app.run(host='0.0.0.0', port=port, threaded=True)
    finally:
        logger.info("Shutting down thread pool...")
        executor.shutdown(wait=True)

if __name__ == '__main__':
    main()
