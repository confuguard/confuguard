import csv
import requests
import matplotlib.pyplot as plt
import pandas as pd
import dateutil.parser

# -----------------------------------------------------
# Helper functions to fetch release data from registries
# -----------------------------------------------------

def get_pypi_releases(package_name):
    """
    Return a list of (version, upload_time) tuples for a package on PyPI.
    Earliest to latest, or reverse it if desired.
    
    Skipped in main() because PyPI timestamps aren't consistently available.
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        releases_data = data.get("releases", {})
        version_time_pairs = []
        for version, rel_info in releases_data.items():
            if rel_info:  # might be empty array sometimes
                # pick the earliest upload_time among files in that version
                upload_times = [f["upload_time"] for f in rel_info if "upload_time" in f]
                if upload_times:
                    upload_time = min(upload_times)
                    version_time_pairs.append((version, upload_time))
        # Sort by upload time
        version_time_pairs.sort(key=lambda x: x[1])
        return version_time_pairs
    except Exception as e:
        print(f"[!] PyPI error for {package_name}: {e}")
        return []

def get_npm_releases(package_name):
    """
    Return a list of (version, time) tuples for a package on npm.
    
    The "time" object in npm metadata looks like:
      "time": {
        "modified": "...",
        "created": "...",
        "1.0.0": "2017-08-01T23:31:50.277Z",
        ...
      }
    We'll ignore the special keys 'modified' and 'created'.
    """
    url = f"https://registry.npmjs.org/{package_name}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        time_data = data.get("time", {})
        version_time_pairs = []
        for version, upload_time in time_data.items():
            if version not in ("modified", "created"):
                version_time_pairs.append((version, upload_time))
        # Sort by upload_time
        version_time_pairs.sort(key=lambda x: x[1])
        return version_time_pairs
    except Exception as e:
        print(f"[!] NPM error for {package_name}: {e}")
        return []

def get_rubygems_releases(package_name):
    """
    Return a list of (version, created_at) for a package on RubyGems.
    
    Skipped in main() because RubyGems timestamps aren't consistently available.
    """
    url = f"https://rubygems.org/api/v1/versions/{package_name}.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        version_time_pairs = [
            (str(item["number"]), item["created_at"]) for item in data
        ]
        # Sort by created_at
        version_time_pairs.sort(key=lambda x: x[1])
        return version_time_pairs
    except Exception as e:
        print(f"[!] RubyGems error for {package_name}: {e}")
        return []

def get_releases(package_name, ecosystem):
    """
    Wrapper to pick the right function for the ecosystem.
    """
    # If needed in the future, you can re-enable PyPI / RubyGems.
    if ecosystem.lower() == "npm":
        return get_npm_releases(package_name)
    else:
        # For now, skip others since we lack reliable timestamps.
        return []

# -----------------------------------------------------
# Main logic
# -----------------------------------------------------

def main():
    # Load CSV with pandas (adjust delimiter if needed)
    df = pd.read_csv("./datasets/NeupaneDB_real_malware.csv", sep="\t")

    # We'll store the differences in release counts
    differences = []

    for idx, row in df.iterrows():
        malicious_pkg = row["Malicious Package Name"]
        original_pkg  = row["Original Package Name"]
        ecosystem     = row["Ecosystem"].lower()

        # Only proceed if npm
        if ecosystem != "npm":
            # Skip PyPI or RubyGems packages
            print(f"Skipping {malicious_pkg} (ecosystem: {ecosystem}) - no consistent timestamps available.")
            continue

        # Fetch releases for original and malicious
        original_releases = get_releases(original_pkg, ecosystem)
        malicious_releases = get_releases(malicious_pkg, ecosystem)

        # If we have zero releases for the malicious, skip
        if not malicious_releases:
            print(f"No releases found for malicious package '{malicious_pkg}'. Skipping.")
            continue

        # In a real scenario, you would identify which release introduced the malware.
        # For simplicity, let's assume the *first release* is malicious:
        malicious_introduction_index = 0
        malicious_date_str = malicious_releases[malicious_introduction_index][1]

        # Convert to datetime object
        try:
            malicious_dt = dateutil.parser.parse(malicious_date_str)
        except Exception as e:
            print(f"Could not parse malicious release date for {malicious_pkg}: {e}")
            continue

        # Count how many original releases were published *before* that date
        count_original_before = 0
        for (ver, odate_str) in original_releases:
            try:
                odt = dateutil.parser.parse(odate_str)
                if odt < malicious_dt:
                    count_original_before += 1
            except:
                pass

        # Append that difference to our results
        differences.append(count_original_before)

    # Now plot a bar chart of those differences
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(differences)), differences, color='red')
    plt.title("Number of Original (npm) Releases Before Malicious Introduction")
    plt.xlabel("Malicious Package Index")
    plt.ylabel("Count of Original Releases Before Malicious Release")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
