#!/bin/bash

set -euo pipefail

# Run get_legit_packages.py with retries
echo "Running get_legit_packages.py..."
TIMEOUT=60  # 1 minute timeout
START_TIME=$SECONDS

while true; do
    python -m confuguard.Part2.get_legit_packages

    # Check if legit_packages directory has files
    if [ -n "$(ls -A confuguard/Part2/legit_packages 2>/dev/null)" ]; then
        echo "Successfully populated legit_packages directory."
        break
    fi

    # Check if we've exceeded timeout
    if [ $(($SECONDS - $START_TIME)) -ge $TIMEOUT ]; then
        echo "Error: Failed to populate legit_packages directory after 1 minute. Exiting."
        exit 1
    fi

    echo "legit_packages directory is empty, retrying in 5 seconds..."
    sleep 5
done
echo "Completed get_legit_packages.py."

# Run update_pop_pkgs.py in the background
echo "Starting update_pop_pkgs.py in the background..."
python confuguard/Part2/update_pop_pkgs.py &
PKG_UPDATE_PID=$!

# Get number of threads from env var or use default
THREAD_COUNT=${GUNICORN_THREAD_NUM:-4}

# Start the Flask app using Gunicorn with gthread workers in the background
echo "Starting Flask app with Gunicorn using gthread workers..."
gunicorn --bind 0.0.0.0:5444 --worker-class=gthread --workers=1 --threads=$THREAD_COUNT confuguard.app:app &
GUNICORN_PID=$!

# Wait for both background processes to finish
wait $PKG_UPDATE_PID $GUNICORN_PID
