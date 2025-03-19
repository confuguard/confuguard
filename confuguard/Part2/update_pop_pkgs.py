try:
    from python.typosquat.get_legit_packages import get_all_legit_packages
except:
    from get_legit_packages import get_all_legit_packages

import requests
import schedule
import time
from loguru import logger

def update_pop_packages():
    logger.info("Starting weekly update of popular packages")

    sources = ['npm', 'pypi', 'maven', 'golang', 'ruby', 'nuget']


    try:
        get_all_legit_packages(ecosystems=sources, push_to_postgres=False)
        logger.info(f"Successfully updated popular packages for {sources}")
    except requests.RequestException as e:
        logger.error(f"Failed to update popular packages for {sources}: {str(e)}")

    logger.info("Weekly update of popular packages completed")

def run_scheduler():
    schedule.every().monday.at("00:00").do(update_pop_packages)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    logger.info("Starting weekly popular packages update scheduler")
    run_scheduler()
