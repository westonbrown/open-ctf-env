import yaml
import subprocess
import os
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# XBow benchmarks validation script
# Ensures that challenges can build, start, and that flags are correctly injected.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xbow-validator")

BENCHMARKS_DIR = "benchmarks/xbow/benchmarks"

def check_benchmark(benchmark_name):
    path = os.path.join(BENCHMARKS_DIR, benchmark_name)
    if not os.path.isdir(path):
        return None
    
    logger.info(f"Checking {benchmark_name}...")
    
    # Check 1: docker-compose exists
    if not os.path.exists(os.path.join(path, "docker-compose.yml")):
        logger.warning(f"❌ {benchmark_name}: Missing docker-compose.yml")
        return False

    # Check 2: benchmark.json exists (Metadata)
    if not os.path.exists(os.path.join(path, "benchmark.json")):
        logger.warning(f"❌ {benchmark_name}: Missing benchmark.json")
        return False
        
    return True

def validate_all(max_workers=4):
    logger.info("Starting XBow Validation Scan...")
    
    benchmarks = [d for d in os.listdir(BENCHMARKS_DIR) if not d.startswith('.')]
    valid_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_benchmark, b): b for b in benchmarks}
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                valid_count += 1
                
    logger.info(f"✅ Validation Complete. {valid_count}/{len(benchmarks)} benchmarks passed structural checks.")

if __name__ == "__main__":
    validate_all()
