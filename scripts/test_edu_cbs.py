#!/usr/bin/env python
"""Test CBS validation for Edu-Primary with fixed plotting."""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.cbs_validation import run_cbs_validation

try:
    logger.info("Running CBS validation for Edu-Primary")
    artifacts = run_cbs_validation(
        'results/Edu-Primary',
        output_dir='results/Edu-Primary/cbs_validation',
        mc_iterations=2000,
        random_state=42
    )
    logger.info(f"CBS validation completed successfully for Edu-Primary")
    logger.info(f"Artifacts: {artifacts}")
except Exception as e:
    logger.exception(f"CBS validation failed for Edu-Primary: {e}")
    sys.exit(1)
