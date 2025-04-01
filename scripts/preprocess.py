#!/usr/bin/env python3
"""
Preprocess the vulnerability patch dataset
"""

import os
import argparse
import logging
from vulngraph.data.preprocessor import VulnerabilityPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Preprocess vulnerability patch dataset')
    parser.add_argument('--input_dir', type=str, default='./ab_file',
                        help='Directory containing CVE data')
    parser.add_argument('--output_dir', type=str, default='./processed_data',
                        help='Directory to save processed data')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of samples to process')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = VulnerabilityPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Process the dataset
    logger.info("Starting preprocessing pipeline")
    preprocessor.process_all()
    logger.info("Preprocessing complete")

if __name__ == "__main__":
    main()