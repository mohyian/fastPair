# scripts/preprocess_graphspd.py
import os
import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.graphspd_preprocessor import GraphSPDPreprocessor

def main():
    parser = argparse.ArgumentParser(description='Preprocess GraphSPD data for PatchPairVul')
    parser.add_argument('--input_dir', type=str, default='./ab_file',
                        help='Directory containing GraphSPD data')
    parser.add_argument('--output_dir', type=str, default='./processed_data',
                        help='Directory to save processed data')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of samples to process')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = GraphSPDPreprocessor(
        root_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Process all data
    preprocessor.process_all()

if __name__ == "__main__":
    main()