#!/usr/bin/env python
"""
Prediction Script

This script runs the vulnerability detection model on new code files to predict
vulnerabilities, identifies patterns, and generates reports with suggested fixes.
"""

import os
import sys
import argparse
import yaml
import torch
import glob
from tqdm import tqdm
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.graphsage import HeteroGraphSAGE
from src.models.pattern_learning import PatternLearningModule
from src.utils.logger import initialize_logging, get_logger
from src.inference.scoring import VulnerabilityScorer
from src.inference.pattern_matching import PatternMatcher
from src.inference.reporting import VulnerabilityReporter
from src.extensions.language_adapters.cpp_adapter import CPPLanguageAdapter
from src.extensions.vulnerability_handlers.buffer_overflow import BufferOverflowHandler
from src.data.graph_processing import GraphProcessor
from src.data.hetero_graph import HeteroGraphBuilder

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict vulnerabilities in code')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--code_path', type=str, required=True,
                      help='Path to code file or directory to analyze')
    parser.add_argument('--output_dir', type=str, default='results/predictions',
                      help='Path to output directory for results')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for prediction (overrides config)')
    parser.add_argument('--threshold', type=float, default=0.7,
                      help='Confidence threshold for vulnerability detection')
    parser.add_argument('--html_report', action='store_true',
                      help='Generate HTML reports')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config_with_args(config, args):
    """Update configuration with command line arguments."""
    if args.device:
        config['training']['device'] = args.device
    
    # Update confidence threshold
    config['inference']['confidence_threshold'] = args.threshold
    
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    return config

def load_model(model_path, config, device):
    """Load the model from a checkpoint."""
    # Create model
    model = HeteroGraphSAGE(config)
    
    # Create pattern learning module
    pattern_module = PatternLearningModule(config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model and pattern module state
    model.load_state_dict(checkpoint['model_state_dict'])
    pattern_module.load_state_dict(checkpoint['pattern_module_state_dict'])
    
    # Move to device
    model.to(device)
    pattern_module.to(device)
    
    # Set to evaluation mode
    model.eval()
    pattern_module.eval()
    
    return model, pattern_module, checkpoint

def collect_code_files(code_path):
    """Collect code files to analyze."""
    code_files = []
    
    if os.path.isfile(code_path):
        # Single file
        code_files.append(code_path)
    elif os.path.isdir(code_path):
        # Directory - find all supported files
        for ext in ['c', 'cpp', 'cc', 'h', 'hpp']:
            code_files.extend(glob.glob(os.path.join(code_path, f'**/*.{ext}'), recursive=True))
    else:
        logger.error(f"Code path {code_path} does not exist")
        return []
    
    return code_files

def create_components(model, pattern_module, config, device):
    """Create the necessary components for prediction."""
    # Create language adapter
    language_adapter = CPPLanguageAdapter(config)
    
    # Create vulnerability handlers
    buffer_overflow_handler = BufferOverflowHandler(config)
    
    # Create graph processor
    graph_processor = GraphProcessor(config)
    
    # Create hetero graph builder
    hetero_graph_builder = HeteroGraphBuilder(config)
    
    # Create vulnerability scorer
    vulnerability_scorer = VulnerabilityScorer(model, pattern_module, config)
    
    # Create pattern matcher
    pattern_matcher = PatternMatcher(pattern_module, config)
    
    # Create vulnerability reporter
    vulnerability_reporter = VulnerabilityReporter(config)
    
    return {
        'language_adapter': language_adapter,
        'buffer_overflow_handler': buffer_overflow_handler,
        'graph_processor': graph_processor,
        'hetero_graph_builder': hetero_graph_builder,
        'vulnerability_scorer': vulnerability_scorer,
        'pattern_matcher': pattern_matcher,
        'vulnerability_reporter': vulnerability_reporter
    }

def process_file(file_path, components, model, pattern_module, device, config, output_dir, html_report=False):
    """Process a single code file."""
    logger.info(f"Processing file: {file_path}")
    
    # Get components
    language_adapter = components['language_adapter']
    graph_processor = components['graph_processor']
    vulnerability_scorer = components['vulnerability_scorer']
    pattern_matcher = components['pattern_matcher']
    vulnerability_reporter = components['vulnerability_reporter']
    
    # Check if file is supported
    if not language_adapter.supports_file(file_path):
        logger.warning(f"Unsupported file: {file_path}")
        return None
    
    try:
        # Parse file into graph
        logger.info(f"Parsing file: {file_path}")
        nx_graph = language_adapter.parse_file(file_path)
        
        # Convert to HeteroData
        logger.info(f"Converting to heterogeneous graph")
        hetero_graph = graph_processor._convert_to_hetero(nx_graph, is_vulnerable=False)
        
        # Move to device
        hetero_graph = hetero_graph.to(device)
        
        # Get node embeddings
        with torch.no_grad():
            node_embeddings = model.get_node_embeddings(hetero_graph)
        
        # Score graph for vulnerabilities
        logger.info(f"Scoring graph for vulnerabilities")
        scoring_result = vulnerability_scorer.score_graph(hetero_graph)
        
        # Match patterns
        logger.info(f"Matching vulnerability patterns")
        pattern_match_result = pattern_matcher.match_patterns(hetero_graph, node_embeddings)
        
        # Suggest fixes
        logger.info(f"Suggesting fixes")
        fix_result = vulnerability_scorer.suggest_fix(hetero_graph, scoring_result)
        
        # Extract code information
        code_info = {
            'file_path': file_path,
            'language': language_adapter.language_name,
            'nodes': len(nx_graph),
            'edges': nx_graph.number_of_edges(),
            'functions': len(language_adapter.extract_function_nodes(nx_graph)),
            'variables': len(language_adapter.extract_variable_nodes(nx_graph))
        }
        
        # Generate report
        logger.info(f"Generating vulnerability report")
        report_path = vulnerability_reporter.generate_report(
            scoring_result, pattern_match_result, code_info, file_path
        )
        
        # Generate HTML report if requested
        if html_report:
            html_path = vulnerability_reporter.generate_html_report(report_path)
            logger.info(f"Generated HTML report: {html_path}")
        
        # Return results
        return {
            'file_path': file_path,
            'vulnerability_score': scoring_result['vulnerability_score'],
            'is_vulnerable': scoring_result['is_vulnerable'],
            'pattern_matches': len(pattern_match_result.get('matches', [])),
            'report_path': report_path
        }
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return {
            'file_path': file_path,
            'error': str(e)
        }

def generate_summary_report(results, output_dir):
    """Generate a summary report of all files analyzed."""
    vulnerable_files = []
    clean_files = []
    error_files = []
    
    for result in results:
        if 'error' in result:
            error_files.append(result)
        elif result.get('is_vulnerable', False):
            vulnerable_files.append(result)
        else:
            clean_files.append(result)
    
    # Generate summary
    summary = f"""
Vulnerability Detection Summary Report
-------------------------------------

Total files analyzed: {len(results)}
Vulnerable files: {len(vulnerable_files)}
Clean files: {len(clean_files)}
Files with errors: {len(error_files)}

Vulnerable Files:
"""
    
    # Sort vulnerable files by score
    vulnerable_files.sort(key=lambda x: x['vulnerability_score'], reverse=True)
    
    for i, vuln_file in enumerate(vulnerable_files):
        file_name = os.path.basename(vuln_file['file_path'])
        summary += f"{i+1}. {file_name} - Score: {vuln_file['vulnerability_score']:.2f}, Patterns: {vuln_file['pattern_matches']}\n"
    
    if error_files:
        summary += "\nFiles with Errors:\n"
        for i, error_file in enumerate(error_files):
            file_name = os.path.basename(error_file['file_path'])
            summary += f"{i+1}. {file_name} - Error: {error_file.get('error', 'Unknown error')}\n"
    
    # Write summary to file
    summary_path = os.path.join(output_dir, 'summary_report.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    logger.info(f"Generated summary report: {summary_path}")
    
    return summary_path

def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    # Initialize logging
    initialize_logging()
    
    # Set device
    device = torch.device(config['training']['device'] 
                        if torch.cuda.is_available() and config['training']['device'] == 'cuda' 
                        else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model, pattern_module, checkpoint = load_model(args.model_path, config, device)
    
    # Collect code files
    logger.info(f"Collecting code files from {args.code_path}")
    code_files = collect_code_files(args.code_path)
    
    if not code_files:
        logger.error(f"No supported code files found at {args.code_path}")
        return
    
    logger.info(f"Found {len(code_files)} code files to analyze")
    
    # Create components
    logger.info("Creating prediction components")
    components = create_components(model, pattern_module, config, device)
    
    # Process each file
    logger.info("Starting prediction")
    results = []
    
    for file_path in tqdm(code_files, desc="Analyzing files"):
        result = process_file(
            file_path, components, model, pattern_module, device, config, 
            args.output_dir, args.html_report
        )
        
        if result:
            results.append(result)
    
    # Generate summary report
    logger.info("Generating summary report")
    summary_path = generate_summary_report(results, args.output_dir)
    
    # Print summary
    vulnerable_count = sum(1 for r in results if r.get('is_vulnerable', False))
    logger.info(f"Prediction complete. {vulnerable_count}/{len(results)} files flagged with potential vulnerabilities.")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info(f"Summary report: {summary_path}")

if __name__ == '__main__':
    # Get global logger
    logger = get_logger(__name__)
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise