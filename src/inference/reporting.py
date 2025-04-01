"""
Vulnerability Reporting Module

This module provides functionality for generating reports on detected vulnerabilities,
including detailed information about the vulnerability patterns, affected code regions,
and suggested fixes.
"""

import os
import json
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import networkx as nx
import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)

class VulnerabilityReporter:
    """
    Class for generating reports on detected vulnerabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vulnerability reporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.report_dir = config["inference"]["report_output"]
        self.visualization_dir = config["inference"]["visualization_output"]
        
        # Create output directories if they don't exist
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
    
    def generate_report(
        self,
        scoring_result: Dict[str, Any],
        pattern_match_result: Dict[str, Any],
        code_info: Dict[str, Any] = None,
        filename: str = None
    ) -> str:
        """
        Generate a comprehensive vulnerability report.
        
        Args:
            scoring_result: Result from vulnerability scoring
            pattern_match_result: Result from pattern matching
            code_info: Information about the original code
            filename: Original file name
            
        Returns:
            Path to the generated report file
        """
        # Generate a timestamp for the report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate a report ID
        report_id = f"vuln_report_{timestamp}"
        if filename:
            # Use the filename (without extension) as part of the report ID
            import os
            basename = os.path.basename(filename)
            name_without_ext = os.path.splitext(basename)[0]
            report_id = f"vuln_report_{name_without_ext}_{timestamp}"
        
        # Create the report data structure
        report = {
            "report_id": report_id,
            "timestamp": timestamp,
            "filename": filename,
            "summary": self._generate_summary(scoring_result),
            "vulnerability_details": self._extract_vulnerability_details(scoring_result, pattern_match_result),
            "pattern_matches": pattern_match_result["matches"] if "matches" in pattern_match_result else [],
            "suggested_fixes": scoring_result.get("suggestions", []),
            "code_info": code_info or {}
        }
        
        # Save the report as JSON
        report_path = os.path.join(self.report_dir, f"{report_id}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated vulnerability report: {report_path}")
        
        # Generate visualizations if vulnerabilities were found
        if report["summary"]["is_vulnerable"]:
            self._generate_visualizations(report)
        
        return report_path
    
    def _generate_summary(self, scoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the vulnerability analysis.
        
        Args:
            scoring_result: Result from vulnerability scoring
            
        Returns:
            Summary dictionary
        """
        return {
            "is_vulnerable": scoring_result.get("is_vulnerable", False),
            "vulnerability_score": scoring_result.get("vulnerability_score", 0.0),
            "confidence": scoring_result.get("confidence", 0.0),
            "num_vulnerable_patterns": len(scoring_result.get("top_vulnerable_nodes", [])),
            "time_of_analysis": datetime.datetime.now().isoformat(),
        }
    
    def _extract_vulnerability_details(
        self,
        scoring_result: Dict[str, Any],
        pattern_match_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract detailed information about detected vulnerabilities.
        
        Args:
            scoring_result: Result from vulnerability scoring
            pattern_match_result: Result from pattern matching
            
        Returns:
            List of vulnerability details
        """
        vulnerability_details = []
        
        # Get top vulnerable nodes from scoring result
        top_vulnerable_nodes = scoring_result.get("top_vulnerable_nodes", [])
        
        # Get pattern matches
        pattern_matches = pattern_match_result.get("matches", [])
        
        # Match top vulnerable nodes with pattern match information
        for vuln_node_info in top_vulnerable_nodes:
            pattern_id = vuln_node_info.get("pattern_id")
            
            # Find corresponding pattern match
            match_info = None
            for match in pattern_matches:
                if match.get("pattern_id") == pattern_id:
                    match_info = match
                    break
            
            if match_info is None:
                continue
            
            # Extract vulnerability details
            detail = {
                "pattern_id": pattern_id,
                "vulnerable_nodes": vuln_node_info.get("nodes", []),
                "subgraph": match_info.get("subgraph", {}),
                "repository_matches": match_info.get("repository_matches", []),
                "vulnerability_type": "Unknown"  # In a real implementation, this would be inferred
            }
            
            vulnerability_details.append(detail)
        
        return vulnerability_details
    
    def _generate_visualizations(self, report: Dict[str, Any]) -> None:
        """
        Generate visualizations for the vulnerability report.
        
        Args:
            report: Vulnerability report data
        """
        report_id = report["report_id"]
        
        # Create visualization directory for this report
        viz_dir = os.path.join(self.visualization_dir, report_id)
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate vulnerability subgraph visualizations
        for i, vuln_detail in enumerate(report["vulnerability_details"]):
            if "subgraph" in vuln_detail and vuln_detail["subgraph"]:
                self._visualize_subgraph(vuln_detail["subgraph"], i, viz_dir)
        
        # Generate overall vulnerability score visualization
        self._visualize_vulnerability_score(report["summary"]["vulnerability_score"], viz_dir)
        
        logger.info(f"Generated visualizations in: {viz_dir}")
    
    def _visualize_subgraph(
        self, 
        subgraph_data: Dict[str, Any],
        index: int,
        output_dir: str
    ) -> None:
        """
        Visualize a vulnerability subgraph.
        
        Args:
            subgraph_data: Subgraph data from the report
            index: Index of the vulnerability detail
            output_dir: Output directory for visualizations
        """
        # Create a NetworkX graph from subgraph data
        G = nx.Graph()
        
        # Add nodes
        for node in subgraph_data.get("nodes", []):
            G.add_node(node)
        
        # Add edges
        for edge in subgraph_data.get("edges", []):
            if len(edge) >= 2:
                G.add_edge(edge[0], edge[1])
        
        # Create the visualization
        plt.figure(figsize=(10, 8))
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(G)
        
        # Draw the graph
        nx.draw(
            G, 
            pos, 
            with_labels=True, 
            node_color='lightblue',
            node_size=500, 
            font_size=10, 
            font_weight='bold',
            edge_color='gray'
        )
        
        # Set title
        plt.title(f"Vulnerability Pattern Subgraph {index+1}")
        
        # Save the visualization
        output_path = os.path.join(output_dir, f"vuln_subgraph_{index+1}.png")
        plt.savefig(output_path)
        plt.close()
    
    def _visualize_vulnerability_score(self, score: float, output_dir: str) -> None:
        """
        Visualize the overall vulnerability score.
        
        Args:
            score: Vulnerability score
            output_dir: Output directory for visualizations
        """
        # Create a gauge chart to visualize the vulnerability score
        plt.figure(figsize=(8, 6))
        
        # Define colors based on score
        if score < 0.3:
            color = 'green'
        elif score < 0.7:
            color = 'orange'
        else:
            color = 'red'
        
        # Create a simple gauge chart
        plt.pie(
            [score, 1-score], 
            colors=[color, 'lightgray'],
            startangle=90, 
            counterclock=False
        )
        
        # Add a circle at the center to make it look like a gauge
        circle = plt.Circle((0, 0), 0.6, fc='white')
        plt.gca().add_artist(circle)
        
        # Add the score as text
        plt.text(0, 0, f"{score:.2f}", fontsize=24, ha='center', va='center')
        
        # Add title
        plt.title("Vulnerability Score", fontsize=16)
        
        # Save the visualization
        output_path = os.path.join(output_dir, "vulnerability_score.png")
        plt.savefig(output_path)
        plt.close()

    def generate_html_report(
        self,
        report_path: str
    ) -> str:
        """
        Generate an HTML version of the vulnerability report.
        
        Args:
            report_path: Path to the JSON report file
            
        Returns:
            Path to the generated HTML report
        """
        # Load the JSON report
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        # Generate HTML content
        html_content = self._generate_html_content(report)
        
        # Save the HTML report
        html_path = report_path.replace('.json', '.html')
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {html_path}")
        
        return html_path
    
    def _generate_html_content(self, report: Dict[str, Any]) -> str:
        """
        Generate HTML content for the vulnerability report.
        
        Args:
            report: Vulnerability report data
            
        Returns:
            HTML content as a string
        """
        # This is a simplified HTML template. In a real implementation,
        # you would use a proper templating engine like Jinja2.
        
        # Generate the vulnerability score color
        score = report["summary"]["vulnerability_score"]
        if score < 0.3:
            score_color = 'green'
        elif score < 0.7:
            score_color = 'orange'
        else:
            score_color = 'red'
        
        # Start HTML content
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Vulnerability Report - {report["report_id"]}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .vulnerability {{ background-color: #fff; padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 15px; }}
        .score {{ display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; background-color: {score_color}; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .footer {{ margin-top: 30px; font-size: 12px; color: #777; }}
    </style>
</head>
<body>
    <h1>Vulnerability Detection Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>File:</strong> {report.get("filename", "Unknown")}</p>
        <p><strong>Analysis Date:</strong> {report["summary"]["time_of_analysis"]}</p>
        <p><strong>Vulnerability Status:</strong> {"Vulnerable" if report["summary"]["is_vulnerable"] else "Not Vulnerable"}</p>
        <p><strong>Vulnerability Score:</strong> <span class="score">{report["summary"]["vulnerability_score"]:.2f}</span></p>
        <p><strong>Confidence:</strong> {report["summary"]["confidence"]:.2f}</p>
        <p><strong>Number of Vulnerable Patterns:</strong> {report["summary"]["num_vulnerable_patterns"]}</p>
    </div>
"""
        
        # Add vulnerability details
        if report["vulnerability_details"]:
            html += """
    <h2>Vulnerability Details</h2>
"""
            
            for i, vuln_detail in enumerate(report["vulnerability_details"]):
                html += f"""
    <div class="vulnerability">
        <h3>Vulnerability Pattern {i+1}</h3>
        <p><strong>Pattern ID:</strong> {vuln_detail["pattern_id"]}</p>
        <p><strong>Vulnerability Type:</strong> {vuln_detail.get("vulnerability_type", "Unknown")}</p>
        <p><strong>Affected Nodes:</strong> {len(vuln_detail.get("vulnerable_nodes", []))}</p>
        <p><strong>Subgraph Size:</strong> {vuln_detail.get("subgraph", {}).get("num_nodes", 0)} nodes, {vuln_detail.get("subgraph", {}).get("num_edges", 0)} edges</p>
        
        <h4>Vulnerable Nodes</h4>
        <table>
            <tr>
                <th>Node ID</th>
                <th>Score</th>
            </tr>
"""
                
                for node in vuln_detail.get("vulnerable_nodes", []):
                    html += f"""
            <tr>
                <td>{node.get("node_idx", "")}</td>
                <td>{node.get("score", 0.0):.4f}</td>
            </tr>
"""
                
                html += """
        </table>
    </div>
"""
        
        # Add suggested fixes
        if report["suggested_fixes"]:
            html += """
    <h2>Suggested Fixes</h2>
"""
            
            for i, fix in enumerate(report["suggested_fixes"]):
                html += f"""
    <div class="vulnerability">
        <h3>Fix Suggestion {i+1}</h3>
        <p><strong>Pattern ID:</strong> {fix.get("pattern_id", "")}</p>
        <p><strong>Suggestion Type:</strong> {fix.get("suggestion_type", "")}</p>
        
        <h4>Vulnerable Nodes</h4>
        <table>
            <tr>
                <th>Node ID</th>
                <th>Score</th>
            </tr>
"""
                
                for node in fix.get("vulnerable_nodes", []):
                    html += f"""
            <tr>
                <td>{node.get("node_idx", "")}</td>
                <td>{node.get("score", 0.0):.4f}</td>
            </tr>
"""
                
                html += """
        </table>
        
        <h4>Suggested Replacement Nodes</h4>
        <table>
            <tr>
                <th>Node ID</th>
                <th>Similarity Score</th>
            </tr>
"""
                
                for node in fix.get("suggested_nodes", []):
                    html += f"""
            <tr>
                <td>{node.get("node_idx", "")}</td>
                <td>{node.get("similarity_score", 0.0):.4f}</td>
            </tr>
"""
                
                html += """
        </table>
    </div>
"""
        
        # Finalize HTML content
        html += """
    <div class="footer">
        <p>Generated by Graph-Based Vulnerability Detection System</p>
    </div>
</body>
</html>
"""
        
        return html