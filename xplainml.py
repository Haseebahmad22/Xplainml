#!/usr/bin/env python3
"""
XplainML - Main Entry Point
Coordinates the backend and frontend components
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='XplainML - Interpretable Machine Learning Tool')
    parser.add_argument('--mode', choices=['cli', 'web'], default='cli', 
                       help='Run mode: cli for command line, web for Streamlit dashboard')
    parser.add_argument('--port', type=int, default=8501,
                       help='Port for web dashboard (default: 8501)')
    
    # Pass through all other arguments to the appropriate module
    args, unknown = parser.parse_known_args()
    
    if args.mode == 'web':
        # Launch Streamlit dashboard
        import subprocess
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 
            'frontend/app.py', 
            '--server.port', str(args.port)
        ] + unknown
        subprocess.run(cmd)
    
    else:
        # Run CLI interface
        backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
        cli_script = os.path.join(backend_dir, 'xplainml.py')
        
        # Execute the CLI script with filtered arguments
        import subprocess
        
        # Remove --mode and --port arguments for CLI
        filtered_args = []
        skip_next = False
        for i, arg in enumerate(sys.argv[1:]):
            if skip_next:
                skip_next = False
                continue
            if arg == '--mode':
                skip_next = True
                continue
            if arg == '--port':
                skip_next = True
                continue
            if arg.startswith('--port='):
                continue
            filtered_args.append(arg)
        
        cmd = [sys.executable, cli_script] + filtered_args
        subprocess.run(cmd)

if __name__ == '__main__':
    main()