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
    
    # Support both positional and named arguments for mode
    if len(sys.argv) > 1 and sys.argv[1] in ['web', 'cli']:
        mode = sys.argv[1]
        remaining_args = sys.argv[2:]
    else:
        parser.add_argument('--mode', choices=['cli', 'web'], default='cli', 
                           help='Run mode: cli for command line, web for React dashboard')
        parser.add_argument('--port', type=int, default=8501,
                           help='Port for web dashboard (default: 8501)')
        
        # Pass through all other arguments to the appropriate module
        args, unknown = parser.parse_known_args()
        mode = args.mode
        remaining_args = unknown
    
    if mode == 'web':
        # Launch React frontend and FastAPI backend
        import subprocess
        import threading
        import time
        
        def start_backend():
            backend_cmd = [
                sys.executable, '-m', 'uvicorn', 
                'backend-api.main:app', 
                '--host', '0.0.0.0',
                '--port', '8000',
                '--reload'
            ]
            subprocess.run(backend_cmd)
        
        def start_frontend():
            time.sleep(3)  # Wait for backend to start
            frontend_cmd = ['npm', 'start']
            subprocess.run(frontend_cmd, cwd='frontend')
        
        print(f"🚀 Starting XplainML React application...")
        print("Backend API: http://localhost:8000")
        print("Frontend: http://localhost:3000")
        
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=start_backend)
        backend_thread.daemon = True
        backend_thread.start()
        
        # Start frontend
        start_frontend()
    
    else:
        # Run CLI interface
        backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
        cli_script = os.path.join(backend_dir, 'xplainml.py')
        
        # Execute the CLI script with arguments
        import subprocess
        cmd = [sys.executable, cli_script] + remaining_args
        subprocess.run(cmd)

if __name__ == '__main__':
    main()