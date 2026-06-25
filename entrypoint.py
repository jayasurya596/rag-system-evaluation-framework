import os
import sys
import time
import subprocess
import signal
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (Launcher) %(message)s')
logger = logging.getLogger("launcher")

def main():
    # Render maps the listening port to the PORT environment variable.
    port = os.environ.get("PORT", "8501")
    
    logger.info("Starting production service orchestrator...")
    
    # 1. Start FastAPI backend on internal port 8000
    logger.info("Launching FastAPI backend on http://127.0.0.1:8000 ...")
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "backend.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ]
    backend_proc = subprocess.Popen(backend_cmd)
    
    # 2. Start Streamlit frontend on the public container port
    logger.info(f"Launching Streamlit frontend on http://0.0.0.0:{port} ...")
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run", "frontend/dashboard.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false"
    ]
    frontend_proc = subprocess.Popen(frontend_cmd)
    
    def kill_processes(signum, frame):
        logger.info("Received termination signal. Shutting down subprocesses...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        logger.info("All processes stopped. Exiting.")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, kill_processes)
    signal.signal(signal.SIGTERM, kill_processes)
    
    # Keep monitoring the processes
    while True:
        backend_status = backend_proc.poll()
        frontend_status = frontend_proc.poll()
        
        if backend_status is not None:
            logger.error(f"FastAPI backend exited unexpectedly with status {backend_status}. Shutting down.")
            frontend_proc.terminate()
            frontend_proc.wait()
            sys.exit(backend_status or 1)
            
        if frontend_status is not None:
            logger.error(f"Streamlit frontend exited unexpectedly with status {frontend_status}. Shutting down.")
            backend_proc.terminate()
            backend_proc.wait()
            sys.exit(frontend_status or 1)
            
        time.sleep(2)

if __name__ == "__main__":
    main()
