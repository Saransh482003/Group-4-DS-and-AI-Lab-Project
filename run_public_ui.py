import os
import sys
import time
import subprocess
import threading
from pyngrok import ngrok

def main():
    print("="*50)
    print("🚀 NGROK PUBLIC UI EXPOSER 🚀")
    print("="*50)
    print("Which UI would you like to run and tunnel publicly?\n")
    print("1: Streamlit App (app/streamlit_app.py)")
    print("2: Gradio App (app/huggingface_app.py)")
    print("="*50)

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        app_name = "Streamlit"
        port = 8501
        command = [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", str(port)]
    elif choice == "2":
        app_name = "Gradio"
        port = 7860  # Default gradio port
        command = [sys.executable, "app/huggingface_app.py"]
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    print(f"\nStarting {app_name} on local port {port}...")

    # Start the application process in the background
    app_process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)

    # Wait a few seconds for the server to spin up
    time.sleep(4)

    print(f"\nStarting ngrok tunnel to port {port}...")
    try:
        # Tunnel the specific port over ngrok
        tunnel = ngrok.connect(port)
        public_url = tunnel.public_url
        print("="*50)
        print(f"✅ SUCCESS: {app_name} is now publicly accessible at:")
        print(f"🌐 {public_url}")
        print("="*50)
        print("\nPress Ctrl+C to stop the server and close the tunnel.")

        # Keep alive
        while app_process.poll() is None:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down ngrok tunnel and application...")
    finally:
        ngrok.kill()
        if app_process.poll() is None:
            app_process.terminate()
            app_process.wait()
        print("Gracefully exited.")

if __name__ == "__main__":
    main()
