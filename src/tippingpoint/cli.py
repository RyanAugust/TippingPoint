import argparse
import sys
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Tipping Point CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch the interactive Streamlit dashboard")

    args = parser.parse_args()

    if args.command == "dashboard":
        launch_dashboard()
    else:
        parser.print_help()

def launch_dashboard():
    """Launches the Streamlit dashboard."""
    try:
        import streamlit
    except ImportError:
        print("Error: Streamlit is not installed. Please install it with 'pip install streamlit'.")
        sys.exit(1)

    # Get the path to dashboard.py
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")

    print(f"Launching dashboard from {dashboard_path}...")
    subprocess.run(["streamlit", "run", dashboard_path])

if __name__ == "__main__":
    main()
