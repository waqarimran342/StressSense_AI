"""
StressSense AI - Application Launcher
Run this file to start the application: python run.py
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'numpy', 'pandas', 
        'plotly', 'matplotlib', 'networkx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} - OK")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("\n📦 Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])
    print("✅ All dependencies installed!")

def run_application():
    """Launch the Streamlit application"""
    app_path = os.path.join("interface", "app.py")
    
    if not os.path.exists(app_path):
        print(f"❌ Error: Cannot find {app_path}")
        print("Make sure you're running from the StressSense_AI directory")
        sys.exit(1)
    
    print("\n🚀 Launching StressSense AI...")
    print("📌 The app will open in your browser automatically")
    print("📌 If not, go to: http://localhost:8501")
    print("📌 Press Ctrl+C to stop the application\n")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
    ])

def main():
    print("=" * 60)
    print("   🧠 StressSense AI - Expert System")
    print("   Stress & Mental Wellbeing Assessment")
    print("=" * 60)
    
    print("\n🔍 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        response = input("Install missing packages? (y/n): ")
        if response.lower() == 'y':
            install_dependencies()
        else:
            print("❌ Cannot run without required packages")
            sys.exit(1)
    else:
        print("✅ All dependencies found!")
    
    run_application()

if __name__ == "__main__":
    main()