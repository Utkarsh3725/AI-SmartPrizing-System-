"""
==============================================================================
AI-Driven Smart Pricing System for Theme Parks
Main Entry Point
Team B | Utkarsh Arya | SRM University | Deloitte Digital Camp 2026
==============================================================================

Usage:
    python main.py          # Generate data + Train model
    streamlit run dashboard/app.py   # Launch dashboard
"""

import subprocess
import sys
import os
import time


def print_banner():
    """Print the project banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🎢  AI-DRIVEN SMART PRICING SYSTEM                      ║
║         FOR THEME PARKS                                      ║
║                                                              ║
║     Deloitte Digital Camp 2026                               ║
║     Team B | Utkarsh Arya | SRM University                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_step(step_name, script_path):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"  Step: {step_name}")
    print(f"  Script: {script_path}")
    print(f"{'='*60}\n")
    
    # Run the script using the same Python interpreter
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n❌ Error in {step_name}! Return code: {result.returncode}")
        sys.exit(1)
    
    print(f"\n✅ {step_name} completed successfully!")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print("🔍 Checking dependencies...")
    
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
    }
    
    optional = {
        'xgboost': 'xgboost',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} — MISSING")
            missing.append(package)
    
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ⚠️  {package} — optional (will use sklearn fallback)")
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        
        user_input = input("\nWould you like to install them now? (y/n): ").strip().lower()
        if user_input == 'y':
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
            print("✅ Packages installed!")
        else:
            print("⚠️  Please install missing packages and try again.")
            sys.exit(1)
    else:
        print("\n✅ All required dependencies are installed!")
    
    print()


def main():
    """Main execution pipeline."""
    print_banner()
    
    start_time = time.time()
    
    # Check dependencies
    check_dependencies()
    
    # Get project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Generate Data
    data_script = os.path.join(project_root, 'data', 'generate_data.py')
    run_step("📊 Data Generation", data_script)
    
    # Step 2: Train Model
    model_script = os.path.join(project_root, 'model', 'train_model.py')
    run_step("🧠 Model Training", model_script)
    
    elapsed = time.time() - start_time
    
    # Final summary
    print(f"""
{'='*60}
  ✅ SETUP COMPLETE!
{'='*60}

  ⏱️  Total time: {elapsed:.1f} seconds
  
  📊 Data generated:     data/theme_park_data.csv
  🗄️  Database created:   data/theme_park.db
  🧠 Model saved:        model/pricing_model.pkl
  📈 Charts saved:       model/feature_importance.png
                         model/prediction_comparison.png
  
  🚀 To launch the dashboard, run:

     streamlit run dashboard/app.py

{'='*60}
  🎢 AI Smart Pricing System — Ready to optimize revenue!
  Team B | Utkarsh Arya | SRM University
  Deloitte Digital Camp 2026
{'='*60}
""")


if __name__ == "__main__":
    main()
