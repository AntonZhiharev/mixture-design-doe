"""
Quick test of the simplified mixture design streamlit app
"""

import subprocess
import sys
import os

# Run the simplified streamlit app
if __name__ == "__main__":
    app_path = os.path.join("src", "apps", "streamlit_app_simplified.py")
    
    print("Running simplified mixture design app...")
    print(f"App path: {app_path}")
    
    try:
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    except KeyboardInterrupt:
        print("\nApp closed.")
    except Exception as e:
        print(f"Error running app: {e}")
        print("\nTry running manually with:")
        print(f"streamlit run {app_path}")
