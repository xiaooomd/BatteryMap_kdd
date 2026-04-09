from pathlib import Path
import runpy

print("Deprecated entrypoint: use `python scripts/preprocess_scripts.py` instead.")
runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "preprocess_scripts.py"), run_name="__main__")
