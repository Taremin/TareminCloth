import sys
import subprocess
from pathlib import Path
from tools.blender_manager import resolve_blender

def main():
    blender_exe = resolve_blender()
    cmd = [
        str(blender_exe),
        "--background",
        "--factory-startup",
        "--python",
        "benchmarks/benchmark_tuning_options.py",
    ]
    subprocess.run(cmd, cwd=str(Path(__file__).parent))

if __name__ == "__main__":
    main()
