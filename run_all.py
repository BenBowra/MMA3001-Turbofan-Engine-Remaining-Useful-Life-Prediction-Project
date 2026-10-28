"""Run the entire FD001 analysis using Python scripts, without Jupyter.

Examples:
    python run_all.py --archive "C:/Downloads/CMAPSSData.zip"
    python run_all.py     # if the FD001 raw files are already installed

The NASA RUL labels are used only in Stage 4; all model choices are frozen
from Stage 3 beforehand. This re-creates generated figures and CSV results.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def execute(relative_script: str, *args: str) -> None:
    """Execute one stage in a subprocess using the current Python environment."""
    script = ROOT / relative_script
    print(f"\n{'=' * 65}\nRunning {relative_script}\n{'=' * 65}", flush=True)
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'  # Save figures without opening plot windows.
    # Avoid excessive BLAS threading for this relatively small dataset.
    threads = env.get('MMA3001_BLAS_THREADS', '1')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = threads
    subprocess.run([sys.executable, str(script), *args], cwd=ROOT, env=env, check=True)


def main() -> None:
    """Execute all four stages in order, optionally extracting the original NASA ZIP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path,
        help='NASA ZIP archive, either outer repository ZIP or CMAPSSData.zip')
    args = parser.parse_args()

    if args.archive is not None:
        execute('scripts/setup_dataset.py', str(args.archive.expanduser().resolve()))
    elif not (ROOT/'stage1/data/train_FD001.txt').is_file():
        parser.error('FD001 data not found. Pass --archive path/to/CMAPSSData.zip')

    execute('stage1/01_Data_Exploration.py')
    execute('stage2/02_Data_Preprocessing.py')
    execute('scripts/sync_stage3_data.py')
    execute('stage3/03_Model_Training.py')

    decision = ROOT/'stage3/results/stage3_model_decision.json'
    if not decision.is_file():
        raise FileNotFoundError(f'Stage 3 model decision was not generated: {decision}')
    stage4_data = ROOT/'stage4/data'
    stage4_data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(decision, stage4_data/'stage3_model_decision.json')
    execute('stage4/04_Final_Evaluation.py')
    print('\nAll four stages completed. See stage4/results/ for official test metrics.', flush=True)


if __name__ == '__main__':
    main()
