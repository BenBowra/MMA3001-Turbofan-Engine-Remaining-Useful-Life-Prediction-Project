"""Copy Stage 2 preprocessing outputs into the Stage 3 input directory.
Run after completing stage2/02_Data_Preprocessing.py.
"""
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]
src = ROOT/'stage2'/'processed'
dst = ROOT/'stage3'/'data'
needed=('feature_manifest.json','train_engine_rows.csv','validation_engine_rows.csv','validation_snapshots.csv')
missing=[n for n in needed if not (src/n).is_file()]
if missing:raise SystemExit(f'Missing Stage 2 outputs: {missing}; run Stage 2 first.')
dst.mkdir(exist_ok=True,parents=True)
for n in needed:
    shutil.copy2(src/n,dst/n)
    print('Prepared',dst/n)
print('Stage 3 is ready.')
