"""Install only NASA C-MAPSS FD001 source text files into the expected stage folders.

Usage:
    python scripts/setup_dataset.py /path/to/CMAPSSData.zip
    python scripts/setup_dataset.py /path/to/NASA_outer_archive.zip

Supports NASA's outer archive (containing CMAPSSData.zip) or inner archive.
Only the FD001 files are extracted. Dataset is not committed to this repository.
"""
from pathlib import Path
from zipfile import ZipFile
import argparse
import io
import sys

ROOT = Path(__file__).resolve().parents[1]
NEEDED = ('train_FD001.txt','test_FD001.txt','RUL_FD001.txt','readme.txt')

def extract(path):
    """Read original FD001 files from a NASA inner or outer ZIP archive."""
    with ZipFile(path) as outer:
        inner_names = [n for n in outer.namelist() if n.endswith('CMAPSSData.zip')]
        if inner_names:
            with outer.open(inner_names[0]) as f:
                payload=f.read()
            with ZipFile(io.BytesIO(payload)) as inner:
                return read_data(inner)
        return read_data(outer)

def read_data(z):
    """Read exactly one copy of each required FD001 archive member."""
    out={}
    for filename in NEEDED:
        matches=[n for n in z.namelist() if n.split('/')[-1] == filename]
        if len(matches)!=1:
            raise RuntimeError(f'Expected one {filename} in archive, found {len(matches)}')
        out[filename] = z.read(matches[0])
    return out

def main():
    """Extract original FD001 source data into stage-specific data folders."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path,help='NASA outer ZIP or CMAPSSData.zip')
    args=parser.parse_args()
    if not args.archive.is_file():parser.error(f'Archive not found: {args.archive}')
    data=extract(args.archive)
    for name in ['stage1','stage2','stage4']:
        directory=ROOT/name/'data'
        directory.mkdir(parents=True,exist_ok=True)
        for filename,payload in data.items():
            (directory/filename).write_bytes(payload)
        print(f'Set up {name}/data: {len(data)} files')
    print('Done. Next: run Stage 1 and Stage 2 as shown in README.md.')

if __name__=='__main__':main()
