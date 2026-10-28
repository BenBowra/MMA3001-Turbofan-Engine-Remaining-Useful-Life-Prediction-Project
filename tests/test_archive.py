"""Check NASA-style ZIP setup without bundling or uploading NASA source data."""
import importlib.util
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import pytest

ROOT=Path(__file__).resolve().parents[1]

def setup_module_from_path():
    """Import the side-effect-free data installation module."""
    spec=importlib.util.spec_from_file_location('fd001_setup',ROOT/'scripts/setup_dataset.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sample_zip(missing=None):
    """Create a tiny synthetic FD001 archive for unit tests."""
    payload=BytesIO()
    with ZipFile(payload,'w') as z:
        for name in ('train_FD001.txt','test_FD001.txt','RUL_FD001.txt','readme.txt'):
            if name!=missing:
                z.writestr('CMAPSSData/'+name,name.encode())
    return payload.getvalue()

def test_inner_archive(tmp_path):
    """Extract four named files from an inner NASA-format ZIP."""
    p=tmp_path/'CMAPSSData.zip'
    p.write_bytes(sample_zip())
    files=setup_module_from_path().extract(p)
    assert len(files)==4 and files['train_FD001.txt']==b'train_FD001.txt'

def test_outer_archive(tmp_path):
    """Support the outer NASA archive containing CMAPSSData.zip."""
    p=tmp_path/'nasa_outer.zip'
    with ZipFile(p,'w') as z:
        z.writestr('folder/CMAPSSData.zip',sample_zip())
    files=setup_module_from_path().extract(p)
    assert len(files)==4 and files['RUL_FD001.txt']==b'RUL_FD001.txt'

def test_missing_member_rejected(tmp_path):
    """Do not silently accept incomplete NASA archives."""
    p=tmp_path/'bad.zip'
    p.write_bytes(sample_zip(missing='test_FD001.txt'))
    with pytest.raises(RuntimeError,match='test_FD001'):
        setup_module_from_path().extract(p)

def test_duplicate_member_rejected(tmp_path):
    """Ambiguous dataset members cause a clear error."""
    p=tmp_path/'duplicated.zip'
    with ZipFile(p,'w') as z:
        for name in ('train_FD001.txt','test_FD001.txt','RUL_FD001.txt','readme.txt'):
            z.writestr('a/'+name,b'test')
        z.writestr('b/train_FD001.txt',b'test')
    with pytest.raises(RuntimeError,match='train_FD001'):
        setup_module_from_path().extract(p)
