"""Execute the notebook using exactly this Python, keeping Jupyter files local."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
local = ROOT / 'data' / 'jupyter'
kernel = local / 'kernels' / 'hw1'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Python 3 (HW1)', 'language': 'python'}))
os.environ['JUPYTER_PATH'] = str(local)
os.environ['JUPYTER_DATA_DIR'] = str(local)
os.environ['JUPYTER_CONFIG_DIR'] = str(local / 'config')
os.environ['JUPYTER_RUNTIME_DIR'] = str(local / 'runtime')
os.environ['IPYTHONDIR'] = str(local / 'ipython')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'data' / 'matplotlib')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import nbformat
from nbclient import NotebookClient

notebook = nbformat.read(ROOT / 'experiments.ipynb', as_version=4)


def cell_finished(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Executed code cell {cell_index}', flush=True)
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                print(output.text[-4000:], flush=True)
        nbformat.write(notebook, ROOT / 'experiments.ipynb')


client = NotebookClient(notebook, timeout=1800, kernel_name='hw1',
                        resources={'metadata': {'path': str(ROOT)}},
                        on_cell_executed=cell_finished)
try:
    client.execute()
finally:
    nbformat.write(notebook, ROOT / 'experiments.ipynb')
print('Notebook fully executed and saved.', flush=True)
