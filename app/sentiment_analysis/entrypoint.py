import papermill as pm
import mlflow
from pathlib import Path
import os
import sys
import subprocess
sys.path.append(os.path.abspath('..'))
from utils.pmexec import pmexec
import json

experiment_name = sys.argv[2]
run_id = sys.argv[4]
uri = sys.argv[6] 
conf = json.loads(sys.argv[8])

# TODO: condense into single config from sys args
conf.update({
   "run_id": run_id, 
   "experiment_name": experiment_name
})

# Papermill Executor
pmexec(
   uri,
   './input_notebooks/test.ipynb', 
   './output_notebooks/test.ipynb',
   pm_params = conf
)


