import papermill as pm
import mlflow
from pathlib import Path
import sys, subprocess

def pmexec(uri, in_nb, out_nb, pm_params = {}, verbose = False):
    ''' Function to execute notebooks using Papermill '''

    # Set ML Flow Experiment Information
    mlflow.set_experiment(pm_params['experiment_name'])
    mlflow.set_tracking_uri(uri) 
    mlflow.start_run(pm_params['run_id'])
    experiment_id = mlflow.active_run().info.experiment_id

    if verbose:
        print("Executing experiment ID: ", experiment_id)
    
    # Executor
    pm.execute_notebook(
        in_nb,  # input notebook path
        out_nb, # output notebook path
        parameters=dict(
            pm_params = pm_params # parameters to notebook, must have "params" tag
        )
    )
    
    # Log the Output For Reference
    mlflow.log_artifact(out_nb)
    subprocess.run(["jupyter", "nbconvert", out_nb, "--to", "html"])

    mlflow.log_artifact(out_nb.replace('ipynb', 'html'))
    subprocess.run(["rm", out_nb.replace('ipynb', 'html')])

    mlflow.end_run()