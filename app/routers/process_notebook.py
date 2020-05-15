# Setup/Import ----------------------------------------------------------------
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError
from typing import Dict, List, Union, Any
from typing_extensions import Literal
import os.path as path
import mlflow
import json
from mlflow.tracking import MlflowClient
from utils.get_config import get_config
import time
from fastapi.encoders import jsonable_encoder
from starlette.responses import HTMLResponse, JSONResponse

# Grab Config
conf = get_config()

# URI
uri = conf['uri']
mlClient = MlflowClient(tracking_uri=uri)
mlflow.set_tracking_uri(uri)

# Experiment Name
experiment_name = conf['experiment_name']
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
router = APIRouter()

# Process Endpoint ------------------------------------------------------------

class StatusResponse(BaseModel):
    job_id: str = None
    job_status: Literal[
        (*map(mlflow.entities.RunStatus.to_string, 
        mlflow.entities.RunStatus.all_status()), )
        ] = None
    time_stamp: str = None
        
    class Config:
        schema_extra = {
            "example" : {
                "job_id": "1915ced38b055279ab56e66ca04a0aab",
                "job_status": "SCHEDULED",
                "time_stamp": "Thu May 14 18:17:42 2020",
            }
        }

# TODO: add process request body
class ProcessBody(BaseModel):
    word: str = None

@router.post("/process/", response_model = StatusResponse)
async def process(item: ProcessBody):
    ''' Execute Selected Notebook '''
   
    try:
        job_run = mlflow.start_run(experiment_id=experiment_id)
        job_id = job_run.info.run_id
        job_status = job_run.info.status
    
        outfile = path.join(mlflow.get_artifact_uri()[7:],'<>.json')
        with open(outfile, 'w') as outfile:
            json.dump(item.json(), outfile)

        mlflow.end_run(status='SCHEDULED')

        # Runs a Module of ML Flow
        actual_run = mlflow.projects.run(
            uri='./sentiment_analysis', #TODO: parameterize later for more modules
            entry_point='entrypoint.py',
            parameters={
                'experiment_name': experiment_name, 
                'run_id': job_id,
                'uri': conf['uri'], 
                'conf': json.dumps(conf)
            },
            experiment_name=experiment_name,
            run_id=job_id,
            synchronous=False,
            use_conda=False 
        )

        # Create Response
        response = StatusResponse(
            job_id=job_id, 
            job_status=job_status,
            time_stamp=time.ctime()
        )

    except ValidationError as e:
        raise HTTPException(status_code=500, detail=e.json())

    return response

# Status Endpoint -------------------------------------------------------------
@router.get("/status/{job_id}", response_model=StatusResponse)
def status(job_id: str):
    '''get the status of a previous lauched run'''

    try:
        job_status = mlClient.get_run(run_id=job_id).info.status

        response = StatusResponse(
            job_id=job_id, 
            job_status=job_status,
            time_stamp=time.ctime()
        )

    except ValidationError as e:
        raise HTTPException(status_code=500, detail=e.json())

    return response

# Results Endpoint ------------------------------------------------------------
class ResultsResponse(BaseModel):
    output: str
    time_stamp: str

@router.get("/results/{job_id}", response_model=ResultsResponse)
def results(job_id: str):
    '''Grab Results'''
    
    try:
        job_status = mlClient.get_run(run_id=job_id).info.status

        # Check to see if it is still processing
        if job_status != 'FINISHED':
            return  JSONResponse(
                status_code=102,
                content=jsonable_encoder({
                    "error": {
                        "message": "Notebook still executing...",
                        "code": 102
                    },
                    "timestamp": time.ctime(),
                    "reference": "ML-NB-ERROR"
                })
            )
            
        job = mlClient.get_run(run_id=job_id)

        results_path = path.join(job.info.artifact_uri,'output.json')[7:]
        with open(results_path) as json_file:
            output = json.load(json_file)
        
        response = ResultsResponse(
            output = output,
            time_stamp = time.ctime()
        )

    except ValidationError as e:
        raise HTTPException(status_code=500, detail=e.json())
    
    return response


