import requests
import time
import os
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

_DEFAULT_USER_ID = "unknown"

def _get_user_id():
    """Get the ID of the user for the current run."""
    try:
        import pwd
        return pwd.getpwuid(os.getuid())[0]
    except ImportError:
        return _DEFAULT_USER_ID

class MLflowTrackingRestApi:
    def __init__(self, hostname="127.0.0.1", port=5000, experiment_name=None):
        self.base_url = "http://" + hostname + ":" + str(port) + "/api/2.0/mlflow"
        experiment_name = str(experiment_name)
        experiment_info = self.get_experiment_by_name(experiment_name)
        print("experiment_info: ",experiment_info)
        if experiment_info == None:
            print("Create new experiment.")
            status_code = self.create_experiment(experiment_name = experiment_name)
            if status_code == 200: print("Successfully create new experiment.")
            else: print("experiment creation failed: {}".format(status_code))
        else:
            print("experiment has already been created.")
            self.experiment_id = experiment_info["experiment_id"]
        # Create a new run
        self.run_id = self.create_run()

    def create_run(self):
        """Create a new run for tracking."""
        url = self.base_url + "/runs/create"
        # user_id is deprecated and will be removed from the API in a future release
        payload = {
            "experiment_id": self.experiment_id,
            "start_time": int(time.time() * 1000),
            "user_id": _get_user_id(),
        }
        r = requests.post(url, json=payload)
        run_id = None
        if r.status_code == 200:
            run_id = r.json()["run"]["info"]["run_uuid"]
            print("Successfully create run with run id: {}".format(run_id))
        else:
            print("Creating run failed!")
        return run_id

    def get_run(self):
        """Get run info provided by run id."""
        url = self.base_url + "/runs/get"
        print("run_id: ", self.run_id)
        r = requests.get(url,params={"run_id":self.run_id})
        run = None
        if r.status_code == 200:
            run = r.json()["run"]
        return run

    def create_experiment(self, experiment_name="0", 
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None):
        """Create a new experiment."""
        url = self.base_url + "/experiments/create"
        payload = {
            "name": str(experiment_name),
            "artifact_location": artifact_location,
            "tags": tags,
        }
        r = requests.post(url, json=payload)
        if r.status_code == 200:
            self.experiment_id = r.json()["experiment_id"]
        else:
            print("Creating experiment failed!")
        return r.status_code

    def list_experiments(self):
        """Get all experiments."""
        url = self.base_url + "/experiments/list"
        r = requests.get(url)
        experiments = None
        if r.status_code == 200:
            experiments = r.json()["experiments"]
        return experiments

    def get_experiment_by_id(self,experiment_id):
        """Get one experiment info and runs inside by experiment id."""
        url = self.base_url + "/experiments/get"
        r = requests.get(url,params={"experiment_id":experiment_id})
        experiment = None
        if r.status_code == 200:
            experiment = r.json()["experiment"]
        return experiment

    def get_experiment_by_name(self,experiment_name):
        """Get one experiment info and runs inside by experiment name."""
        url = self.base_url + "/experiments/get-by-name"
        r = requests.get(url,params={"experiment_name":experiment_name})
        experiment = None
        if r.status_code == 200:
            experiment = r.json()["experiment"]
        return experiment


    def log_batch(self, metrics=[], params=[], tags=[]):
        """
        Log a batch of metrics, params, and tags for a run.
        metrics, params, tags support list or dict or empt.
        For example: 
        (1) metrics=[];
        (2) metrics=[{"key": "mse", "value": "0.769"}, {"key": "callback", "value": "0.512"}];
        (3) metrics={"mse": 2500.00, "rmse": 50.00};
        """
        url = self.base_url + "/runs/log-batch"
        # support metrics input with list or dict or empty
        if len(metrics)==0:
            metricsList = []
        elif isinstance(metrics, list):
            metricsList = [ {"key":str(metric["key"]), "value":str(metric["value"])} for metric in metrics]
        elif isinstance(metrics, dict):
            metricsList = [ {"key":key, "value":str(value)} for key, value in metrics.items()]
        # support params input with list or dict or empty
        if len(params)==0:
            paramsList = []
        elif isinstance(params, list):
            paramsList = [ {"key":str(param["key"]), "value":str(param["value"])} for param in params]
        elif isinstance(params, dict):
            paramsList = [ {"key":key, "value":str(value)} for key, value in params.items()]
        # support tags input with list or dict or empty
        if len(tags)==0:
            tagsList = []
        elif isinstance(tags, list):
            tagsList = [ {"key":str(tag["key"]), "value":str(tag["value"])} for tag in tags]
        elif isinstance(tags, dict):
            tagsList = [ {"key":key, "value":str(value)} for key, value in tags.items()]

        payload = {"run_id": self.run_id, "metrics": metricsList, "params": paramsList, "tags": tagsList}
        #print("payload: ",payload)
        r = requests.post(url, json=payload)
        return r.status_code

    def set_tag(self, tag):
        """
        Log a parameter for the given run. Tag support dict. For example,
        - tag={"key": "precision", "value": 0.769}
        - tag={"precision": 0.769}
        """
        url = self.base_url + "/runs/set-tag"
        if tag:
            if tag.keys() == ['key', 'value']:
                payload = {"run_id": self.run_id, "key": tag["key"], "value": str(tag["value"])}
            else:   
                payload = {"run_id": self.run_id, "key": list(tag.keys())[0], "value": str(list(tag.values())[0])}
            r = requests.post(url, json=payload)
            return r.status_code
        else:   # dictionary is empty
            return 0        

    def log_param(self, param):
        """
        Log a parameter for the given run. Param support dict. For example,
        - param={"key": "precision", "value": 0.769}
        - param={"precision": 0.769}
        """
        url = self.base_url + "/runs/log-parameter"
        if param:
            if param.keys() == ['key', 'value']:
                payload = {"run_id": self.run_id, "key": param["key"], "value": param["value"]}
            else:
                payload = {"run_id": self.run_id, "key": list(param.keys())[0], "value": str(list(param.values())[0])}
            r = requests.post(url, json=payload)
            return r.status_code
        else:   # dictionary is empty
            return 0
        

    def log_metric(self, metric={}, step: Optional[int] = None):
        """
        Log a metric for the given run. Metric support dict. For example,
        - metric={"key": "precision", "value": 0.769}
        - metric={"precision": 0.769}
        """
        url = self.base_url + "/runs/log-metric"
        if metric:
            if metric.keys() == ['key', 'value']:
                payload = {"run_id": self.run_id, 
                    "key": metric["key"], 
                    "value": str(metric["value"]),
                    "timestamp": int(time.time() * 1000),
                    "step":step or 0}
            else:   
                payload = {"run_id": self.run_id, 
                    "key": list(metric.keys())[0], 
                    "value": str(list(metric.values())[0]),
                    "timestamp": int(time.time() * 1000),
                    "step":step or 0}
            r = requests.post(url, json=payload)
            return r.status_code
        else:   # dictionary is empty
            return 0

    def log_model(self, model_json):
        """Log a model for the given run."""
        url = self.base_url + "/runs/log-model"
        payload = {"run_id": self.run_id, "model_json": model_json}
        r = requests.post(url, json=payload)
        return r.status_code

    '''
    def record_logged_model(self, run_id, mlflow_model):
        req_body = message_to_json(LogModel(run_id=run_id, model_json=mlflow_model.to_json()))
        self._call_endpoint(LogModel, req_body)
    '''