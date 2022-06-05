# MLflow REST API instance (with code)

In this blog, we will introduce the code of the MLflow REST API and the use of some functions, including creating a new mlflow run, creating a new experiment, obtaining relevant experiment information from the experiment id/name, setting Tag/param/metric information, setting log batch information, and so on. For a detailed introduction to the MLflow REST API, see the blog [MLOps Ultimate Details: 7. MLflow REST API function introduction and application] (https://blog.csdn.net/zyctimes/article/details/123418774 ).

The code for this blog can be [here] (https://gitee.com/yichaoyyds/mlflow-ex-restapi-basic ) Direct download, based on [MLflow official website GitHub link] (https://github.com/mlflow/mlflow ) Rewrite. For the official introduction of the MLflow REST API, see [this link] (https://www.mlflow.org/docs/latest/rest-api.html#create-run ).

-Platform: Win10.
- IDE：Visual Studio Code
-Requires pre-installation: Anaconda3.

Table of contents

 - [MLflow REST API Function Introduction](#mlflow-rest-api-function-introduction)
 - [1 Background Introduction](#-1-background-introduction)
 - [2 Description of the official website](#-2-description-of-the-official-website)
 - [3 code implementation](#-3-code-implementation)
 - [3.1 New mlflow run](#-31-new-mlflow-run)
 - [3.2 Get information related to a certain run](#-32-Get-information-related-to-a-certain-run)
 - [3.3 New experiment](#-33-new-experiment)
 - [3.4 List all experiment information](#-34-list-all-experiment-information)
 - [3.5 Get relevant experiment information from experiment id](#-35-get-relevant-experiment-information-from-experiment-id)
 - [3.6 Get relevant experiment information from experiment name](#-36-get-relevant-experiment-information-from-experiment-name)
 - [3.7 Set tag information](#-37-set-tag-information)
 - [3.8 Set log param information](#-38-set-log-param-information)
 - [3.9 Set log matric information](#-39-set-log-matric-information)
 - [3.10 Set log batch information](#-310-set-log-batch-information)
 
 ## 1 Background introduction

Under normal circumstances, we can directly use the MLflow library to call the functional modules in it, but there are some cases where we don't want to use the MLflow library, or we don't use Python as the development language, then the MLflow REST API is also a good choice. The MLflow REST API allows you to create, list, get experiments and runs, and record parameters, metrics, and artifacts. The API is hosted under the `/api` route on the MLflow tracking server.

## 2 Code run

First download the code locally: 'git clone https://gitee.com/yichaoyyds/mlflow-ex-restapi-basic.git `.

Create a new terminal. After opening it, it is best to create a new folder first, and then enter

```bash
mlflow server
```

I will change a terminal for the new forest. After opening it, enter it in the folder `mlflow-ex-restapi-basic`.：

```bash
python example.py
```

If you are trying to run for the first time, you may see the log in terminal as follows：

```bash
MLFlow RestAPI Example.
experiment_info: None
Create new experiment.
Successfully create new experiment.
Successfully create run with run id: 850898a283554adcb60411ee7e126807
Successfully logged parameter.
Successfully set tag.
Successfully logged parameter
Successfully logged batch
```

If it is not the first time it is run, the log in terminal is roughly as follows：

```bash
experiment_info: {'experiment_id': '5', 'name': 'test2', 'artifact_location': './mlruns/5', 'lifecycle_stage': 'active'}
experiment has already been created.
Successfully create run with run id: a8bbe6717fbc473d8605eb6aae0afe1d
Successfully logged parameter.
Successfully set tag.
Successfully logged parameter
Successfully logged batch
```

## 3 Code implementation

### 3.1 server connection

First of all, we need to start a server. Flask is used behind mlflow, so through the command "mlflow server", the system starts a local server, hostname=127.0.0.1, port=5000. These two are the default values. If you want more input, you can refer to：

```bash
mlflow server \
--host 0.0.0.0 \
--port 8889 \
--serve-artifacts \
--artifacts-destination s3://my-mlflow-bucket/ \
--artifacts-only
```

When our server is up and running, run it in another terminal`example.py `Is only valid, otherwise the following error will appear：

```terminal
requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=5000): Max retries
exceeded with url: /api/2.0/mlflow/experiments/get-by-name? experiment_name=test3 (Caused by NewConnectionError('<urllib3.connection. HttpConnection object at 0x000001E2AC630640>: Failed to establish a new connection: [WinError 10061] Unable to connect due to active rejection by the target computer. '))
```

### 3.2 Code logic

The code we tested is stored in`example.py `In. Some related parameters are stored in`config.txt 'file：

```config
[main]
hostname=127.0.0.1
port=5000
experiment-name=test3
```

Since this case is relatively simple, there are only a few parameters. Note that the host name and port here need to match the actual running server.

`example.py `In the code, we will first instantiate the class 'MLflowTrackingRestApi': 'mlflow_rest = MLflowTrackingRestApi (hostname, port, experiment_name)`. This class actually contains all the functions of the MLflow REST API. Previous blog: [MLOps Ultimate details: 7. MLflow REST API function introduction and application] (https://blog.csdn.net/zyctimes/article/details/123418774 ) There is a detailed introduction to it.

In the `__init__' function, we set the base url and check whether the current experiment name has been created. If not, create this experiment. Then create a new run in this experiment.

```python
def __init__(self, hostname="127.0.0.1", port=5000, experiment_name=None):
    self.base_url = "http://" + hostname + ":" + str(port) + "/api/2.0/mlflow"
    experiment_name = str(experiment_name)
    experiment_info = self.get_experiment_by_name(experiment_name)
    print("experiment_info: ",experiment_info)
    if experiment_info == None:
        print("Create new experiment.")
        status_code = self.create_experiment(experiment_name = experiment_name)
    if status_code == 200: 
        print("Successfully create new experiment.")
    else: 
        print("experiment creation failed: {}".format(status_code))
    else:
        print("experiment has already been created.")
        self.experiment_id = experiment_info["experiment_id"]
        # Create a new run
        self.run_id = self.create_run()
```

After instantiation, we will run all the mlflow rest api functions written before to see the effect：

```python
# Log Parameter
#param = {"alpha": 0.1980}
param = {"key": "alpha", "value": 0.1980}
status_code = mlflow_rest.log_param(param)
if status_code == 200: 
    print("Successfully logged parameter.")
else: 
    print("Logging parameter failed: {}".format(status_code))

# Set Tag
tag = {"tag1": 1}
#tag = {"key": "tag1", "value": 1}
status_code = mlflow_rest.set_tag(tag)
if status_code == 200: 
    print("Successfully set tag.")
else: 
    print("Logging parameter failed: {}".format(status_code))

# Log Metric
metric = {"precision": 0.769}
# metric = {"key": "precision", "value": "0.769"}
status_code = mlflow_rest.log_metric(metric)
if status_code == 200: print("Successfully logged parameter")
else: print("Logging metric failed: {}".format(status_code))

# Log Batch
metrics = {"mse": 2500.00, "rmse": 50.00}
params = {"learning_rate": 0.01, "n_estimators": 10}
#metrics = [{"key": "mse", "value": "0.769"}, {"key": "callback", "value": "0.512"}]
#params = [{"key": "learning_rate", "value": "0.018"}, {"key": "beta", "value": "0.98"}, {"key": "gamma", "value": "512"}]
tags = []
status_code = mlflow_rest.log_batch(metrics, params, tags)
if status_code == 200:
print("Successfully logged batch")
else:
print("Logging batch failed: {}".format(status_code))
```