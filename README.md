# MLFlow REST API 实例（附代码）

在这个博客中，我们将介绍MLFlow REST API的代码，以及一些函数的使用，包括新建mlflow run，新建experiment，从experiment id/name中获得相关experiment信息，设置Tag/param/metric信息，设置log batch信息等等。MLFlow REST API的详细介绍参见博客[MLOps极致细节：7. MLFlow REST API 功能介绍及应用](https://blog.csdn.net/zyctimes/article/details/123418774)。

此博客的代码可以[在这里](https://gitee.com/yichaoyyds/mlflow-ex-restapi-basic)直接下载，基于[MLFlow官网GitHub链接](https://github.com/mlflow/mlflow)改写。关于MLFlow REST API的官方介绍参见[此链接](https://www.mlflow.org/docs/latest/rest-api.html#create-run)。

- 平台：Win10。
- IDE：Visual Studio Code
- 需要预装：Anaconda3。

目录

- [MLFlow REST API 功能介绍](#mlflow-rest-api-功能介绍)
  - [1 背景介绍](#1-背景介绍)
  - [2 官网的描述](#2-官网的描述)
  - [3 代码实现](#3-代码实现)
    - [3.1 新建mlflow run](#31-新建mlflow-run)
    - [3.2 获取和某一个run相关的信息](#32-获取和某一个run相关的信息)
    - [3.3 新建experiment](#33-新建experiment)
    - [3.4 罗列所有experiment信息](#34-罗列所有experiment信息)
    - [3.5 从experiment id中获得相关experiment信息](#35-从experiment-id中获得相关experiment信息)
    - [3.6 从experiment name中获得相关experiment信息](#36-从experiment-name中获得相关experiment信息)
    - [3.7 设置Tag信息](#37-设置tag信息)
    - [3.8 设置log param信息](#38-设置log-param信息)
    - [3.9 设置log matric信息](#39-设置log-matric信息)
    - [3.10 设置log batch信息](#310-设置log-batch信息)

## 1 背景介绍

一般情况下，我们可以直接使用MLflow的库来调用其中的功能模块，但也有一些情况，我们不希望使用MLflow库，或者我们并不是用Python来作为开发语言，那么MLflow REST API也是一种不错的选择。MLflow REST API允许您创建、列出、获取experiment和run，并记录parameters，metrics以及artifacts。API托管在MLflow跟踪服务器上的`/api`路由下。

## 2 代码运行

首先将代码下载到本地：`git clone https://gitee.com/yichaoyyds/mlflow-ex-restapi-basic.git`。

新建一个terminal，打开之后，最好先新建一个文件夹，然后输入

```bash
mlflow server
```

新建林我改一个terminal，打开之后，在`mlflow-ex-restapi-basic`的文件夹下输入：

```bash
python example.py
```

如果你是第一次尝试运行，你可能会看到terminal中的日志如下：

```bash
MLFlow RestAPI Example.
experiment_info:  None
Create new experiment.
Successfully create new experiment.
Successfully create run with run id: 850898a283554adcb60411ee7e126807
Successfully logged parameter.
Successfully set tag.
Successfully logged parameter
Successfully logged batch
```

如果不是第一次运行，terminal中的日志大致如下：

```bash
experiment_info:  {'experiment_id': '5', 'name': 'test2', 'artifact_location': './mlruns/5', 'lifecycle_stage': 'active'}
experiment has already been created.
Successfully create run with run id: a8bbe6717fbc473d8605eb6aae0afe1d
Successfully logged parameter.
Successfully set tag.
Successfully logged parameter
Successfully logged batch
```

## 3 代码解读

### 3.1 server的连接

首先我们需要启动一个server，mlflow背后用的是flask，所以通过`mlflow server`这个指令，系统就启动了一个本地的server，hostname=127.0.0.1，port=5000。这两个是默认值。如果希望有更多输入，可以参考：

```bash
mlflow server \
    --host 0.0.0.0 \
    --port 8889 \
    --serve-artifacts \
    --artifacts-destination s3://my-mlflow-bucket/ \
    --artifacts-only
```

当我们这个server运行起来后，在另一个terminal运行`example.py`才有效，否则就会出现如下error：

```terminal
requests.exceptions.ConnectionError: HTTPConnectionPool(host='127.0.0.1', port=5000): Max retries 
exceeded with url: /api/2.0/mlflow/experiments/get-by-name?experiment_name=test3 (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x000001E2AC630640>: Failed to establish a new connection: [WinError 10061] 由于目标计算机积极拒绝，无法连接。'))
```

### 3.2 代码逻辑

我们测试的代码存放于`example.py`中。相关的一些参数存于`config.txt`文件：

```config
[main]
hostname=127.0.0.1
port=5000
experiment-name=test3
```

由于这个案例比较简单，所以也就这么几个参数。注意，这里的hostname和port需要和实际运行的server相匹配。

`example.py`代码中，我们首先会实例化`MLflowTrackingRestApi`这个类：`mlflow_rest = MLflowTrackingRestApi(hostname, port, experiment_name)`。这个类实际上就包含了所有MLFlow REST API的函数了。之前的博客：[MLOps极致细节：7. MLFlow REST API 功能介绍及应用](https://blog.csdn.net/zyctimes/article/details/123418774)对其有详细的介绍。

在`__init__`函数中，我们设置了base url，并且检查当前experiment name是否已经被创建过，如果没有的话，创建这个experiment。然后在这个experiment里创建一个新的run。

```python
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
```

实例化后，我们就把所有之前写的mlflow rest api函数都跑一遍，看一下效果：

```python
# Log Parameter
#param = {"alpha": 0.1980}
param = {"key": "alpha", "value": 0.1980}
status_code = mlflow_rest.log_param(param)
if status_code == 200: print("Successfully logged parameter.")
else: print("Logging parameter failed: {}".format(status_code))

# Set Tag
tag = {"tag1": 1}
#tag = {"key": "tag1", "value": 1}
status_code = mlflow_rest.set_tag(tag)
if status_code == 200: print("Successfully set tag.")
else: print("Logging parameter failed: {}".format(status_code))

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
