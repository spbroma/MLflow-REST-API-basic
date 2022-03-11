"""
Using MLflow REST API instead of MLflow library might be useful to embed in an application 
where you don't want to depend on the whole MLflow library, or to make your own HTTP requests
in another programming language (not Python).

In this example, we will test many REST API functions, include:
- create new mlflow run
- create new mlflow experiment
- list all experiments
- get specific experiment info by experiment id
- get specific experiment info by experiment name
- log batch
- Set Tag
- log param
- log metric

For more details on MLflow REST API endpoints check the following page:
https://www.mlflow.org/docs/latest/rest-api.html
"""

import os
import sys
from MLflowTrackingRestApi import MLflowTrackingRestApi
import configparser
from optparse import OptionParser

def main(args):

    config = configparser.ConfigParser()                                    
    config.read(cfg_file)
    config.sections()

    # Set properties of parameters inside config file.
    for key in config['main']:
        if key == 'port' :
            port = config.getint('main', key)   
        if key == 'hostname': 
            hostname = config.get('main', key)
        if key == 'experiment-name':
            experiment_name = config.get('main', key)

    print("MLFlow RestAPI Example.")
    mlflow_rest = MLflowTrackingRestApi(hostname, port, experiment_name)

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

def parse_args():
    '''
    Parse and validate input arguments
    '''
    global cfg_file

    parser = OptionParser()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    configfile_dir = os.path.abspath(os.path.join(current_dir,"./config.txt"))
    parser.add_option("-c", "--cfg-file", dest="cfg_file", default=configfile_dir,
                  help="Set the adaptor config file.", metavar="FILE")

    (options, _) = parser.parse_args()
    cfg_file = options.cfg_file
    return 0

if __name__ == '__main__':
    ret = parse_args()
    #If argumer parsing fail, return failure (non-zero)
    if ret == 1:
        sys.exit(1)
    sys.exit(main(sys.argv))
