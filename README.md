L. Baresi, G. Quattrocchi and N. Rasi, "Federated Machine Learning as a Self-Adaptive Problem," in 2021 International Symposium on Software Engineering for Adaptive and Self-Managing Systems (SEAMS), 2021 pp. 41-47.
url: https://doi.ieeecomputersociety.org/10.1109/SEAMS51251.2021.00016

---

# Federated Learning Framework

The FL framework is composed by the following two components:

- device: (client) it executes the the ML Job
- orchestrator: it manages the devices, models and send *jobs* to the devices applying a *control strategy*.
It collects and aggregates results using an *aggregation strategy* (such as *FedAvg*)


## Architecture

The orchestrator exploits devices for training, sending them *jobs* and receiving *results*.

A *job* have different purposes: get initial weights for a model, train a model, evaluate a model.
The *model* is created (or stored) on the devices along with train and test data, while the *orchestrator* only
store the model parameters.

Results from device are collected and aggregated using an *aggregation strategy*,
both for train and evaluation phase.

The orchestrator is designed to support a *control strategy* thus the train configuration changes dynamically
while the process is executed. The strategy allows to define a service level agreement (SLA) for the training phase.
The orchestrator can also be used with a static configuration, thus with static train parameters.

The messages between the *orchestrator* and the devices are exchanged using REST over HTTP. A *job* is submitted
to the device and the connection is closed. The device will send the *result* to the *orchestrator* when the
*job* will be completed. Other type of network protocol can be used, exploiting the *orchestrator* and *device* APIs.


## Setup and Run
```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

### Start Orchestrator
Training parameters can be set with the following environment variables:
- ```FL_ROUNDS```: number of rounds of the FL protocol (R)
- ```FL_EPOCHS```: number of epochs for the local model training (E)
- ```FL_BATCHSIZE```: size of the batch for the local model training (B)
- ```FL_K_FIT```: fraction of clients used for model fit (training) (K_fit)
- ```FL_K_EVAL```: fraction of clients used for model evaluation (test) (K_eval)
- ```FL_MIN```: set the minimum number of devices to wait before starting the train (C). 
- ```FL_CONTROL```: set the type of control strategy (optimizer) used for the training and should be set to one of the following
values:

    - STATIC = 1
    - DYNAMIC_LINEAR_ROUNDS = 2
    - DYNAMIC_QUADRATIC_ROUNDS = 3
    - DYNAMIC_LINEAR_NETWORK = 4
    - DYNAMIC_QUADRATIC_NETWORK = 5
- ```FL_TACCURACY```: target accuracy (SLA) for the optimizer 
- ```FL_TROUNDS```: target number of rounds (R) for the optimizer 
- ```FL_TNETWORK```: target consumed network (UB) for the optimizer 
- ```FL_MODEL```: model to use from the ones already available
    
    - "mnist"
    - "fashion_mnist"
    - "cifar10"
    - "imdb_reviews"
- ```FL_EXPORT_METRICS```: export training metrics at the end
- ```FL_TERMINATE```: terminate python processes (orchestrator and devices) at the end

```bash
export FL_ROUNDS=15 &&
export FL_EPOCHS=1 &&
export FL_BATCHSIZE=32 &&
export FL_K_FIT=1 &&
export FL_K_EVAL=1 &&
export FL_MIN=10 &&
export FL_CONTROL=3 &&
export FL_TACCURACY=0.8 &&
export FL_TROUNDS=10 &&
export FL_TNETWORK=100 &&
export FL_MODEL="mnist" &&
export FL_EXPORT_METRICS=true 
export FL_TERMINATE=true
python main_orchestrator.py
```

### Start Devices
A script is provided to concurrently start devices:

- ```FL_NUM_DEVICES```: number of devices to start (C)
- ```FL_NK```: number of examples to use (N)
- ```FL_MODEL```: model to use from the ones already available
    
    - "mnist"
    - "fashion_mnist"
    - "cifar10"
    - "imdb_reviews"

```bash
export FL_NUM_DEVICES=10 &&
export FL_NK=100 &&
export FL_MODEL="mnist" &&
./start_devices.sh
```