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
both for train and evalutation phase.

The orchestrator is designed to support a *control strategy* thus the train configuration changes dynamically
while the process is executed. The strategy allows to define a service level agreement (SLA) for the train phase.
By the way, the orchestrator can also be used with a static configuration, thus with static train parameters.

The messages between the *orchestrator* and the devices are exchanged using REST over HTTP. A *job* is submitted
to the device and the connection is closed. The device will send the *result* to the *orchestrator* when the
*job* will be completed. Other type of network protocol can be used, exploiting the *orchestrator* and *device* APIs.


## Setup
```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```
