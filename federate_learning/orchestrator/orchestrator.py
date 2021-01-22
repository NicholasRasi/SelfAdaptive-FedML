import json
import threading
import time
import queue
from typing import List, Tuple
from federate_learning.device import Device
from federate_learning.orchestrator import DeviceManager, JobsDispatcher
from federate_learning.common import JobType, Job, Parameters, Result, Model
from federate_learning.orchestrator.aggregation_strategy import FedAvg
from federate_learning.orchestrator.control_strategy import ControlStrategy


class Orchestrator:
    def __init__(self, logger):
        self.logger = logger
        self.status = "active"
        self.jobs_dispatcher = JobsDispatcher(logger=self.logger)
        self.results_queue = queue.Queue()
        self.device_manager = DeviceManager()
        self.condition = threading.Condition()
        self.available_models = {}
        self.export_metrics = False
        self.metrics_file = 'execution_metrics/metrics_{}_{}_{}_{}_{}.json'

    def config(self,
               available_models,
               export_metrics=False):
        self.available_models = available_models
        for model in available_models:
            model.logger = self.logger
            model.control_strategy.logger = self.logger
        self.export_metrics = export_metrics

    def training(self):
        for model in self.available_models:
            self.logger.info("starting training for model {}".format(model.name))
            cs: ControlStrategy = model.control_strategy

            self.logger.debug("waiting min {} devices available".format(cs.min_devices))
            while self.device_manager.num_devices() < cs.min_devices:
                time.sleep(2)
            self.logger.info("starting federated learning,"
                             "init params: R={}, K(train)={}, K(eval)={}".format(cs.num_rounds,
                                                                                 cs.k_fit,
                                                                                 cs.k_eval))

            self.logger.debug("getting init weights from a random client")
            device = self.device_manager.get_sample(num_devices=1, model_name=model.name)[0]
            weights = self.get_init_weights(model=model, device=device)

            self.logger.debug("updating model weights")
            model.weights = weights

            for tround in range(1, cs.num_rounds + 1):
                self.logger.info("round {}/{}".format(tround, cs.num_rounds))

                self.logger.info("applying control strategy")
                k_fit, k_eval, device_config = model.control_strategy.apply_strategy(tround)
                self.logger.info("K(fit)={}, K(eval)={}, config={}".format(k_fit, k_eval, device_config))

                num_devices_fit = int(self.device_manager.num_devices() * k_fit)
                self.logger.debug("selecting {} random devices for fit".format(num_devices_fit))
                devices_fit = self.device_manager.get_sample(model_name=model.name,
                                                             num_devices=num_devices_fit)
                self.logger.info("selected random client sample: {}".format([d.id for d in devices_fit]))

                self.logger.info("fitting model")
                new_weights, aggregated_loss_fit, \
                aggregated_acc_fit, aggregated_cost_fit = self.fit_model(model=model,
                                                                         devices=devices_fit,
                                                                         device_config=device_config)

                self.logger.debug("updating model weights")
                model.weights = Parameters(ndarray_list=new_weights)

                num_devices_eval = int(self.device_manager.num_devices() * k_eval)
                self.logger.debug("selecting {} random devices for eval".format(num_devices_eval))
                devices_eval = self.device_manager.get_sample(model_name=model.name, num_devices=num_devices_eval)
                self.logger.info("selected random client sample: {}".format([d.id for d in devices_eval]))
                aggregated_loss_eval, aggregated_acc_eval = self.evaluate_model(model=model,
                                                                                devices=devices_eval,
                                                                                tround=tround)

                self.logger.debug("updating model metrics")
                model.metrics.add_metrics(k_fit=k_fit,
                                          num_devices_fit=num_devices_fit,
                                          loss_fit=aggregated_loss_fit,
                                          accuracy_fit=aggregated_acc_fit,
                                          computation_cost_fit=aggregated_cost_fit,
                                          network_cost_fit=len(devices_fit),
                                          k_eval=k_eval,
                                          num_devices_eval=num_devices_eval,
                                          loss_eval=aggregated_loss_eval,
                                          accuracy_eval=aggregated_acc_eval,
                                          computation_cost_eval=0,
                                          network_cost_eval=len(devices_eval),
                                          device_config=device_config)

            self.print_model_metrics(model=model)
            if self.export_metrics:
                self.export_model_metrics(model=model)

    def print_model_metrics(self, model: Model):
        cs: ControlStrategy = model.control_strategy
        self.logger.info("training for model {} finished".format(model.name))
        self.logger.info("control type: {}".format(cs.name))
        self.logger.info("target accuracy: {:0.4f}".format(cs.target.accuracy))
        self.logger.info("target num rounds: {}".format(cs.target.num_round))
        self.logger.info("target network cost: {}".format(cs.target.network_cost))
        self.logger.info("losses fit: {}".format(["{0:0.4f}".format(loss) for loss in model.metrics.losses_fit]))
        self.logger.info("accuracies fit: {}".format(["{0:0.4f}".format(acc) for acc in model.metrics.accuracies_fit]))
        self.logger.info("computation costs fit: {}, total: {:0.2f}".format(
            ["{0:0.2f}".format(c) for c in model.metrics.computation_costs_fit], sum(model.metrics.computation_costs_fit)))
        self.logger.info("network costs fit: {}, total: {:0.2f}".format(
            ["{0:0.2f}".format(c) for c in model.metrics.network_costs_fit], sum(model.metrics.network_costs_fit)))
        self.logger.info("losses eval: {}".format(["{0:0.4f}".format(loss) for loss in model.metrics.losses_eval]))
        self.logger.info("accuracies eval: {}".format(["{0:0.4f}".format(acc) for acc in model.metrics.accuracies_eval]))
        self.logger.info("dev_configs: {}".format(model.metrics.device_configs))

    def export_model_metrics(self, model):
        cs = model.control_strategy
        filename = self.metrics_file.format(model.name, cs.name, cs.target.num_round,
                                            int(cs.target.accuracy*100), cs.target.network_cost)
        self.logger.info("saving metrics to file {}".format(filename))
        metrics = dict(model.metrics.__dict__)
        metrics["target"] = model.control_strategy.target.__dict__
        metrics["num_rounds"] = model.control_strategy.num_rounds
        with open(filename, 'w') as outfile:
            json.dump(metrics, outfile)

    def fit_model(self, model: Model, devices: List[Device], device_config: dict):
        job: Job = Job(job_type=JobType.FIT,
                       model_name=model.name,
                       parameters=model.weights,
                       configuration=device_config)

        self.logger.debug("sending jobs to devices")
        self.jobs_dispatcher.send_jobs(job, devices)

        self.wait_for_responses(len(devices))
        results: List[Result] = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get())

        self.logger.info("received {} results".format(len(results)))
        self.logger.info("applying federated fit strategy")

        weights = [(r.num_samples, r.parameters.weights) for r in results]
        losses = [(r.num_samples, r.loss) for r in results]
        accuracies = [(r.num_samples, r.accuracy) for r in results]
        costs = [(r.num_samples, r.configuration) for r in results]

        aggregated_weights = FedAvg.aggregate_fit(weights)
        aggregated_loss = FedAvg.aggregate_losses(losses)
        aggregated_accuracy = FedAvg.aggregate_accuracies(accuracies)
        aggregated_cost = FedAvg.aggregate_costs(costs)

        return aggregated_weights, aggregated_loss, aggregated_accuracy, aggregated_cost

    def evaluate_model(self, model: Model, devices: List[Device], tround: int) -> Tuple[float, float]:
        job: Job = Job(job_type=JobType.EVALUATE,
                       model_name=model.name,
                       parameters=model.weights,
                       configuration={})

        self.logger.debug("sending jobs to devices")
        self.jobs_dispatcher.send_jobs(job, devices)

        self.wait_for_responses(len(devices))
        results: List[Result] = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get())

        self.logger.info("received {} results".format(len(results)))
        self.logger.info("applying federated evaluate strategy")
        ress = [(r.num_samples, r.loss, r.accuracy) for r in results]
        self.logger.debug("model samples/loss/acc for round {}: {}".format(tround, ress))
        aggregated_res = FedAvg.aggregate_evaluate(ress)
        return aggregated_res

    def get_init_weights(self, model: Model, device: Device):
        job = Job(job_type=JobType.GET_WEIGHTS,
                  model_name=model.name)

        self.logger.debug("getting initial weights from {}".format(device.id))
        self.jobs_dispatcher.send_jobs(job, [device])

        self.wait_for_responses(1)
        result: Result = self.results_queue.get()

        self.logger.info("received result")
        parameters: Parameters = result.parameters
        self.logger.debug("parameters shape: {}".format(parameters.get_shape()))
        return parameters

    def wait_for_responses(self, num_responses: int):
        self.logger.debug("waiting for responses...")
        with self.condition:
            self.condition.wait_for(lambda: self.results_queue.qsize() >= num_responses)
            self.logger.debug("responses received")

    def add_result(self, result: Result):
        self.results_queue.put(result)
        with self.condition:
            self.condition.notify_all()
        return True

    """
    
    def send_off(self, devices: List[Device]):
        for device in devices:
            device_host = device.host + ":" + str(device.port)
            self.logger.info("sending off to device {} at {}".format(device.id, device_host))
            req = requests.delete("http://" + device_host + API.status)
            if req.status_code == 200:
                self.logger.info("sent off to device {} off".format(device.id))
        self.logger.debug("off sent to devices")
    
    def results_worker(self):
        while True:
            job = self.results_queue.get()
            self.logger.info("received response for job {}".format(job.id))
            self.process_response(job)
            self.logger.info("processing response for job {} finished".format(job.id))
            self.results_queue.task_done()

    def process_response(self, job):
        pass
    """


