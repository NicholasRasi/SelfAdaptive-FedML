import threading
import time
import queue
from typing import List, Tuple
from federate_learning.device import Device
from federate_learning.orchestrator import DeviceManager, JobsDispatcher
from federate_learning.common import JobType, Job, Parameters, Result, Model
from federate_learning.orchestrator.aggregation_strategy import FedAvg
from federate_learning.orchestrator.control_strategy import ControlType, ControlStrategy
from federate_learning.orchestrator.control_strategy.dynamic_1 import Dynamic1
from federate_learning.orchestrator.control_strategy.static import Static


class Orchestrator:
    def __init__(self, logger):
        self.logger = logger
        self.status = "active"
        self.num_rounds = None
        self.target_accuracy = None
        self.min_devices = None
        self.k_fit = None
        self.k_eval = None
        self.num_epochs = None
        self.batch_size = None
        self.jobs_dispatcher = JobsDispatcher(logger=self.logger)
        self.results_queue = queue.Queue()
        self.device_manager = DeviceManager()
        self.condition = threading.Condition()
        self.available_models = {}
        self.control_strategy_fun = None

    def config(self,
               available_models,
               num_rounds=10,
               min_devices=2,
               k_fit=1,
               k_eval=1,
               num_epochs=10,
               batch_size=32,
               control_type=ControlType.STATIC):
        self.available_models = available_models
        self.num_rounds = int(num_rounds)
        self.min_devices = int(min_devices)

        if control_type == ControlType.DYNAMIC_1:
            control_strategy = Dynamic1(num_epochs=num_epochs,
                                        batch_size=batch_size,
                                        num_rounds=self.num_rounds,
                                        k_fit=k_fit,
                                        k_eval=k_eval,
                                        logger=self.logger)
        else:
            # control_type == ControlType.STATIC:
            control_strategy = Static(num_epochs=num_epochs,
                                      batch_size=batch_size,
                                      num_rounds=self.num_rounds,
                                      k_fit=k_fit,
                                      k_eval=k_eval,
                                      logger=self.logger)
        self.control_strategy_fun = control_strategy.apply_strategy


    def training(self):
        self.logger.debug("waiting min {} devices available".format(self.min_devices))
        while self.device_manager.num_devices() < self.min_devices:
            time.sleep(2)
        self.logger.info("start federated learning R={}, K(train)={}, K(eval)={}".format(self.num_rounds,
                                                                                         self.k_fit,
                                                                                         self.k_eval))

        for model in self.available_models:
            self.logger.info("starting training for model {}".format(model.name))

            self.logger.debug("getting init weights from a random client")
            device = self.device_manager.get_sample(num_devices=1, model_name=None)[0]
            weights = self.get_init_weights(model=model, device=device)

            self.logger.debug("updating model weights")
            model.weights = weights

            for tround in range(1, self.num_rounds + 1):
                self.logger.info("round {}/{}".format(tround, self.num_rounds))

                self.logger.info("applying control strategy")
                k_fit, k_eval, device_config = self.control_strategy_fun(model, tround)
                self.logger.info("K(fit)={}, K(eval)={}, config={}".format(k_fit, k_eval, device_config))

                num_devices_fit = int(self.device_manager.num_devices() * k_fit)
                self.logger.debug("selecting {} random devices for fit".format(num_devices_fit))
                devices_fit = self.device_manager.get_sample(model_name=model.name,
                                                             num_devices=num_devices_fit)
                self.logger.info("selected random client sample: {}".format([d.id for d in devices_fit]))

                self.logger.info("fitting model")
                new_weights, aggregated_loss, \
                aggregated_acc, aggregated_cost = self.fit_model(model=model,
                                                                 devices=devices_fit,
                                                                 device_config=device_config)

                self.logger.debug("updating model weights and appending loss/acc, device config")
                model.weights = Parameters(ndarray_list=new_weights)
                model.metrics.train_losses.append(aggregated_loss)
                model.metrics.train_accuracies.append(aggregated_acc)
                model.metrics.costs.append(aggregated_cost)
                model.metrics.device_configs.append(device_config)

                num_devices_eval = int(self.device_manager.num_devices() * k_eval)
                self.logger.debug("selecting {} random devices for eval".format(num_devices_eval))
                devices_eval = self.device_manager.get_sample(model_name=model.name, num_devices=num_devices_eval)
                self.logger.info("selected random client sample: {}".format([d.id for d in devices_eval]))
                aggregated_loss, aggregated_acc = self.evaluate_model(model=model, devices=devices_eval, tround=tround)
                model.metrics.eval_losses.append(aggregated_loss)
                model.metrics.eval_accuracies.append(aggregated_acc)

            self.print_model_metrics(model=model)

    def print_model_metrics(self, model: Model):
        self.logger.info("training for model {} finished".format(model.name))
        self.logger.info("target accuracy {0:0.4f}".format(model.target_accuracy))
        self.logger.info("train losses: {}".format(["{0:0.4f}".format(l) for l in model.metrics.train_losses]))
        self.logger.info("train accuracies: {}".format(["{0:0.4f}".format(a) for a in model.metrics.train_accuracies]))
        self.logger.info("eval losses: {}".format(["{0:0.4f}".format(l) for l in model.metrics.eval_losses]))
        self.logger.info("eval accuracies: {}".format(["{0:0.4f}".format(a) for a in model.metrics.eval_accuracies]))
        self.logger.info("costs: {}, total: {:0.2f}".format(["{0:0.2f}".format(c) for c in model.metrics.costs],
                                                            sum(model.metrics.costs)))
        self.logger.info("dev_configs: {}".format(model.metrics.device_configs))

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
