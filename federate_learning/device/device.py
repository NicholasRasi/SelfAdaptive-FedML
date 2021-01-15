import uuid
import requests
from typing import List
import threading
import queue
from federate_learning.common import Parameters, Job, JobType, API, Model, Result
from federate_learning.device.results_dispatcher import ResultsDispatcher


class Device:
    def __init__(self,
                 host: str = None,
                 port: int = None,
                 server_host: str = None,
                 available_models: List[Model] = None,
                 logger=None,
                 dict_data: dict = None):
        if dict_data:
            self.__dict__ = dict_data
        else:
            self.status = "active"
            self.id: str = str(uuid.uuid4())
            self.host: str = host
            self.port: int = port
            self.client_capabilities = {}  # todo
            self.logger = logger
            self.models_name_list = []
            self.available_models = []
            self.server_host: str = server_host
            self.jobs_queue = queue.Queue()
            self.jobs_thread = threading.Thread(target=self.jobs_worker)
            self.results_dispatcher = ResultsDispatcher(server_host=self.server_host, logger=self.logger)

    def config(self, host: str, port: int, server_host: str, available_models: List[Model]):
        self.host = str(host)
        self.port = int(port)
        self.server_host = "http://" + str(server_host)
        self.results_dispatcher.server_host = self.server_host
        self.models_name_list = [model.name for model in self.available_models]
        self.available_models = available_models
        # set log for models
        for model in self.available_models:
            model.logger = self.logger
        self.jobs_thread.start()

    def join(self):
        data = self.to_json()
        req = requests.post(self.server_host + API.devices + "/" + self.id, json=data)
        return req.status_code == 200

    def leave(self):
        self.logger.info("leaving...")
        self.jobs_queue.put("STOPFLAG")
        self.status = "inactive"
        req = requests.delete(self.server_host + API.devices + "/" + self.id)
        return req.status_code == 200

    def to_json(self):
        return {"status": self.status,
                "id": self.id,
                "host": self.host,
                "port": self.port,
                "client_capabilities": self.client_capabilities,
                "models_name_list": self.models_name_list}

    def add_job(self, job: Job):
        self.jobs_queue.put(job)

    def jobs_worker(self):
        running = True
        while running:
            job = self.jobs_queue.get()
            if job == "STOPFLAG":
                running = False
            else:
                self.logger.info("working on job {}".format(job.id))
                result = self.process_job(job)
                self.logger.debug("sending job {} result".format(job.id))
                if self.results_dispatcher.send_result(result):
                    self.logger.info("job {} result sent".format(job.id))
                else:
                    self.logger.warning("error sending result for job {}".format(job.id))
                self.jobs_queue.task_done()
                self.logger.info("job {} finished".format(job.id))

    def process_job(self, job):
        model = list(filter(lambda m: job.model_name == m.name, self.available_models))[0]
        result = Result(job_id=job.id, job_type=job.job_type, model_name=job.model_name, configuration=job.configuration)

        if job.job_type == JobType.GET_WEIGHTS:
            self.logger.debug("getting weights")
            weights = model.get_weights()
            result.parameters = Parameters(ndarray_list=weights)
            self.logger.debug("parameters shape: {}".format(result.parameters.get_shape()))
        elif job.job_type == JobType.FIT:
            self.logger.debug("fitting model")
            parameters = job.parameters
            self.logger.debug("parameters shape: {}".format(parameters.get_shape()))
            weights, num_samples, mean_loss, mean_acc = model.fit(weights=parameters.weights, config=job.configuration)
            result.parameters = Parameters(ndarray_list=weights)
            result.num_samples = num_samples
            result.loss = mean_loss
            result.accuracy = mean_acc
        elif job.job_type == JobType.EVALUATE:
            self.logger.debug("evaluating model")
            parameters = job.parameters
            self.logger.debug("parameters shape: {}".format(parameters.get_shape()))
            len_x, loss, accuracy = model.evaluate(weights=parameters.weights, config=job.configuration)
            result.num_samples = len_x
            result.loss = loss
            result.accuracy = accuracy
        return result
