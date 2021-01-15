import coloredlogs, logging
import threading
from flask import Flask, request
from federate_learning.common import API
from federate_learning.common.result import Result
from federate_learning.orchestrator import Orchestrator
from federate_learning.device import Device


class OrchestratorApp(Flask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # init log
        log_format = "%(asctime)s:%(hostname)s:%(message)s"
        logging.basicConfig(level='DEBUG', format=log_format)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logger = logging.getLogger(__name__)
        coloredlogs.install(level='DEBUG', logger=logger, fmt=log_format, datefmt="%H:%M:%S")

        # endpoints definition
        self.add_url_rule('/', view_func=self._get_status)
        self.add_url_rule(API.devices, methods=["GET"], view_func=self._get_devices)
        self.add_url_rule(API.device, methods=["GET"], view_func=self._get_device)
        self.add_url_rule(API.device, methods=["POST"], view_func=self._join_device)
        self.add_url_rule(API.device, methods=["DELETE"], view_func=self._leave_device)
        self.add_url_rule(API.jobs, methods=["POST"], view_func=self._add_result)

        # orchestrator variables
        self.orchestrator = Orchestrator(logger=logger)

    def _get_status(self):
        return {"status": self.orchestrator.status}

    def _join_device(self, device_id):
        data = request.get_json()
        logging.info("join device: {}, with data: {}".format(device_id, data))
        device = Device(dict_data=data)
        logging.info("adding device: {}".format(device.to_json()))
        if self.orchestrator.device_manager.add_device(device):
            return {"result": "ok"}, 200
        else:
            return {"result": "error"}, 500

    def _leave_device(self, device_id):
        if self.orchestrator.device_manager.remove_device(device_id):
            return {"result": "ok"}, 200
        else:
            return {"result": "error"}, 500

    def _get_device(self, device_id):
        device, _ = self.orchestrator.device_manager.get_device(device_id)
        return {"device": device.__dict__}

    def _get_devices(self):
        devices = self.orchestrator.device_manager.get_devices()
        return {"devices": [d.__dict__ for d in devices]}

    def _add_result(self):
        data = request.get_json()
        result = Result(dict_data=data)
        self.logger.debug("adding response for job {}, type {}".format(result.job_id, result.job_type))
        self.orchestrator.add_result(result)
        return {"result": "ok"}, 200

    def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
        logging.info("start training thread")
        threading.Thread(target=self.orchestrator.training).start()
        logging.info("start server")
        super().run(host=host, port=port, debug=debug, load_dotenv=load_dotenv, threaded=False, **options)
