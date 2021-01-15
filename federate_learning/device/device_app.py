import coloredlogs, logging
from flask import Flask, request
from federate_learning.common import API, Job
from federate_learning.device import Device


class DeviceApp(Flask):
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
        self.add_url_rule('/', methods=["GET"], view_func=self._get_status)
        self.add_url_rule(API.status, methods=["GET"], view_func=self._get_status)
        self.add_url_rule(API.status, methods=["DELETE"], view_func=self._leave)
        self.add_url_rule(API.jobs, methods=["POST"], view_func=self._add_job)

        # device variables
        self.device = Device(logger=logger)

    def _get_status(self):
        return {"status": self.device.status}, 200

    def _leave(self):
        self.device.leave()
        return {"status": self.device.status}, 200

    def _add_job(self):
        data = request.get_json()
        job = Job(dict_data=data)
        logging.info("adding job id {}".format(job.id))
        self.device.add_job(job)
        return {"result": "ok"}, 200

    def run(self, debug=None, load_dotenv=True, **options):
        # join the device pool
        if self.device.join():
            logging.info("joined")
            # start the server
            super().run(host=self.device.host, port=self.device.port,
                        debug=debug, load_dotenv=load_dotenv, threaded=False, **options)
            logging.info("server closed, leave")
            self.device.leave()
            logging.info("leaved")
        else:
            logging.error("error join")
