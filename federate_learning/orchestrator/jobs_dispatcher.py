import requests
import concurrent.futures
from typing import List
from federate_learning.common import Job, API
from federate_learning.device import Device


class JobsDispatcher:
    def __init__(self, logger):
        self.logger = logger

    def send_jobs(self, job: Job, devices: List[Device]):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for device in devices:
                device_host = device.host + ":" + str(device.port)
                futures.append(executor.submit(self.send_job, job=job, device_host=device_host))
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    pass
                    # self.logger.debug("job sent")
        self.logger.debug("jobs sent to devices")

    def send_job(self, job: Job, device_host: str):
        # self.logger.debug("sending job to {}...".format(device_host))
        req = requests.post("http://" + device_host + API.jobs, json=job.to_json())
        return req.status_code == 200
