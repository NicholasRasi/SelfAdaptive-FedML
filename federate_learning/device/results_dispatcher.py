import requests
from federate_learning.common import Result, API


class ResultsDispatcher:
    def __init__(self, server_host, logger):
        self.server_host = server_host
        self.logger = logger

    def send_result(self, result: Result):
        req = requests.post(self.server_host + API.jobs, json=result.to_json())
        return req.status_code == 200
