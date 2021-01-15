import uuid
from enum import IntEnum
from federate_learning.common.parameters import Parameters


class JobType(IntEnum):
    GET_WEIGHTS = 1
    FIT = 2
    EVALUATE = 3


class Job:
    def __init__(self,
                 job_type: JobType = None,
                 model_name: str = None,
                 parameters: Parameters = None,
                 configuration: dict = None,
                 dict_data=None):
        if dict_data:
            self.__dict__ = dict_data
            if self.job_type == JobType.GET_WEIGHTS:
                self.parameters = None
            else:
                self.parameters = Parameters(list_of_values=self.parameters)
        else:
            self.job_type = job_type
            self.model_name = model_name
            self.parameters = parameters
            self.configuration = configuration
        self.id: str = str(uuid.uuid4())

    def to_json(self):
        parameters_list = None
        if self.parameters:
            parameters_list = self.parameters.to_list()
        return {
            "id": self.id,
            "job_type": int(self.job_type),
            "model_name": self.model_name,
            "parameters": parameters_list,
            "configuration": self.configuration
        }
