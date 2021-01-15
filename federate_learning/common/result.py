from federate_learning.common.job import JobType
from federate_learning.common.parameters import Parameters


class Result:
    def __init__(self,
                 job_id: str = None,
                 job_type: JobType = None,
                 model_name: str = None,
                 parameters: Parameters = None,
                 num_samples: int = None,
                 configuration: dict = None,
                 accuracy: float = None,
                 loss: float = None,
                 dict_data=None):
        if dict_data:
            self.__dict__ = dict_data
            if self.parameters:
                self.parameters = Parameters(list_of_values=self.parameters)
        else:
            self.job_id = job_id
            self.job_type = job_type
            self.model_name = model_name
            self.parameters = parameters
            self.num_samples = num_samples
            self.configuration = configuration
            self.accuracy = accuracy
            self.loss = loss

    def to_json(self):
        parameters_list = None
        if self.parameters:
            parameters_list = self.parameters.to_list()
        return {
            "job_id": self.job_id,
            "job_type": int(self.job_type),
            "model_name": self.model_name,
            "parameters": parameters_list,
            "num_samples": self.num_samples,
            "configuration": self.configuration,
            "accuracy": self.accuracy,
            "loss": self.loss
        }
