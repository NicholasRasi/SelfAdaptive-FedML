from typing import List
import random
from federate_learning.device import Device


class DeviceManager:
    def __init__(self):
        self.devices: List[Device] = []

    def get_device(self, device_id: str,):
        for i, device in enumerate(self.devices):
            if device.id == device_id:
                return device, i
        return None, -1

    def get_devices(self) -> List:
        return self.devices

    def num_devices(self) -> int:
        return len(self.devices)

    def add_device(self, device: Device) -> bool:
        # check if not present
        dev, i = self.get_device(device.id)
        if not dev:
            self.devices.append(device)
            return True
        else:
            return False

    def remove_device(self, device_id: str) -> bool:
        device, i = self.get_device(device_id)
        if device:
            del self.devices[i]
            return True
        else:
            return False

    def get_sample(self, num_devices: int, model_name: str = None) -> List[Device]:
        # todo: get device with the available model
        return random.sample(self.devices, num_devices)
