import time
from multiprocessing.managers import BaseManager
from threading import Lock


class CallingThrottle:
    def __init__(self, nb_call_times_limit: int, expired_time: float):
        self.nb_call_times_limit = nb_call_times_limit
        self.expired_time = expired_time
        self.called_timestamps = list()
        self.lock = Lock()

    def throttle(self):
        with self.lock:
            while len(self.called_timestamps) == self.nb_call_times_limit:
                now = time.time()
                self.called_timestamps = list(
                    filter(
                        lambda x: now - x < self.expired_time,
                        self.called_timestamps,
                    )
                )
                if len(self.called_timestamps) == self.nb_call_times_limit:
                    time_to_sleep = (
                        self.called_timestamps[0] + self.expired_time - now
                    )
                    time.sleep(time_to_sleep)
            self.called_timestamps.append(time.time())


# A "managed" CallingThrottle is required for use with multiprocessing:
class CallingThrottleManager(BaseManager):
    pass


CallingThrottleManager.register("calling_throttle", CallingThrottle)
