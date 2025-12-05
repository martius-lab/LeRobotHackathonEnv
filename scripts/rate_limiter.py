import time
class RateLimiter:
    def __init__(self, calls_per_second):
        self.min_interval = 1.0 / (calls_per_second + 1)
        self.last_call_time = 0.0

    def wait(self):
        current_time = time.time()
        next_allowed_time = self.last_call_time + self.min_interval
        wait_time = next_allowed_time - current_time
        if wait_time > 0:
            time.sleep(wait_time)
            current_time = time.time()
        self.last_call_time = current_time
