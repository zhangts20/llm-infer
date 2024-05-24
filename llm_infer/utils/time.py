import time

__all__ = ["calculate_time"]


def calculate_time(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Function {func.__name__} took {round(end_time - start_time, 4)} seconds to execute"
        )
        return result

    return wrapper
