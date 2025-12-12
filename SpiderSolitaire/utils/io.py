import queue
import threading


def timed_input(timeout: int=60, prompt: str='') -> str|None:
    """Read input with optional timeout. 
        Returns None if timeout expires, entered string if input was provided in time."""
    q = queue.Queue()
    def reader():
        try:
            q.put(input(prompt))
        except EOFError:
            q.put(None)

    t = threading.Thread(target=reader)
    t.daemon = True
    t.start()

    try:
        return q.get(timeout=timeout)
    except:
        return None