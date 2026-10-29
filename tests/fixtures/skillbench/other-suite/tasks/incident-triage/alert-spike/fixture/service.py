"""Checkout worker connection pool."""


class ConnectionPool:
    def __init__(self, max_connections: int = 16) -> None:
        self.max_connections = max_connections
        self.in_use = 0

    def acquire(self) -> int:
        if self.in_use >= self.max_connections:
            raise RuntimeError("503 pool exhausted")
        self.in_use += 1
        return self.in_use

    def release(self) -> None:
        self.in_use = max(0, self.in_use - 1)


def handle_checkout(pool: ConnectionPool, timeout: bool = False) -> str:
    pool.acquire()
    if timeout:
        raise TimeoutError("upstream gateway timed out")
    pool.release()
    return "200 OK"
