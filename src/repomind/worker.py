"""Run the durable RepoMind worker with ``python -m repomind.worker``."""

from threading import Event

from repomind.api.dependencies import ServiceContainer
from repomind.config import get_settings
from repomind.jobs.worker import JobWorker


def main() -> None:
    settings = get_settings()
    container = ServiceContainer(settings=settings)
    services = container.get()
    worker = JobWorker(
        services.jobs.store,
        services.jobs.broker,
        services.execution,
        lease_seconds=settings.job_lease_seconds,
    )
    stop = Event()
    try:
        while not stop.is_set():
            if not worker.run_once():
                services.jobs.broker.wait_for_work(settings.job_poll_seconds)
    except KeyboardInterrupt:
        stop.set()
    finally:
        container.close()


if __name__ == "__main__":
    main()
