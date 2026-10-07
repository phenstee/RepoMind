"""Run the durable RepoMind worker with ``python -m repomind.worker``."""

import logging
import signal
import time
from threading import Event

from repomind.api.dependencies import ServiceContainer
from repomind.config import get_settings
from repomind.jobs.broker import JobBroker
from repomind.jobs.worker import JobWorker

logger = logging.getLogger(__name__)

_INITIAL_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 30.0
# Upper bound on how long an idle worker takes to notice a shutdown request.
_SHUTDOWN_CHECK_SECONDS = 1.0


def install_shutdown_handler(stop: Event) -> None:
    """SIGTERM lets the current job finish, then exits; it never interrupts a job."""

    def request_stop(signum, frame) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)


def wait_for_work(broker: JobBroker, timeout: float, stop: Event) -> None:
    """Idle until a wakeup, the poll interval, or a shutdown request, whichever is first."""

    deadline = time.monotonic() + timeout
    while not stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0 or broker.wait_for_work(min(remaining, _SHUTDOWN_CHECK_SECONDS)):
            return


def run(worker: JobWorker, broker: JobBroker, stop: Event, *, poll_seconds: float) -> None:
    """Poll for durable jobs until ``stop`` is set, surviving transient failures."""

    backoff = _INITIAL_BACKOFF_SECONDS
    while not stop.is_set():
        try:
            if not worker.run_once():
                wait_for_work(broker, poll_seconds, stop)
        except Exception:
            # PostgreSQL/Redis outages are expected; the worker must outlive them.
            logger.exception("Worker iteration failed; retrying in %.0fs", backoff)
            stop.wait(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
        else:
            backoff = _INITIAL_BACKOFF_SECONDS


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
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
    install_shutdown_handler(stop)
    logger.info("Worker %s started", worker.worker_id)
    try:
        run(worker, services.jobs.broker, stop, poll_seconds=settings.job_poll_seconds)
    except KeyboardInterrupt:
        stop.set()
    finally:
        container.close()
    logger.info("Worker %s stopped", worker.worker_id)


if __name__ == "__main__":
    main()
