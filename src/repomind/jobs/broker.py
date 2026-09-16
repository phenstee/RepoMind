"""Best-effort Redis coordination; durable job truth never lives here."""

import logging
from typing import Protocol
from uuid import UUID

import redis

from repomind.api.streaming import ProgressEvent

logger = logging.getLogger(__name__)


class ProgressSubscription(Protocol):
    def next(self, timeout: float) -> ProgressEvent | None: ...
    def close(self) -> None: ...


class JobBroker(Protocol):
    def notify_job(self, job_id: UUID) -> None: ...
    def publish_progress(self, job_id: UUID, event: ProgressEvent) -> None: ...
    def subscribe(self, job_id: UUID) -> ProgressSubscription: ...
    def wait_for_work(self, timeout: float) -> None: ...


class _RedisSubscription:
    def __init__(self, client: redis.Redis, channel: str) -> None:
        self.pubsub = client.pubsub(ignore_subscribe_messages=True)
        self.pubsub.subscribe(channel)

    def next(self, timeout: float) -> ProgressEvent | None:
        try:
            message = self.pubsub.get_message(timeout=timeout)
            if message is None or message.get("type") != "message":
                return None
            return ProgressEvent.model_validate_json(message["data"])
        except (redis.RedisError, ValueError):
            return None

    def close(self) -> None:
        self.pubsub.close()


class RedisJobBroker:
    """Redis wakeups and Pub/Sub fan-out; failures leave jobs queued in PostgreSQL."""

    def __init__(self, url: str) -> None:
        self.url = url

    def _client(self) -> redis.Redis:
        return redis.Redis.from_url(self.url, decode_responses=True)

    @staticmethod
    def _channel(job_id: UUID) -> str:
        return f"repomind:jobs:{job_id}"

    def notify_job(self, job_id: UUID) -> None:
        self._publish("repomind:jobs:wakeup", str(job_id))

    def publish_progress(self, job_id: UUID, event: ProgressEvent) -> None:
        self._publish(self._channel(job_id), event.model_dump_json())

    def _publish(self, channel: str, payload: str) -> None:
        try:
            self._client().publish(channel, payload)
        except redis.RedisError:
            logger.warning("Redis coordination unavailable; PostgreSQL job remains durable")

    def subscribe(self, job_id: UUID) -> ProgressSubscription:
        try:
            return _RedisSubscription(self._client(), self._channel(job_id))
        except redis.RedisError:
            return NullSubscription()

    def wait_for_work(self, timeout: float) -> None:
        pubsub = None
        try:
            pubsub = self._client().pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe("repomind:jobs:wakeup")
            pubsub.get_message(timeout=timeout)
        except redis.RedisError:
            NullSubscription().next(timeout)
        finally:
            if pubsub is not None:
                pubsub.close()


class NullSubscription:
    def next(self, timeout: float) -> ProgressEvent | None:
        from threading import Event

        Event().wait(timeout)
        return None

    def close(self) -> None:
        return None
