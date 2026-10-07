"""Best-effort Redis coordination; durable job truth never lives here."""

import logging
import time
from threading import Event
from typing import Protocol
from uuid import UUID

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from repomind.api.streaming import ProgressEvent

logger = logging.getLogger(__name__)

# Bounded so a Redis outage degrades to PostgreSQL polling instead of stalling callers:
# progress is published synchronously from the job's own trace listener.
_SOCKET_CONNECT_TIMEOUT_SECONDS = 2.0
_SOCKET_TIMEOUT_SECONDS = 5.0
_MAX_COALESCED_WAKEUPS = 1_000


class ProgressSubscription(Protocol):
    def next(self, timeout: float) -> ProgressEvent | None: ...
    def close(self) -> None: ...


class JobBroker(Protocol):
    def notify_job(self, job_id: UUID) -> None: ...
    def publish_progress(self, job_id: UUID, event: ProgressEvent) -> None: ...
    def subscribe(self, job_id: UUID) -> ProgressSubscription: ...
    def wait_for_work(self, timeout: float) -> bool: ...
    def close(self) -> None: ...


def _sleep(seconds: float) -> None:
    Event().wait(max(0.0, seconds))


def _pubsub(client: redis.Redis, channel: str) -> redis.client.PubSub:
    # Subscribe confirmations are filtered by ``_next_message``, not by redis-py: with
    # ``ignore_subscribe_messages`` a consumed confirmation also reads as ``None``, which
    # is indistinguishable from "nothing published" for a zero-timeout drain.
    pubsub = client.pubsub()
    try:
        pubsub.subscribe(channel)
    except BaseException:
        _close_quietly(pubsub)
        raise
    return pubsub


def _next_message(pubsub: redis.client.PubSub, deadline: float) -> dict | None:
    """Return the next published message before ``deadline``, or ``None``.

    A single ``get_message`` call is not a bounded wait: it returns as soon as it
    reads anything, including a subscribe confirmation. Keep reading until a real
    message arrives or nothing more is available by the deadline.
    """

    while True:
        remaining = deadline - time.monotonic()
        message = pubsub.get_message(timeout=max(0.0, remaining))
        if message is not None and message.get("type") == "message":
            return message
        if message is None and remaining <= 0:
            return None


def _close_quietly(pubsub: redis.client.PubSub) -> None:
    try:
        pubsub.close()
    except redis.RedisError:
        logger.debug("Redis Pub/Sub connection was already unusable during close")


class _RedisSubscription:
    def __init__(self, client: redis.Redis, channel: str) -> None:
        self.pubsub = _pubsub(client, channel)

    def next(self, timeout: float) -> ProgressEvent | None:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                message = _next_message(self.pubsub, deadline)
            except redis.RedisError:
                # Wait out the timeout: callers poll PostgreSQL between calls, and an
                # immediate return would turn a Redis outage into a database hot loop.
                _sleep(deadline - time.monotonic())
                return None
            if message is None:
                return None
            try:
                return ProgressEvent.model_validate_json(message["data"])
            except ValueError:
                continue  # malformed fan-out is dropped; durable state lives in PostgreSQL

    def close(self) -> None:
        _close_quietly(self.pubsub)


class RedisJobBroker:
    """Redis wakeups and Pub/Sub fan-out; failures leave jobs queued in PostgreSQL."""

    def __init__(self, url: str, *, channel_prefix: str = "repomind:jobs") -> None:
        self.url = url
        self.channel_prefix = channel_prefix
        # One pooled client per broker: no connection is opened per published event.
        self.client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=_SOCKET_CONNECT_TIMEOUT_SECONDS,
            socket_timeout=_SOCKET_TIMEOUT_SECONDS,
            # One immediate retry replaces a stale pooled connection; no backoff sleeps.
            retry=Retry(NoBackoff(), 1),
        )
        self._wakeups: redis.client.PubSub | None = None

    @property
    def _wakeup_channel(self) -> str:
        return f"{self.channel_prefix}:wakeup"

    def _channel(self, job_id: UUID) -> str:
        return f"{self.channel_prefix}:{job_id}"

    def notify_job(self, job_id: UUID) -> None:
        self._publish(self._wakeup_channel, str(job_id))

    def publish_progress(self, job_id: UUID, event: ProgressEvent) -> None:
        self._publish(self._channel(job_id), event.model_dump_json())

    def _publish(self, channel: str, payload: str) -> None:
        try:
            self.client.publish(channel, payload)
        except redis.RedisError:
            logger.warning("Redis coordination unavailable; PostgreSQL job remains durable")

    def subscribe(self, job_id: UUID) -> ProgressSubscription:
        try:
            return _RedisSubscription(self.client, self._channel(job_id))
        except redis.RedisError:
            return NullSubscription()

    def wait_for_work(self, timeout: float) -> bool:
        """Block for up to ``timeout`` seconds; ``True`` only when a wakeup arrived.

        The wakeup subscription stays open between calls, so a job queued while the
        worker was busy is still delivered and an idle worker does not reconnect on
        every poll. Only the single worker loop thread may call this.
        """

        deadline = time.monotonic() + max(0.0, timeout)
        try:
            if self._wakeups is None:
                self._wakeups = _pubsub(self.client, self._wakeup_channel)
            if _next_message(self._wakeups, deadline) is None:
                return False
            # Coalesce wakeups that queued up while busy into a single claim attempt.
            for _ in range(_MAX_COALESCED_WAKEUPS):
                if _next_message(self._wakeups, time.monotonic()) is None:
                    break
            return True
        except redis.RedisError:
            logger.warning("Redis wakeups unavailable; falling back to PostgreSQL polling")
            self._reset_wakeups()
            _sleep(deadline - time.monotonic())
            return False

    def _reset_wakeups(self) -> None:
        if self._wakeups is not None:
            _close_quietly(self._wakeups)
            self._wakeups = None

    def close(self) -> None:
        self._reset_wakeups()
        try:
            self.client.close()
        except redis.RedisError:
            logger.debug("Redis client was already unusable during close")


class NullSubscription:
    def next(self, timeout: float) -> ProgressEvent | None:
        _sleep(timeout)
        return None

    def close(self) -> None:
        return None
