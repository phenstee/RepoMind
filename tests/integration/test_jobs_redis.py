"""Real Redis Pub/Sub coverage for best-effort durable-job coordination."""

import os
import time
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import pytest
import redis

from repomind.api.streaming import ProgressEvent
from repomind.jobs.broker import RedisJobBroker

pytestmark = pytest.mark.redis
_WAKEUP_CHANNEL = "repomind:jobs:wakeup"


def _database_15(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit(parsed._replace(path="/15"))


@pytest.fixture
def redis_test_url() -> str:
    configured = os.environ.get("REDIS_TEST_URL")
    if not configured:
        pytest.skip("set REDIS_TEST_URL to run real Redis tests")
    url = _database_15(configured)
    client = redis.Redis.from_url(url, decode_responses=True)
    try:
        client.ping()
    except redis.RedisError as exc:
        pytest.fail(f"configured Redis test database is unavailable: {exc}")
    finally:
        client.close()
    return url


def _message(pubsub: redis.client.PubSub, *, timeout: float = 1.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        message = pubsub.get_message(timeout=max(0.0, deadline - time.monotonic()))
        if message is not None and message.get("type") == "message":
            return message
        if time.monotonic() >= deadline:
            pytest.fail("Redis Pub/Sub message was not received before the bounded timeout")


def _subscription(pubsub: redis.client.PubSub) -> None:
    deadline = time.monotonic() + 1.0
    while True:
        message = pubsub.get_message(timeout=max(0.0, deadline - time.monotonic()))
        if message is not None and message.get("type") == "subscribe":
            return
        if time.monotonic() >= deadline:
            pytest.fail("Redis subscription was not acknowledged before the bounded timeout")


def test_redis_wakeup_notification_reaches_worker_coordination_channel(redis_test_url: str):
    client = redis.Redis.from_url(redis_test_url, decode_responses=True)
    pubsub = client.pubsub()
    try:
        pubsub.subscribe(_WAKEUP_CHANNEL)
        _subscription(pubsub)
        job_id = uuid4()
        RedisJobBroker(redis_test_url).notify_job(job_id)
        message = _message(pubsub)
        assert message["channel"] == _WAKEUP_CHANNEL
        assert message["data"] == str(job_id)
    finally:
        pubsub.close()
        client.close()


def test_redis_progress_fanout_round_trips_safe_event_metadata(redis_test_url: str):
    broker = RedisJobBroker(redis_test_url)
    job_id = uuid4()
    event = ProgressEvent(
        run_id=uuid4(),
        sequence=1,
        event="index.started",
        timestamp=datetime.now(UTC),
        data={"repository_id": 42},
    )
    subscription = broker.subscribe(job_id)
    try:
        subscription.next(timeout=0.1)
        broker.publish_progress(job_id, event)
        received = subscription.next(timeout=1.0)
    finally:
        subscription.close()

    assert received == event
    assert received.data == {"repository_id": 42}
    serialized = received.model_dump_json()
    assert "password" not in serialized
    assert "C:\\" not in serialized
