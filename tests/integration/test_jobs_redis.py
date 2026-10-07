"""Real Redis Pub/Sub coverage for best-effort durable-job coordination."""

import os
import threading
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


def _isolated_broker(url: str) -> RedisJobBroker:
    # Pub/Sub channels ignore the database number; a unique prefix keeps
    # concurrent test runs from waking each other.
    return RedisJobBroker(url, channel_prefix=f"repomind-test:{uuid4().hex}")


def test_idle_wait_for_work_blocks_for_the_full_timeout(redis_test_url: str):
    broker = _isolated_broker(redis_test_url)
    try:
        for _ in range(2):  # the first call also consumes the subscribe confirmation
            started = time.monotonic()
            assert broker.wait_for_work(0.5) is False
            assert time.monotonic() - started >= 0.4
    finally:
        broker.close()


def test_wait_for_work_returns_promptly_on_a_wakeup(redis_test_url: str):
    broker = _isolated_broker(redis_test_url)
    publisher = RedisJobBroker(redis_test_url, channel_prefix=broker.channel_prefix)
    try:
        assert broker.wait_for_work(0.05) is False
        timer = threading.Timer(0.1, publisher.notify_job, args=(uuid4(),))
        timer.start()
        started = time.monotonic()
        assert broker.wait_for_work(5.0) is True
        assert time.monotonic() - started < 1.0
        timer.join()

        # A wakeup published while the worker was busy is not lost, and a burst of
        # wakeups is coalesced into one claim attempt.
        publisher.notify_job(uuid4())
        publisher.notify_job(uuid4())
        time.sleep(0.05)
        started = time.monotonic()
        assert broker.wait_for_work(5.0) is True
        assert time.monotonic() - started < 0.5
        assert broker.wait_for_work(0.2) is False
    finally:
        publisher.close()
        broker.close()


def test_progress_subscription_waits_out_its_timeout_when_redis_fails(
    redis_test_url: str, monkeypatch
):
    broker = _isolated_broker(redis_test_url)
    subscription = broker.subscribe(uuid4())
    try:

        def unavailable(*args, **kwargs):
            raise redis.ConnectionError("Redis went away")

        monkeypatch.setattr(subscription.pubsub, "get_message", unavailable)
        started = time.monotonic()
        assert subscription.next(timeout=0.3) is None
        assert time.monotonic() - started >= 0.25
    finally:
        subscription.close()
        broker.close()


def test_unreachable_redis_degrades_to_bounded_polling_waits():
    broker = RedisJobBroker("redis://127.0.0.1:1/0")
    try:
        started = time.monotonic()
        assert broker.wait_for_work(0.3) is False
        assert broker.subscribe(uuid4()).next(0.2) is None
        broker.notify_job(uuid4())
        assert 0.45 <= time.monotonic() - started < 5.0
    finally:
        broker.close()
