"""Multiplex API-server probe isolation and context propagation regressions.

These tests do not open sockets and do not require a Hermes gateway process.
The synthetic secret_scope follows the actual ContextVar contract.
"""

from __future__ import annotations

import sys
import threading
import time
from contextvars import ContextVar
from types import ModuleType

import pytest

import talk_apiserver
import talk_config


class _Response:
    status_code = 200
    text = ""

    def json(self):
        return {"features": {"run_submission": True}}


@pytest.fixture
def profile_scope(monkeypatch):
    """A fake Hermes profile scope, including its fail-closed secret resolver."""
    state = ContextVar("test_talk_profile_secret_scope", default=None)

    class UnscopedSecretError(RuntimeError):
        def __init__(self, name, detail=""):
            super().__init__(f"unscoped profile secret: {name}")

    def get_secret(name, default=None):
        current = state.get()
        if current is None:
            raise UnscopedSecretError(name)
        return current[1].get(name, default)

    scoped_module = ModuleType("agent.secret_scope")
    scoped_module.UnscopedSecretError = UnscopedSecretError
    scoped_module.is_multiplex_active = lambda: True
    scoped_module.current_secret_scope = lambda: (state.get() or (None, None))[1]
    scoped_module.current_secret_scope_home = lambda: (
        f"/fake/profiles/{state.get()[0]}" if state.get() else None
    )
    if "agent" not in sys.modules:
        parent = ModuleType("agent")
        parent.__path__ = []
        monkeypatch.setitem(sys.modules, "agent", parent)
    monkeypatch.setitem(sys.modules, "agent.secret_scope", scoped_module)
    monkeypatch.setattr(talk_config, "_hermes_get_secret", get_secret)
    monkeypatch.setattr(talk_apiserver, "_lane_enabled", lambda: True)
    talk_apiserver.reset_for_tests()
    yield state
    talk_apiserver.reset_for_tests()


def _profile(state, name, key="shared-key"):
    state.set((name, {
        "TALK_API_SERVER_URL": "http://127.0.0.1:8642",
        "TALK_API_SERVER_KEY": key,
    }))


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "timed out waiting for background probe"


def test_background_refresh_carries_the_invoking_secret_scope(monkeypatch, profile_scope):
    seen = []
    completed = threading.Event()

    def request(url, **kwargs):
        seen.append((
            profile_scope.get()[0],
            url,
            kwargs["headers"].get("Authorization"),
        ))
        completed.set()
        return _Response()

    monkeypatch.setattr(talk_apiserver.httpx, "get", request)
    _profile(profile_scope, "migi-discord", "voice-only-credential")

    cold = talk_apiserver.status()
    assert cold.available is False
    assert cold.detail == talk_apiserver.CHECKING_DETAIL
    assert completed.wait(2.0)
    _wait_for(lambda: talk_apiserver.status().available)

    assert seen == [(
        "migi-discord",
        "http://127.0.0.1:8642/v1/capabilities",
        "Bearer voice-only-credential",
    )]


def test_cache_is_partitioned_even_when_profiles_share_endpoint_and_key(
    monkeypatch, profile_scope
):
    calls = []

    def make_probe():
        name = profile_scope.get()[0]
        calls.append(name)
        return talk_apiserver.ApiServerStatus(
            available=name == "migi-discord",
            reason=(
                talk_apiserver.REASON_OK if name == "migi-discord"
                else talk_apiserver.REASON_ERROR
            ),
            detail=name,
        )

    monkeypatch.setattr(talk_apiserver, "probe", make_probe)

    _profile(profile_scope, "migi-discord")
    assert talk_apiserver.warm().available is True

    _profile(profile_scope, "work-agent")
    assert talk_apiserver.status().available is False  # cold, never borrows Migi's verdict
    assert talk_apiserver.warm().available is False

    _profile(profile_scope, "migi-discord")
    assert talk_apiserver.status().available is True
    _profile(profile_scope, "work-agent")
    assert talk_apiserver.status().available is False
    assert calls.count("migi-discord") == 1


def test_refreshes_for_different_profiles_do_not_block_each_other(
    monkeypatch, profile_scope
):
    started_a = threading.Event()
    started_b = threading.Event()
    release_a = threading.Event()

    def probe():
        name = profile_scope.get()[0]
        if name == "migi-discord":
            started_a.set()
            assert release_a.wait(2.0), "A probe was never released"
        else:
            started_b.set()
        return talk_apiserver.ApiServerStatus(True, talk_apiserver.REASON_OK, name)

    monkeypatch.setattr(talk_apiserver, "probe", probe)
    try:
        _profile(profile_scope, "migi-discord")
        assert talk_apiserver.status().available is False
        assert started_a.wait(2.0)
        _profile(profile_scope, "work-agent")
        assert talk_apiserver.status().available is False
        assert started_b.wait(2.0), "B's probe was blocked by A's in-flight refresh"
    finally:
        release_a.set()
    _profile(profile_scope, "migi-discord")
    _wait_for(lambda: talk_apiserver.status().available)


def test_unscoped_caller_never_reuses_a_profile_verdict(monkeypatch, profile_scope):
    monkeypatch.setattr(
        talk_apiserver, "probe",
        lambda: talk_apiserver.ApiServerStatus(True, talk_apiserver.REASON_OK, "ok"),
    )
    _profile(profile_scope, "migi-discord")
    assert talk_apiserver.warm().available is True

    profile_scope.set(None)
    cold = talk_apiserver.status()
    assert cold.available is False
    assert cold.reason == talk_apiserver.REASON_ERROR
    assert "UnscopedSecretError" in cold.detail
    assert talk_apiserver.warm().available is False


def test_key_rotation_changes_cache_identity(monkeypatch, profile_scope):
    calls = []

    def probe():
        calls.append(talk_config.api_server_key())
        return talk_apiserver.ApiServerStatus(True, talk_apiserver.REASON_OK, "ok")

    monkeypatch.setattr(talk_apiserver, "probe", probe)
    _profile(profile_scope, "migi-discord", key="first")
    old_key = talk_apiserver._cache_key()
    assert talk_apiserver.warm().available

    _profile(profile_scope, "migi-discord", key="second")
    new_key = talk_apiserver._cache_key()
    assert old_key != new_key
    assert old_key[1] == new_key[1]  # endpoint unchanged
    assert talk_apiserver.warm().available
    assert calls[0] == "first"
    assert calls[-1] == "second"
