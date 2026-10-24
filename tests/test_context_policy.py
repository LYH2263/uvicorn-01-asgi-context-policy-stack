"""Acceptance tests for the per-request/connection ASGI context policy stack.

The tests are split by exit, as required:

1. stack mechanics and the merged ``Config.reset_contextvars`` view;
2. HTTP exits: h11, httptools, zttp and HTTP/2 streams (3.11+ spelling plus
   the pre-3.11 ``Context.run`` spelling);
3. WebSocket exits: wsproto, websockets-sansio and the legacy websockets
   implementation (when importable);
4. HTTP -> WebSocket upgrade reading the stack top at handshake time;
5. access log ``logging_binding`` fork;
6. WSGI ``wsgi_thread`` executor fork;
7. lifespan sibling task non-subscription;
8. real TCP server parity with the in-memory ASGI harness.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import sys
import warnings
from typing import Any

import httpx2
import pytest
import zttp

from tests.utils import run_server
from uvicorn import EffectivePolicy, PolicyFrame, effective_policy, policy_stack, pop_policy, push_policy
from uvicorn.config import Config
from uvicorn.context import create_asgi_task
from uvicorn.lifespan.off import LifespanOff
from uvicorn.lifespan.on import LifespanOn
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.protocols.http.zttp_h2_impl import ZttpH2Protocol
from uvicorn.protocols.websockets.websockets_sansio_impl import WebSocketsSansIOProtocol
from uvicorn.protocols.websockets.wsproto_impl import WSProtocol as WsprotoProtocol
from uvicorn.server import ServerState

try:
    from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol
except ModuleNotFoundError:  # pragma: no cover
    HttpToolsProtocol = None  # type: ignore[assignment,misc]

try:
    from uvicorn.protocols.http.zttp_impl import ZttpProtocol
except ModuleNotFoundError:  # pragma: no cover
    ZttpProtocol = None  # type: ignore[assignment,misc]

try:  # legacy implementation is untested upstream; the installed websockets
    # release is incompatible with it, so only the import/source is checked.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from uvicorn.protocols.websockets.websockets_impl import WebSocketProtocol as LegacyWSProtocol
except ModuleNotFoundError:  # pragma: no cover
    LegacyWSProtocol = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.anyio

REQUEST_VAR: contextvars.ContextVar[str] = contextvars.ContextVar("request_var", default="MISSING")
WORKER_VAR: contextvars.ContextVar[str] = contextvars.ContextVar("worker_var", default="MISSING")

SIMPLE_GET = b"\r\n".join([b"GET / HTTP/1.1", b"Host: example.org", b"", b""])

UPGRADE_WITH_KEY = b"\r\n".join(
    [
        b"GET / HTTP/1.1",
        b"Host: example.org",
        b"Connection: upgrade",
        b"Upgrade: websocket",
        b"Sec-WebSocket-Version: 13",
        b"Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==",
        b"",
        b"",
    ]
)

HTTP1_PROTOCOLS = [
    pytest.param(H11Protocol, id="h11"),
    pytest.param(
        HttpToolsProtocol, id="httptools", marks=pytest.mark.skipif(HttpToolsProtocol is None, reason="httptools")
    ),
    pytest.param(ZttpProtocol, id="zttp", marks=pytest.mark.skipif(ZttpProtocol is None, reason="zttp")),
]

WS_PROTOCOLS = [
    pytest.param(WsprotoProtocol, id="wsproto"),
    pytest.param(WebSocketsSansIOProtocol, id="websockets-sansio"),
]


class MemoryTransport:
    def __init__(self, sslcontext: bool = False):
        self.sockname = ("127.0.0.1", 8000)
        self.peername = ("127.0.0.1", 8001)
        self.sslcontext = sslcontext
        self.buffer = b""
        self.closed = False
        self.read_paused = False
        self.protocol: asyncio.Protocol | None = None

    def get_extra_info(self, key: str) -> Any:
        return {"sockname": self.sockname, "peername": self.peername, "sslcontext": self.sslcontext}.get(key)

    def write(self, data: bytes) -> None:
        self.buffer += data

    def writelines(self, data: Any) -> None:
        for chunk in data:
            self.write(chunk)

    def close(self) -> None:
        self.closed = True

    abort = close

    def can_write_eof(self) -> bool:
        return False

    def is_closing(self) -> bool:
        return self.closed

    def pause_reading(self) -> None:
        self.read_paused = True

    def resume_reading(self) -> None:
        self.read_paused = False

    def set_protocol(self, protocol: asyncio.Protocol) -> None:
        self.protocol = protocol

    def get_protocol(self) -> asyncio.Protocol | None:
        return self.protocol


class RecordingLoop:
    """Real loop wrapper recording contexts passed to ``create_task``."""

    def __init__(self, inner: asyncio.AbstractEventLoop):
        self.inner = inner
        self.contexts: list[contextvars.Context | None] = []

    def create_task(self, coroutine: Any, context: contextvars.Context | None = None, **kwargs: Any) -> Any:
        self.contexts.append(context)
        if context is None:
            return self.inner.create_task(coroutine, **kwargs)
        return self.inner.create_task(coroutine, context=context, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


async def settle(state: ServerState) -> None:
    while state.tasks:
        # done_callbacks remove the tasks from the shared set as they finish.
        await asyncio.gather(*list(state.tasks), return_exceptions=True)


def make_http1(
    protocol_cls: Any,
    app: Any,
    loop: Any,
    state: ServerState | None = None,
    **config_kwargs: Any,
) -> tuple[Any, MemoryTransport, Config, ServerState]:
    config = Config(app=app, http=protocol_cls, lifespan="off", **config_kwargs)
    if state is None:
        state = ServerState()
    protocol = protocol_cls(config=config, server_state=state, app_state=LifespanOff(config).state, _loop=loop)
    transport = MemoryTransport()
    protocol.connection_made(transport)
    return protocol, transport, config, state


def make_h2(app: Any, loop: Any, **config_kwargs: Any) -> tuple[Any, MemoryTransport, Config, ServerState]:
    config = Config(app=app, http="zttp", http2=True, lifespan="off", **config_kwargs)
    state = ServerState()
    protocol = ZttpH2Protocol(config=config, server_state=state, app_state={}, _loop=loop)
    transport = MemoryTransport(sslcontext=True)
    protocol.connection_made(transport)
    return protocol, transport, config, state


def http_recorder(results: dict[str, str], name: str, gate: asyncio.Event | None = None) -> Any:
    async def app(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        results[name] = REQUEST_VAR.get()
        if gate is not None:
            await gate.wait()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": results[name].encode()})

    return app


class H2Client:
    def __init__(self) -> None:
        self.conn = zttp.Connection(zttp.CLIENT, protocol=zttp.HTTP2)

    def request(self, target: bytes = b"/") -> None:
        self.conn.send_request(b"GET", target, b"2", [(b"host", b"example.org")]).end_message()

    def send(self, protocol: Any, transport: MemoryTransport) -> None:
        protocol.data_received(self.conn.data_to_send())

    def absorb(self, transport: MemoryTransport) -> None:
        self.conn.receive_data(transport.buffer)
        transport.buffer = b""
        while self.conn.next_event() is not zttp.NEED_DATA:
            pass


# ---------------------------------------------------------------------------
# 1. Stack mechanics and merged effective view
# ---------------------------------------------------------------------------


def test_baseline_matches_config_only() -> None:
    assert effective_policy(None) == EffectivePolicy("inherit", "asgi", "empty")
    assert effective_policy(Config(app=None, reset_contextvars=False)) == EffectivePolicy("inherit", "asgi", "empty")
    assert effective_policy(Config(app=None, reset_contextvars=True)) == EffectivePolicy("fresh", "asgi", "empty")


def test_empty_frame_rejected() -> None:
    with pytest.raises(ValueError):
        push_policy()


def test_frame_overrides_only_its_dimensions() -> None:
    config = Config(app=None, reset_contextvars=True)
    with policy_stack(logging_binding="protocol"):
        assert effective_policy(config) == EffectivePolicy("fresh", "protocol", "empty")
    with policy_stack(wsgi_thread="copy"):
        assert effective_policy(config) == EffectivePolicy("fresh", "asgi", "copy")
    assert effective_policy(config) == EffectivePolicy("fresh", "asgi", "empty")


def test_nested_pop_returns_to_outer_frame_never_global_singleton() -> None:
    config = Config(app=None, reset_contextvars=False)
    outer = push_policy(task_context="fresh")
    assert effective_policy(config).task_context == "fresh"
    inner = push_policy(task_context="copy", logging_binding="protocol")
    assert effective_policy(config) == EffectivePolicy("copy", "protocol", "empty")
    pop_policy(inner)
    # The outer user frame is still there; this is neither the process global
    # nor the bare Config default.
    assert effective_policy(config) == EffectivePolicy("fresh", "asgi", "empty")
    pop_policy(outer)
    assert effective_policy(config) == EffectivePolicy("inherit", "asgi", "empty")


def test_context_manager_unwinds_on_exception() -> None:
    config = Config(app=None, reset_contextvars=True)
    with pytest.raises(RuntimeError):
        with policy_stack(task_context="inherit"):
            assert effective_policy(config).task_context == "inherit"
            raise RuntimeError("boom")
    assert effective_policy(config).task_context == "fresh"


def test_stack_and_config_recover_independently() -> None:
    config = Config(app=None, reset_contextvars=True)
    token = push_policy(task_context="inherit")
    assert effective_policy(config).task_context == "inherit"
    # The frame can be popped without touching Config, and Config can change
    # while a frame remains pushed.
    config.reset_contextvars = False
    assert effective_policy(config).task_context == "inherit"
    pop_policy(token)
    assert effective_policy(config).task_context == "inherit"
    config.reset_contextvars = True
    inner = push_policy(task_context="copy")
    assert effective_policy(config).task_context == "copy"
    pop_policy(inner)
    assert effective_policy(config).task_context == "fresh"


def test_policy_frame_is_a_public_namedtuple() -> None:
    assert PolicyFrame(task_context="copy").task_context == "copy"
    assert PolicyFrame().task_context is None


def test_create_asgi_task_spellings_python311_plus() -> None:
    seen: list[contextvars.Context | None] = []

    class FakeLoop:
        def __init__(self, inner: asyncio.AbstractEventLoop):
            self.inner = inner

        def create_task(self, coro: Any, context: contextvars.Context | None = None, **kwargs: Any) -> Any:
            seen.append(context)
            if context is None:
                return self.inner.create_task(coro, **kwargs)
            return self.inner.create_task(coro, context=context, **kwargs)

    async def noop() -> None:
        return None

    async def scenario() -> None:
        REQUEST_VAR.set("OUTER")
        loop = FakeLoop(asyncio.get_running_loop())

        await create_asgi_task(loop, noop(), "inherit")
        await create_asgi_task(loop, noop(), "fresh")
        await create_asgi_task(loop, noop(), "copy")

    asyncio.run(scenario())

    assert seen[0] is None
    assert seen[1] is not None and len(seen[1]) == 0
    assert seen[2] is not None and len(seen[2]) >= 1
    assert seen[2].run(REQUEST_VAR.get) == "OUTER"  # type: ignore[union-attr]


def test_create_asgi_task_spellings_pre311(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class FakeLoop:
        def __init__(self, inner: asyncio.AbstractEventLoop):
            self.inner = inner

        def create_task(self, coro: Any) -> Any:
            # The context we run in here is exactly what Context.run selected.
            seen.append(REQUEST_VAR.get())
            return self.inner.create_task(coro)

    async def noop() -> None:
        return None

    async def scenario() -> None:
        REQUEST_VAR.set("OUTER")
        loop = FakeLoop(asyncio.get_running_loop())

        monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
        await create_asgi_task(loop, noop(), "inherit")
        await create_asgi_task(loop, noop(), "fresh")
        await create_asgi_task(loop, noop(), "copy")

    asyncio.run(scenario())

    assert seen == ["OUTER", "MISSING", "OUTER"]


def test_copy_is_a_point_in_time_snapshot() -> None:
    REQUEST_VAR.set("v1")
    snapshot = contextvars.copy_context()
    REQUEST_VAR.set("v2")
    assert snapshot.run(REQUEST_VAR.get) == "v1"


# ---------------------------------------------------------------------------
# 2a. HTTP/1.1 exits read the merged task_context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protocol_cls", HTTP1_PROTOCOLS)
@pytest.mark.parametrize(
    "mode,expected,kind",
    [
        ("inherit", "OUTER", "none"),
        ("fresh", "MISSING", "empty"),
        ("copy", "OUTER", "copy"),
    ],
)
async def test_http1_exit_task_context(protocol_cls: Any, mode: str, expected: str, kind: str) -> None:
    REQUEST_VAR.set("OUTER")
    results: dict[str, str] = {}
    loop = RecordingLoop(asyncio.get_running_loop())
    protocol, _transport, _config, state = make_http1(protocol_cls, http_recorder(results, "r"), loop)

    with policy_stack(task_context=mode):  # type: ignore[arg-type]
        protocol.data_received(SIMPLE_GET)
    await settle(state)

    assert results["r"] == expected
    captured = loop.contexts[-1]
    if kind == "none":
        assert captured is None
    else:
        assert captured is not None
        assert captured.run(REQUEST_VAR.get) == expected
        assert (len(captured) == 0) is (kind == "empty")


@pytest.mark.parametrize("protocol_cls", HTTP1_PROTOCOLS)
async def test_http1_exit_empty_stack_matches_config_baseline(protocol_cls: Any) -> None:
    # reset_contextvars=False: plain create_task, inheriting creation context.
    REQUEST_VAR.set("OUTER")
    results: dict[str, str] = {}
    loop = RecordingLoop(asyncio.get_running_loop())
    protocol, _t, _c, state = make_http1(protocol_cls, http_recorder(results, "r"), loop, reset_contextvars=False)
    protocol.data_received(SIMPLE_GET)
    await settle(state)
    assert results["r"] == "OUTER"
    assert loop.contexts[-1] is None

    # reset_contextvars=True: empty Context baseline survives.
    REQUEST_VAR.set("OUTER")
    results = {}
    loop = RecordingLoop(asyncio.get_running_loop())
    protocol, _t, _c, state = make_http1(protocol_cls, http_recorder(results, "r"), loop, reset_contextvars=True)
    protocol.data_received(SIMPLE_GET)
    await settle(state)
    assert results["r"] == "MISSING"
    assert len(loop.contexts[-1]) == 0  # type: ignore[arg-type]


@pytest.mark.parametrize("protocol_cls", HTTP1_PROTOCOLS)
async def test_http1_task_context_writes_never_leak_back(protocol_cls: Any) -> None:
    REQUEST_VAR.set("OUTER")
    results: dict[str, str] = {}
    loop = asyncio.get_running_loop()
    protocol, _t, _c, state = make_http1(protocol_cls, http_recorder(results, "r"), loop)
    with policy_stack(task_context="copy"):
        protocol.data_received(SIMPLE_GET)
    await settle(state)
    assert REQUEST_VAR.get() == "OUTER"


# ---------------------------------------------------------------------------
# 2b. Concurrent HTTP/1.1 connections do not trample each other
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protocol_cls", HTTP1_PROTOCOLS)
async def test_concurrent_connections_isolated_when_other_pushes_and_pops(
    protocol_cls: Any,
) -> None:
    REQUEST_VAR.set("OUTER")
    gate = asyncio.Event()
    results_a: dict[str, str] = {}
    results_b: dict[str, str] = {}

    loop = asyncio.get_running_loop()
    protocol_a, transport_a, _c, state_a = make_http1(protocol_cls, http_recorder(results_a, "a", gate), loop)
    protocol_b, transport_b, _c, state_b = make_http1(protocol_cls, http_recorder(results_b, "b"), loop)

    post_headers = b"POST / HTTP/1.1\r\nHost: example.org\r\nContent-Length: 5\r\n\r\n"
    with policy_stack(task_context="fresh"):
        protocol_a.data_received(post_headers)
    # Let the ASGI task for A reach its gate.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # B pushes, serves and pops while A is still mid-request.
    with policy_stack(task_context="copy"):
        protocol_b.data_received(SIMPLE_GET)
        await settle(state_b)

    assert results_b == {"b": "OUTER"}
    # A recorded its own view before reaching the gate, and B's push/pop must
    # not have rewritten it while A was still streaming the request body.
    assert results_a == {"a": "MISSING"}

    gate.set()
    await settle(state_a)
    assert results_a == {"a": "MISSING"}  # A never adopted B's "copy"
    assert b"HTTP/1.1 200" in transport_a.buffer
    assert b"MISSING" in transport_a.buffer
    assert REQUEST_VAR.get() == "OUTER"


# ---------------------------------------------------------------------------
# 2c. HTTP/2 per-stream exits on one shared protocol object
# ---------------------------------------------------------------------------


async def _h2_path_recorder(results: dict[str, str], gate: asyncio.Event | None = None) -> Any:
    async def app(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        path = scope["path"].lstrip("/")
        results[path] = REQUEST_VAR.get()
        if gate is not None and path == "a":
            await gate.wait()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": results[path].encode()})

    return app


async def _h2_handshake(protocol: Any, transport: MemoryTransport) -> H2Client:
    client = H2Client()
    protocol.data_received(client.conn.data_to_send())
    client.absorb(transport)
    return client


@pytest.mark.parametrize(
    "mode,expected,kind",
    [
        ("inherit", "OUTER", "none"),
        ("fresh", "MISSING", "empty"),
        ("copy", "OUTER", "copy"),
    ],
)
async def test_h2_stream_exit_task_context(mode: str, expected: str, kind: str) -> None:
    REQUEST_VAR.set("OUTER")
    results: dict[str, str] = {}
    loop = RecordingLoop(asyncio.get_running_loop())
    protocol, transport, _c, state = make_h2(await _h2_path_recorder(results), loop)
    client = await _h2_handshake(protocol, transport)

    client.request(b"/")
    with policy_stack(task_context=mode):  # type: ignore[arg-type]
        client.send(protocol, transport)
    await settle(state)

    assert results[""] == expected
    captured = loop.contexts[-1]
    if kind == "none":
        assert captured is None
    else:
        assert captured is not None
        assert captured.run(REQUEST_VAR.get) == expected


async def test_h2_concurrent_streams_see_only_their_own_view() -> None:
    REQUEST_VAR.set("OUTER")
    results: dict[str, str] = {}
    loop = RecordingLoop(asyncio.get_running_loop())
    protocol, transport, _c, state = make_h2(await _h2_path_recorder(results), loop)
    client = await _h2_handshake(protocol, transport)

    client.request(b"/a")
    data_a = client.conn.data_to_send()
    client.request(b"/b")
    data_b = client.conn.data_to_send()

    with policy_stack(task_context="fresh"):
        protocol.data_received(data_a)
    with policy_stack(task_context="copy"):
        protocol.data_received(data_b)
    await settle(state)

    assert results == {"a": "MISSING", "b": "OUTER"}
    fresh_context, copy_context = loop.contexts
    assert fresh_context is not None and len(fresh_context) == 0
    assert copy_context is not None and copy_context is not fresh_context
    assert copy_context.run(REQUEST_VAR.get) == "OUTER"


async def test_h2_baseline_reset_contextvars_per_stream() -> None:
    REQUEST_VAR.set("OUTER")
    results: dict[str, str] = {}
    loop = RecordingLoop(asyncio.get_running_loop())
    protocol, transport, _c, state = make_h2(await _h2_path_recorder(results), loop, reset_contextvars=True)
    client = await _h2_handshake(protocol, transport)
    client.request(b"/a")
    client.send(protocol, transport)
    await settle(state)
    assert results == {"a": "MISSING"}
    assert len(loop.contexts[-1]) == 0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. WebSocket exits read the stack (and the Config baseline)
# ---------------------------------------------------------------------------


def ws_recorder(results: dict[str, str]) -> Any:
    async def app(scope: Any, receive: Any, send: Any) -> None:
        assert scope["type"] == "websocket"
        results["ctx"] = REQUEST_VAR.get()
        await send({"type": "websocket.accept"})

    return app


@pytest.mark.parametrize("ws_cls", WS_PROTOCOLS)
@pytest.mark.parametrize(
    "frame,reset,expected",
    [
        (None, False, "OUTER"),  # empty stack + reset False: pre-change inherit behaviour
        ("fresh", False, "MISSING"),
        ("copy", False, "OUTER"),
        (None, True, "MISSING"),  # WebSocket now obeys the Config baseline too
    ],
)
async def test_websocket_exits(
    unused_tcp_port: int, ws_cls: Any, frame: str | None, reset: bool, expected: str
) -> None:
    REQUEST_VAR.set("OUTER")
    results: dict[str, str] = {}
    config = Config(
        app=ws_recorder(results),
        ws=ws_cls,
        http=H11Protocol,
        lifespan="off",
        port=unused_tcp_port,
        reset_contextvars=reset,
    )

    from websockets.asyncio.client import connect

    if frame is None:
        async with run_server(config):
            async with connect(f"ws://127.0.0.1:{unused_tcp_port}"):
                pass
    else:
        with policy_stack(task_context=frame):  # type: ignore[arg-type]
            async with run_server(config):
                async with connect(f"ws://127.0.0.1:{unused_tcp_port}"):
                    pass

    assert results["ctx"] == expected


def test_legacy_websocket_exit_reads_the_merged_view() -> None:
    # The legacy implementation cannot run against the installed websockets
    # release (upstream marks it untested), so guard its wiring at source
    # level: it must read the merged task_context and create through the
    # shared primitive instead of ignoring the switch like it used to.
    if LegacyWSProtocol is None:
        pytest.skip("legacy websockets implementation not importable")
    import inspect

    source = inspect.getsource(LegacyWSProtocol.process_request)
    assert "effective_policy(self.config).task_context" in source
    assert "create_asgi_task(self.loop, self.run_asgi(), task_context)" in source


# ---------------------------------------------------------------------------
# 4. HTTP -> WebSocket upgrade follows the stack top at handshake completion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "http_cls",
    [
        pytest.param(H11Protocol, id="h11"),
        pytest.param(ZttpProtocol, id="zttp", marks=pytest.mark.skipif(ZttpProtocol is None, reason="zttp")),
    ],
)
@pytest.mark.parametrize(
    "ws_cls", [pytest.param(WsprotoProtocol, id="wsproto"), pytest.param(WebSocketsSansIOProtocol, id="sansio")]
)
@pytest.mark.parametrize("mode,expected", [("fresh", "MISSING"), ("copy", "OUTER")])
async def test_upgrade_uses_stack_top_when_handshake_completes(
    http_cls: Any, ws_cls: Any, mode: str, expected: str
) -> None:
    REQUEST_VAR.set("OUTER")
    results: dict[str, str] = {}
    loop = asyncio.get_running_loop()
    config = Config(app=ws_recorder(results), http=http_cls, ws=ws_cls, lifespan="off")
    state = ServerState()
    protocol = http_cls(config=config, server_state=state, app_state={}, _loop=loop)
    transport = MemoryTransport()
    protocol.connection_made(transport)

    # The protocol object exists before any frame is pushed; the read happens
    # while the upgrade handshake is replayed, not at construction time.
    with policy_stack(task_context=mode):  # type: ignore[arg-type]
        protocol.data_received(UPGRADE_WITH_KEY)
    await settle(state)

    assert results["ctx"] == expected
    assert b"101" in transport.buffer


@pytest.mark.parametrize(
    "ws_cls", [pytest.param(WsprotoProtocol, id="wsproto"), pytest.param(WebSocketsSansIOProtocol, id="sansio")]
)
async def test_two_upgrades_do_not_share_a_frozen_snapshot(ws_cls: Any) -> None:
    REQUEST_VAR.set("OUTER")
    results_a: dict[str, str] = {}
    results_b: dict[str, str] = {}
    loop = asyncio.get_running_loop()
    config_a = Config(app=ws_recorder(results_a), http=H11Protocol, ws=ws_cls, lifespan="off")
    config_b = Config(app=ws_recorder(results_b), http=H11Protocol, ws=ws_cls, lifespan="off")
    state_a, state_b = ServerState(), ServerState()
    proto_a = H11Protocol(config=config_a, server_state=state_a, app_state={}, _loop=loop)
    proto_b = H11Protocol(config=config_b, server_state=state_b, app_state={}, _loop=loop)
    transport_a, transport_b = MemoryTransport(), MemoryTransport()
    proto_a.connection_made(transport_a)
    proto_b.connection_made(transport_b)

    with policy_stack(task_context="fresh"):
        proto_a.data_received(UPGRADE_WITH_KEY)
    # Fresh is popped before B upgrades with copy; A must keep "fresh".
    with policy_stack(task_context="copy"):
        proto_b.data_received(UPGRADE_WITH_KEY)
    await settle(state_a)
    await settle(state_b)

    assert results_a == {"ctx": "MISSING"}
    assert results_b == {"ctx": "OUTER"}


# ---------------------------------------------------------------------------
# 5. Access log logging_binding fork
# ---------------------------------------------------------------------------


class VarCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.values.append(REQUEST_VAR.get())


class AccessCaptureFixture:
    """Attaches after Config.load(), which itself reconfigures loggers."""

    def __init__(self) -> None:
        self.handler = VarCapture()
        self._logger: logging.Logger | None = None
        self._previous_level = logging.NOTSET

    def start(self) -> None:
        logger = logging.getLogger("uvicorn.access")
        self._logger = logger
        self._previous_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(self.handler)

    def stop(self) -> None:
        if self._logger is not None:
            self._logger.removeHandler(self.handler)
            self._logger.setLevel(self._previous_level)
            self._logger = None

    @property
    def values(self) -> list[str]:
        return self.handler.values


def app_sets_var_before_send(set_after_logger_bound: bool = False) -> Any:
    async def app(scope: Any, receive: Any, send: Any) -> None:
        REQUEST_VAR.set("APP")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    return app


@pytest.fixture
def access_capture() -> Any:
    capture = AccessCaptureFixture()
    try:
        yield capture
    finally:
        capture.stop()


@pytest.mark.parametrize("protocol_cls", HTTP1_PROTOCOLS)
async def test_access_log_asgi_binding_sees_app_context(protocol_cls: Any, access_capture: VarCapture) -> None:
    REQUEST_VAR.set("OUTER")
    loop = asyncio.get_running_loop()
    protocol, _t, _c, state = make_http1(protocol_cls, app_sets_var_before_send(), loop)
    access_capture.start()
    protocol.data_received(SIMPLE_GET)
    await settle(state)
    assert access_capture.values == ["APP"]


@pytest.mark.parametrize("protocol_cls", HTTP1_PROTOCOLS)
async def test_access_log_protocol_binding_hides_later_app_writes(
    protocol_cls: Any, access_capture: VarCapture
) -> None:
    REQUEST_VAR.set("OUTER")
    loop = asyncio.get_running_loop()
    protocol, _t, _c, state = make_http1(protocol_cls, app_sets_var_before_send(), loop)
    access_capture.start()
    with policy_stack(logging_binding="protocol"):
        protocol.data_received(SIMPLE_GET)
        await settle(state)
    assert access_capture.values == ["OUTER"]


@pytest.mark.parametrize("protocol_cls", HTTP1_PROTOCOLS)
async def test_access_log_protocol_binding_works_with_fresh_task(protocol_cls: Any, access_capture: VarCapture) -> None:
    REQUEST_VAR.set("OUTER")
    seen: dict[str, str] = {}
    loop = asyncio.get_running_loop()

    async def app(scope: Any, receive: Any, send: Any) -> None:
        seen["app"] = REQUEST_VAR.get()
        REQUEST_VAR.set("APP")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    protocol, _t, _c, state = make_http1(protocol_cls, app, loop)
    access_capture.start()
    with policy_stack(task_context="fresh", logging_binding="protocol"):
        protocol.data_received(SIMPLE_GET)
        await settle(state)
    assert seen == {"app": "MISSING"}
    assert access_capture.values == ["OUTER"]


def logging_override_middleware(inner: Any, binding: str) -> Any:
    async def app(scope: Any, receive: Any, send: Any) -> None:
        with policy_stack(logging_binding=binding):  # type: ignore[arg-type]
            await inner(scope, receive, send)

    return app


@pytest.mark.parametrize("protocol_cls", HTTP1_PROTOCOLS)
async def test_middleware_can_switch_binding_inside_task(protocol_cls: Any, access_capture: VarCapture) -> None:
    REQUEST_VAR.set("OUTER")
    loop = asyncio.get_running_loop()

    # Default asgi at creation; middleware narrows to protocol inside the task.
    protocol, _t, _c, state = make_http1(
        protocol_cls, logging_override_middleware(app_sets_var_before_send(), "protocol"), loop
    )
    access_capture.start()
    protocol.data_received(SIMPLE_GET)
    await settle(state)
    assert access_capture.values == ["OUTER"]

    # Outer protocol frame; middleware widens back to asgi inside the task.
    # The second Config.load() reconfigures logging, so re-arm the capture
    # after constructing that protocol.
    REQUEST_VAR.set("OUTER")
    access_capture.stop()
    protocol, _t, _c, state = make_http1(
        protocol_cls, logging_override_middleware(app_sets_var_before_send(), "asgi"), loop
    )
    access_capture.start()
    with policy_stack(logging_binding="protocol"):
        protocol.data_received(SIMPLE_GET)
        await settle(state)
    assert access_capture.values == ["OUTER", "APP"]


async def test_protocol_layer_503_warning_stays_emitted_under_frame(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate = asyncio.Event()
    loop = asyncio.get_running_loop()
    shared = ServerState()
    protocol, _t, _c, state = make_http1(H11Protocol, http_recorder({}, "a", gate), loop, shared, limit_concurrency=1)
    protocol_2, _t2, _c2, _ = make_http1(H11Protocol, http_recorder({}, "b"), loop, shared, limit_concurrency=1)

    post_headers = b"POST /a HTTP/1.1\r\nHost: example.org\r\nContent-Length: 5\r\n\r\n"
    with policy_stack(task_context="fresh", logging_binding="protocol"):
        protocol.data_received(post_headers)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        protocol_2.data_received(SIMPLE_GET)
        await settle(shared)

    assert "Exceeded concurrency limit." in caplog.messages
    gate.set()
    await settle(shared)


# ---------------------------------------------------------------------------
# 6. WSGI wsgi_thread fork
# ---------------------------------------------------------------------------


def wsgi_var_app(environ: Any, start_response: Any) -> list[bytes]:
    value = REQUEST_VAR.get()
    start_response("200 OK", [("content-type", "text/plain"), ("content-length", str(len(value)))])
    return [value.encode()]


async def _wsgi_call() -> httpx2.Response:
    from uvicorn.middleware.wsgi import _WSGIMiddleware

    app = _WSGIMiddleware(wsgi_var_app)
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/")


async def test_wsgi_thread_default_empty() -> None:
    REQUEST_VAR.set("OUTER")
    response = await _wsgi_call()
    assert response.content == b"MISSING"


async def test_wsgi_thread_copy() -> None:
    REQUEST_VAR.set("OUTER")
    with policy_stack(wsgi_thread="copy"):
        response = await _wsgi_call()
    assert response.content == b"OUTER"


async def test_wsgi_thread_nested_override_restores() -> None:
    REQUEST_VAR.set("OUTER")
    with policy_stack(wsgi_thread="copy"):
        with policy_stack(wsgi_thread="empty"):
            inner = await _wsgi_call()
        outer_again = await _wsgi_call()
    assert inner.content == b"MISSING"
    assert outer_again.content == b"OUTER"


def wsgi_worker_leak_app(environ: Any, start_response: Any) -> list[bytes]:
    current = WORKER_VAR.get()
    WORKER_VAR.set("written-in-worker")
    start_response("200 OK", [("content-type", "text/plain"), ("content-length", str(len(current)))])
    return [current.encode()]


async def test_wsgi_worker_context_does_not_leak_between_calls() -> None:
    from uvicorn.middleware.wsgi import _WSGIMiddleware

    app = _WSGIMiddleware(wsgi_worker_leak_app)
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.get("/")
        second = await client.get("/")
    assert first.content == b"MISSING"
    assert second.content == b"MISSING"


# ---------------------------------------------------------------------------
# 7. Lifespan sibling task does not subscribe to the request stack
# ---------------------------------------------------------------------------


async def test_lifespan_ignores_request_level_frame() -> None:
    REQUEST_VAR.set("OUTER")
    seen: dict[str, str] = {}

    async def app(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "lifespan":
            return
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                seen["ctx"] = REQUEST_VAR.get()
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    config = Config(app=app, lifespan="on")
    with policy_stack(task_context="fresh"):
        lifespan = LifespanOn(config)
        try:
            await lifespan.startup()
        finally:
            assert seen == {"ctx": "OUTER"}
            await lifespan.shutdown()


# ---------------------------------------------------------------------------
# 8. Real TCP server behaves like the in-memory harness
# ---------------------------------------------------------------------------


async def test_real_server_http_copy(unused_tcp_port: int) -> None:
    REQUEST_VAR.set("OUTER")
    seen: dict[str, str] = {}

    async def app(scope: Any, receive: Any, send: Any) -> None:
        seen["ctx"] = REQUEST_VAR.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": seen["ctx"].encode()})

    config = Config(app=app, port=unused_tcp_port, lifespan="off")
    with policy_stack(task_context="copy"):
        async with run_server(config):
            async with httpx2.AsyncClient() as client:
                response = await client.get(f"http://127.0.0.1:{unused_tcp_port}")
    assert seen == {"ctx": "OUTER"}
    assert response.content == b"OUTER"
    assert REQUEST_VAR.get() == "OUTER"


async def test_real_server_http_fresh(unused_tcp_port: int) -> None:
    REQUEST_VAR.set("OUTER")
    seen: dict[str, str] = {}

    async def app(scope: Any, receive: Any, send: Any) -> None:
        seen["ctx"] = REQUEST_VAR.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"done"})

    config = Config(app=app, port=unused_tcp_port, lifespan="off")
    with policy_stack(task_context="fresh"):
        async with run_server(config):
            async with httpx2.AsyncClient() as client:
                await client.get(f"http://127.0.0.1:{unused_tcp_port}")
    assert seen == {"ctx": "MISSING"}
