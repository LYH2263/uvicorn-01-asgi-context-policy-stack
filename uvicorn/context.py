"""Connection/request scoped ASGI context execution policy.

This module holds a stack of :class:`PolicyFrame` objects in a
:class:`contextvars.ContextVar`. The stack is independent from
``Config.reset_contextvars``, from the ASGI ``scope["state"]`` dictionary and
from ``LifespanOn.state``: it never mutates the process-wide configuration and
it never stores anything on a protocol or connection object, so concurrent
connections and concurrent HTTP/2 streams each observe their own view.

A frame may override any of three dimensions:

* ``task_context`` selects the ``contextvars.Context`` the ASGI task is
  created with:

  - ``"inherit"``: the task copies the context current at creation time.
  - ``"fresh"``: the task starts with an empty ``contextvars.Context`` (the
    same semantics ``--reset-contextvars`` gives HTTP requests).
  - ``"copy"``: the task runs in an explicit copy of the current context.

* ``logging_binding`` selects where the HTTP access log reads context from:

  - ``"asgi"``: the access log is emitted inside the ASGI task, so it sees
    ``ContextVar`` values the application wrote.
  - ``"protocol"``: the access log runs in a snapshot of the context visible
    to the protocol layer when the request cycle was created, so values the
    application writes later in the ASGI task stay invisible.

* ``wsgi_thread`` selects the context of the worker thread used by the WSGI
  middleware:

  - ``"empty"``: the worker thread starts from an empty ``contextvars.Context``.
  - ``"copy"``: the context current before ``run_in_executor`` is copied into
    the worker thread.

Read :func:`effective_policy` for the view obtained by merging the current
stack top with the ``Config.reset_contextvars`` baseline. Frames nest: when an
inner frame is popped, the frame pushed by the outer scope becomes visible
again. ``pop_policy`` (or the :func:`policy_stack` context manager) restores
the previous value via a ``contextvars`` token, which also unwinds cleanly on
exception paths.
"""

from __future__ import annotations

import contextvars
import sys
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop, Task

    from uvicorn.config import Config

TaskContext = Literal["inherit", "fresh", "copy"]
LoggingBinding = Literal["asgi", "protocol"]
WSGIThreadContext = Literal["copy", "empty"]


class PolicyFrame(NamedTuple):
    """One entry of the ASGI context policy stack.

    Dimensions left as ``None`` defer to an outer frame, then to the
    ``Config.reset_contextvars`` baseline.
    """

    task_context: TaskContext | None = None
    logging_binding: LoggingBinding | None = None
    wsgi_thread: WSGIThreadContext | None = None


class EffectivePolicy(NamedTuple):
    """The merged view a protocol exit acts on."""

    task_context: TaskContext
    logging_binding: LoggingBinding
    wsgi_thread: WSGIThreadContext


_policy_stack: contextvars.ContextVar[tuple[PolicyFrame, ...]] = contextvars.ContextVar(
    "uvicorn_asgi_context_policy_stack",
    default=(),
)


def push_policy(
    *,
    task_context: TaskContext | None = None,
    logging_binding: LoggingBinding | None = None,
    wsgi_thread: WSGIThreadContext | None = None,
) -> contextvars.Token[tuple[PolicyFrame, ...]]:
    """Push a frame onto the current context's policy stack.

    Pair with :func:`pop_policy`, or use :func:`policy_stack` for automatic
    unwinding on exception paths. At least one dimension must be set.
    """
    if task_context is None and logging_binding is None and wsgi_thread is None:
        raise ValueError("A policy frame must override at least one dimension.")
    frame = PolicyFrame(task_context=task_context, logging_binding=logging_binding, wsgi_thread=wsgi_thread)
    return _policy_stack.set(_policy_stack.get() + (frame,))


def pop_policy(token: contextvars.Token[tuple[PolicyFrame, ...]]) -> None:
    """Pop the frame pushed with *token*, restoring the previous stack.

    Like every ``contextvars`` token reset, this only affects the context in
    which :func:`push_policy` was called; concurrent connections hold their
    own copies and are never rewound.
    """
    _policy_stack.reset(token)


@contextmanager
def policy_stack(
    *,
    task_context: TaskContext | None = None,
    logging_binding: LoggingBinding | None = None,
    wsgi_thread: WSGIThreadContext | None = None,
) -> Iterator[None]:
    """Context manager that pushes a frame and always pops it again."""
    token = push_policy(task_context=task_context, logging_binding=logging_binding, wsgi_thread=wsgi_thread)
    try:
        yield
    finally:
        pop_policy(token)


def effective_policy(config: Config | None = None) -> EffectivePolicy:
    """Merge the current stack with the process configuration baseline.

    Baseline:

    * ``task_context`` is ``"fresh"`` when ``Config.reset_contextvars`` is
      enabled, ``"inherit"`` otherwise;
    * ``logging_binding`` is ``"asgi"`` (access logging happens on the ASGI
      task's send path);
    * ``wsgi_thread`` is ``"empty"`` (the executor worker does not inherit
      the asyncio task context).

    The innermost frame that sets a dimension wins for that dimension alone.
    """
    stack = _policy_stack.get()
    task_context: TaskContext = "fresh" if config is not None and config.reset_contextvars else "inherit"
    logging_binding: LoggingBinding = "asgi"
    wsgi_thread: WSGIThreadContext = "empty"
    for frame in reversed(stack):
        if frame.task_context is not None:
            task_context = frame.task_context
            break
    for frame in reversed(stack):
        if frame.logging_binding is not None:
            logging_binding = frame.logging_binding
            break
    for frame in reversed(stack):
        if frame.wsgi_thread is not None:
            wsgi_thread = frame.wsgi_thread
            break
    return EffectivePolicy(
        task_context=task_context,
        logging_binding=logging_binding,
        wsgi_thread=wsgi_thread,
    )


def current_logging_binding() -> LoggingBinding | None:
    """``logging_binding`` dictated by frames visible in this context.

    Returns ``None`` when no frame sets the dimension, so callers can fall
    back to the binding captured at request cycle creation instead of
    inventing a second default.
    """
    for frame in reversed(_policy_stack.get()):
        if frame.logging_binding is not None:
            return frame.logging_binding
    return None


def create_asgi_task(
    loop: AbstractEventLoop,
    coroutine: Coroutine[Any, Any, Any],
    task_context: TaskContext,
) -> Task[Any]:
    """Create the ASGI application task under *task_context*.

    ``"inherit"`` delegates to the plain ``loop.create_task`` call. Both
    ``"fresh"`` and ``"copy"`` cover the two ``create_task`` spellings:
    Python 3.11+ passes the context explicitly, older versions run
    ``create_task`` inside the chosen context.
    """
    if task_context == "inherit":
        return loop.create_task(coroutine)

    context = contextvars.copy_context() if task_context == "copy" else contextvars.Context()
    if sys.version_info >= (3, 11):  # pragma: py-lt-311
        return loop.create_task(coroutine, context=context)
    # pragma: py-gte-311
    return context.run(loop.create_task, coroutine)
