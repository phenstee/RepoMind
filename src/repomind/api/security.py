"""Request guards for a trusted-local API that deliberately has no authentication.

Without credentials, the only things standing between a hostile web page and the
local API are the browser's own rules. These guards make those rules bite:

* the Host check defeats DNS rebinding, where an attacker's domain re-resolves to
  127.0.0.1 and the browser then treats the API as same-origin;
* the client header forces a CORS preflight for every state-changing request, so a
  cross-site form or ``text/plain`` "simple request" cannot queue work or cancel jobs.
"""

from collections.abc import Iterable

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

CLIENT_HEADER = "X-RepoMind-Client"
STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def request_hostname(host: str) -> str:
    """Lower-case host name without its port; bracketed IPv6 keeps its brackets.

    Starlette's own TrustedHostMiddleware splits on the first ``:``, which turns
    ``[::1]:8000`` into ``[`` and so can never allow an IPv6 loopback client.
    """

    if host.startswith("["):
        end = host.find("]")
        return host[: end + 1].lower() if end != -1 else ""
    return host.split(":", 1)[0].lower()


class TrustedHostMiddleware:
    def __init__(self, app: ASGIApp, allowed_hosts: Iterable[str]) -> None:
        self.app = app
        self.allowed_hosts = frozenset(host.lower() for host in allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            host = request_hostname(Headers(scope=scope).get("host", ""))
            if host not in self.allowed_hosts:
                response = _error(400, "invalid_host", "Request host is not allowed.")
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class ClientHeaderMiddleware:
    """Reject state-changing requests that lack the non-simple client header.

    Installed inside CORS so a trusted frontend can still read the 403 envelope.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope["method"] in STATE_CHANGING_METHODS
            and not Headers(scope=scope).get(CLIENT_HEADER, "").strip()
        ):
            response = _error(
                403,
                "missing_client_header",
                f"State-changing requests require the {CLIENT_HEADER} header.",
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
