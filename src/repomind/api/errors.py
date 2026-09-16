"""Deliberately small, public-safe HTTP error vocabulary."""

from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.exceptions import HTTPException

from repomind.db.repositories import RepositoryNotFoundError


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, trace_run_id: UUID | None = None):
        super().__init__(code)
        self.status = status
        self.code = code
        self.message = message
        self.trace_run_id = trace_run_id


def public_error(exc: Exception) -> APIError:
    if isinstance(exc, APIError):
        return exc
    if isinstance(exc, RepositoryNotFoundError):
        return APIError(404, "repository_not_found", "Repository not found.")
    if isinstance(exc, IntegrityError):
        return APIError(409, "repository_conflict", "Repository registration conflicts.")
    if isinstance(exc, SQLAlchemyError):
        return APIError(503, "storage_unavailable", "Storage is unavailable.")
    return APIError(500, "internal_error", "The operation could not be completed.")


def install_error_handlers(app: FastAPI) -> None:
    async def handle_error(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, RequestValidationError):
            error = APIError(422, "invalid_request", "Request validation failed.")
        elif isinstance(exc, HTTPException):
            error = APIError(exc.status_code, "http_error", "HTTP request could not be handled.")
        else:
            error = public_error(exc)
        body = {"code": error.code, "message": error.message}
        if error.trace_run_id is not None:
            body["trace_run_id"] = str(error.trace_run_id)
        return JSONResponse(status_code=error.status, content={"error": body})

    for kind in (APIError, RequestValidationError, HTTPException, Exception):
        app.add_exception_handler(kind, handle_error)
