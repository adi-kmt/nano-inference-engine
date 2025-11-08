import asyncio
from typing import Optional, AsyncGenerator
from threading import Lock

from fastapi import Depends
from engine.engine import Engine


class DependencyContainer:
    _instance: Optional["DependencyContainer"] = None
    _lock = Lock()

    def __init__(self) -> None:
        if self._instance is not None:
            raise RuntimeError("Use DependencyContainer.get_instance() or init()")
        self._engine: Optional[Engine] = None
        self._is_initialized = False

    @classmethod
    def init(cls, engine: Optional[Engine] = None) -> "DependencyContainer":
        """Initialize singleton (idempotent)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = cls.__new__(cls)
                    instance._engine = engine or Engine()
                    instance._is_initialized = True
                    cls._instance = instance
        return cls._instance

    @classmethod
    def get_instance(cls) -> "DependencyContainer":
        """Get singleton instance — raises if not initialized."""
        if cls._instance is None or not cls._instance._is_initialized:
            raise RuntimeError(
                "DependencyContainer not initialized. Call DependencyContainer.init() first."
            )
        return cls._instance

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("Engine not initialized")
        return self._engine

    async def shutdown(self) -> None:
        """Async cleanup — idempotent."""
        if self._engine is not None:
            shutdown_method = getattr(self._engine, "shutdown", None)
            if shutdown_method:
                if asyncio.iscoroutinefunction(shutdown_method):
                    await shutdown_method()
                else:
                    shutdown_method()
            self._engine = None
            self._is_initialized = False

    @staticmethod
    def provide_engine() -> Engine:
        return DependencyContainer.get_instance().engine


def get_engine() -> Engine:
    return DependencyContainer.provide_engine()