from contextlib import asynccontextmanager
from fastapi import FastAPI

from dependency_container import DependencyContainer
from router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing dependencies...")
    DependencyContainer.init()
    print("Dependencies initialized")

    yield

    print("Shutting down dependencies...")
    await DependencyContainer.get_instance().shutdown()
    print("Dependencies shut down")


application = FastAPI(lifespan=lifespan)
application.include_router(router, prefix="/api")