"""Transaction categorization API. Run from the project root: uvicorn api.main:app"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import categorize, load_model


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load at startup so a missing model stops the service immediately.
    load_model()
    yield


app = FastAPI(title="Personal Finance Analyzer API", lifespan=lifespan)


class CategorizeRequest(BaseModel):
    description: str = Field(min_length=1, max_length=500)


class CategorizeResponse(BaseModel):
    category: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/categorize", response_model=CategorizeResponse)
def categorize_transaction(request: CategorizeRequest) -> CategorizeResponse:
    return CategorizeResponse(category=categorize([request.description])[0])
