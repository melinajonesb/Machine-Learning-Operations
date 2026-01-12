from fastapi import FastAPI
from loguru import logger

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """Log when the API starts up."""
    logger.info("API server starting up")


@app.get("/health")
def health():
    """Health check endpoint."""
    logger.debug("Health check requested")
    return {"ok": True}