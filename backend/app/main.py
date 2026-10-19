from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Guitar Tab Transcriber API",
    description="API для разделения аудио (Demucs), транскрибации гитарных партий и личного кабинета",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "AI Guitar Tab Transcriber API is running",
        "status": "healthy",
        "version": "0.1.0",
    }


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
