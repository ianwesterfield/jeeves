import inspect
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from pathlib import Path
from app.api import memory

print("[main] Starting OpenWebUI Memory Service...")

app = FastAPI(title="OpenWebUI Memory Service", version="1.0")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"[main] Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"[main] Response status: {response.status_code}")
    return response

# CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://localhost:8080",
    "http://open-webui:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static plugin files from app/static under /.well-known
static_dir = Path(__file__).resolve().parent / "static"
print(f"[main] Mounting static files from {static_dir}")
app.mount("/.well-known", StaticFiles(directory=str(static_dir)), name="static")

print("[main] Including memory router at /api/memory")
app.include_router(memory.router, prefix="/api/memory")

# Tool discovery endpoint for Open‑WebUI
@app.get("/api/memory/filter", response_class=PlainTextResponse)
def get_memory_filter():
    """
    Return the Function class source code as plain text for Open‑WebUI to parse.
    """
    print("[main] GET /api/memory/filter called")
    
    functions_module_path = Path(__file__).resolve().parent / "memory.filter.py"
    with open(functions_module_path, "r") as f:
        return f.read()

print("[main] Memory service initialized and ready")