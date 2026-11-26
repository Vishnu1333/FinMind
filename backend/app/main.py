import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router as api_router
from .api.auth import router as auth_router
from .api.accounts import router as accounts_router
from .api.transactions import router as transactions_router
from .api.categories import router as categories_router
from .api.business import router as business_router
from .api.analytics import router as analytics_router
from .api.documents import router as documents_router
from .api.knowledge import router as knowledge_router
from .api.assistant import router as assistant_router
from .api.notifications import router as notifications_router
from .api.cards import router as cards_router
from .api.help_desk import router as help_desk_router
from .api.settings import router as settings_router
from .api.predictions import router as predictions_router

app = FastAPI(title="FinMind", version="1.0.0", description="AI-powered SME financial intelligence platform")

# CORS – the Next.js frontend is a separate origin during development/production.
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(auth_router, prefix="/auth")
app.include_router(accounts_router)
app.include_router(transactions_router)
app.include_router(categories_router)
app.include_router(business_router)
app.include_router(analytics_router, prefix="/analytics")
app.include_router(documents_router)
app.include_router(knowledge_router)
app.include_router(assistant_router)
app.include_router(notifications_router)
app.include_router(cards_router)
app.include_router(help_desk_router)
app.include_router(settings_router)
app.include_router(predictions_router)
