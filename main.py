# main.py

from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="AI Inspection - NAAC Document Classifier")

app.include_router(router)
