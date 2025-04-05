# app/api/routes.py

from fastapi import APIRouter, UploadFile, File
import tempfile
from app.ml.classification import model
from app.ml.classification.extract_text import extract_text_from_pdf, extract_text_from_docx, extract_text_from_xlsx

router = APIRouter()

@router.post("/classify-document")
async def classify_document(file: UploadFile = File(...)):
    suffix = file.filename.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    if suffix == "pdf":
        text = extract_text_from_pdf(tmp_path)
    elif suffix == "docx":
        text = extract_text_from_docx(tmp_path)
    elif suffix == "xlsx":
        text = extract_text_from_xlsx(tmp_path)
    else:
        return {"error": "Unsupported file type"}

    result = model.predict(text)
    return result
