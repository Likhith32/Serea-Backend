import uuid
import time
import re
import io
import json
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings
from pdf_parser import parse_pdf, PDFParsingError, get_pdf_form_fields, fill_pdf_form
from summarizer import chunk_text, summarize_text, summarize_by_keyword, search_text, call_gemini


app = FastAPI(title="PDF Summarizer & Keyword Tools API")

# Configure CORS so our React frontend can interact with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session database (tied to server lifecycle)
# Structure: { session_id: { text_by_page, full_text, chunks, metadata, created_at } }
sessions: Dict[str, Dict[str, Any]] = {}

def clean_old_sessions():
    """
    Background task to evict sessions older than 2 hours or to cap total
    in-memory sessions to prevent memory exhaustion in demo environments.
    """
    now = time.time()
    # Evict sessions older than 2 hours (7200 seconds)
    expired_ids = [sid for sid, data in sessions.items() if now - data.get("created_at", 0) > 7200]
    for sid in expired_ids:
        sessions.pop(sid, None)
        
    # Cap total sessions to 50, evicting the oldest if limit exceeded
    if len(sessions) > 50:
        sorted_sessions = sorted(sessions.items(), key=lambda x: x[1].get("created_at", 0))
        for sid, _ in sorted_sessions[:15]:
            sessions.pop(sid, None)

# Request Models
class SummarizeRequest(BaseModel):
    session_id: str

class KeywordSummarizeRequest(BaseModel):
    session_id: str
    keyword: str

class SearchRequest(BaseModel):
    session_id: str
    search_term: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: List[ChatMessage]

class FillFormRequest(BaseModel):
    session_id: str
    field_values: Dict[str, Any] = None


@app.get("/")
def read_root():
    return {"status": "running", "model": settings.GEMINI_MODEL}

@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts PDF upload, parses full text using PyMuPDF, chunks it,
    and returns a unique session ID. Checks size limits (max 20MB).
    """
    # Evict stale sessions periodically on upload
    background_tasks.add_task(clean_old_sessions)
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    # Read bytes and check size
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File size exceeds the limit of {settings.MAX_UPLOAD_SIZE / (1024 * 1024):.1f}MB."
        )
        
    try:
        parsed_data = parse_pdf(contents)
    except PDFParsingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error parsing PDF: {str(e)}")
        
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    # Store session data in memory
    sessions[session_id] = {
        "text_by_page": parsed_data["text_by_page"],
        "full_text": parsed_data["full_text"],
        "chunks": chunk_text(parsed_data["full_text"]),
        "metadata": parsed_data["metadata"],
        "file_bytes": contents,
        "form_state": {},
        "created_at": time.time()
    }
    
    return {
        "session_id": session_id,
        "page_count": parsed_data["page_count"],
        "metadata": parsed_data["metadata"]
    }

@app.post("/summarize")
def summarize(request: SummarizeRequest):
    """
    Generates a normal summary from the extracted PDF text (using Map-Reduce).
    """
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found or has expired.")
        
    session_data = sessions[session_id]
    
    try:
        summary_text = summarize_text(session_data["full_text"])
        return {"summary": summary_text}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@app.post("/summarize-keyword")
def summarize_keyword(request: KeywordSummarizeRequest):
    """
    Generates a focused summary around a specific keyword using relevant text chunks.
    """
    session_id = request.session_id
    keyword = request.keyword.strip()
    
    if not keyword:
        raise HTTPException(status_code=400, detail="Keyword cannot be empty.")
        
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found or has expired.")
        
    session_data = sessions[session_id]
    
    try:
        keyword_summary = summarize_by_keyword(session_data["chunks"], keyword)
        return {"summary": keyword_summary}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate keyword summary: {str(e)}")

@app.post("/search")
def search(request: SearchRequest):
    """
    Searches the PDF text for matching sentences, returning page numbers and clean lines.
    """
    session_id = request.session_id
    search_term = request.search_term.strip()
    
    if not search_term:
        raise HTTPException(status_code=400, detail="Search term cannot be empty.")
        
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found or has expired.")
        
    session_data = sessions[session_id]
    
    try:
        search_results = search_text(session_data["text_by_page"], search_term)
        return {"results": search_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/chat")
def chat(request: ChatRequest):
    """
    Handles PDF-scoped Q&A or Form-Filling depending on user intent.
    """
    session_id = request.session_id
    message = request.message.strip()
    history = request.history

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found or has expired.")

    session_data = sessions[session_id]

    # 1. Classify intent (fill_form vs qa) using Gemini
    intent = "qa"
    intent_prompt = (
        "Analyze the following user message for a PDF assistant and classify its intent.\n\n"
        "Available Intents:\n"
        "- \"fill_form\": The user wants to fill, complete, or update a form/document, or is answering/providing values for fields in a form (e.g. \"fill the form\", \"complete it\", \"my name is John\", \"here is my email\").\n"
        "- \"qa\": The user is asking a question about the document, general Q&A, or anything else (e.g. \"what is the revenue?\", \"summarize page 2\").\n\n"
        f"User Message: \"{message}\"\n\n"
        "Respond with ONLY a JSON object containing the intent, like this:\n"
        "{\"intent\": \"fill_form\"} or {\"intent\": \"qa\"}"
    )

    try:
        raw_intent = call_gemini(intent_prompt, system_prompt="You are an intent classifier. Respond with valid JSON only.")
        cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', raw_intent).strip()
        parsed = json.loads(cleaned)
        if parsed.get("intent") in ["fill_form", "qa"]:
            intent = parsed["intent"]
    except Exception:
        # Fallback keyword match
        keywords = ["fill", "complete the form", "fill it out", "populate", "form values", "fill form", "form fields"]
        if any(k in message.lower() for k in keywords):
            intent = "fill_form"

    # 2. Process based on intent
    try:
        if intent == "qa":
            # RAG-based search
            words = re.findall(r'\b\w+\b', message.lower())
            stopwords = {
                "what", "is", "the", "a", "of", "and", "in", "to", "for", "on", "with", "at", 
                "by", "from", "an", "how", "why", "where", "who", "which", "are", "do", "does",
                "can", "could", "would", "should", "you", "i", "he", "she", "they", "we", "it", 
                "me", "him", "her", "them", "us", "my", "your", "his", "their", "our", "its", 
                "about", "this", "that", "these", "those", "or", "but", "if", "then", "else", 
                "any", "all", "some", "none", "pdf", "document", "file"
            }
            keywords = [w for w in words if w not in stopwords and len(w) > 2]
            if not keywords:
                keywords = [w for w in words if len(w) > 2]

            scored_chunks = []
            chunks = session_data.get("chunks", [])
            for chunk in chunks:
                chunk_lower = chunk.lower()
                score = sum(chunk_lower.count(k) for k in keywords)
                if score > 0:
                    scored_chunks.append((score, chunk))

            if scored_chunks:
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                relevant_chunks = [c for s, c in scored_chunks[:3]]
            else:
                relevant_chunks = chunks[:3]

            relevant_text = "\n\n--- SECTION ---\n\n".join(relevant_chunks)

            # Construct message history
            history_text = ""
            recent_history = history[-10:] if history else []
            for msg in recent_history:
                role_name = "User" if msg.role == "user" else "Assistant"
                history_text += f"{role_name}: {msg.content}\n"

            prompt = (
                "Answer the user's question using only the provided document content.\n"
                "If the answer isn't in the text, say so clearly. Do not invent any facts.\n\n"
                f"Document Content:\n{relevant_text}\n\n"
                f"Conversation History:\n{history_text}\n"
                f"User: {message}\n"
                "Assistant:"
            )

            bot_response = call_gemini(
                prompt,
                system_prompt="You are Serea, a precise PDF assistant. Answer only based on the document content."
            )
            return {"response": bot_response, "intent": "qa"}

        else:  # fill_form
            file_bytes = session_data.get("file_bytes")
            fields = get_pdf_form_fields(file_bytes)
            if not fields:
                # Fallback: Extract blank fields from PDF content text as plain text
                sample_text = "\n\n".join(session_data.get("chunks", [])[:3])
                prompt = (
                    "The user wants to fill out this document, but it does not contain interactive PDF form fields.\n"
                    "Analyze the following document content, identify any blank sections, questionnaires, tables, or fields that seem to require user input, and list/summarize them clearly as plain text so the user knows what information to provide.\n\n"
                    f"Document Content:\n{sample_text}"
                )
                bot_response = (
                    "This PDF does not contain interactive, fillable form fields (AcroForm widgets).\n"
                    "However, here are the sections and information that appear to require filling:\n\n"
                )
                bot_response += call_gemini(prompt, system_prompt="You are a form field extraction assistant. List the blank sections clearly.")
                return {"response": bot_response, "intent": "fill_form"}

            # Form fields exist! Map incoming message to fields
            fields_desc = ", ".join([f"'{f['name']}'" for f in fields])
            form_state = session_data.setdefault("form_state", {})

            extract_prompt = (
                "You are an expert data extraction assistant.\n"
                f"The user is filling out a form with the following fields: [{fields_desc}].\n"
                f"Current filled values: {form_state}\n\n"
                f"User message: \"{message}\"\n\n"
                "Extract any new or updated field values provided in the user's message.\n"
                "Return a JSON object mapping the field names to their values.\n"
                "Only extract values explicitly provided or strongly implied (e.g. today's date if they say 'today').\n"
                "If no fields are mentioned or provided in the message, return an empty JSON object: {}.\n"
                "Do not invent values.\n"
                "Respond with ONLY valid JSON."
            )

            extracted_values = {}
            try:
                raw_extracted = call_gemini(extract_prompt, system_prompt="You are a data extractor. Respond with valid JSON only.")
                cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', raw_extracted).strip()
                extracted_values = json.loads(cleaned)
            except Exception:
                pass

            # Merge new extracted values
            field_names = {f["name"] for f in fields}
            for k, v in extracted_values.items():
                if k in field_names:
                    form_state[k] = v

            # Check status and format response
            filled = []
            missing = []
            for f in fields:
                name = f["name"]
                val = form_state.get(name, "") or f["value"]
                if val:
                    filled.append(f"{name}: {val}")
                else:
                    missing.append(name)

            if not filled:
                bot_response = (
                    f"I found these fillable fields in the PDF: {fields_desc}.\n"
                    "What values should I fill in for each?"
                )
            elif missing:
                filled_str = "\n".join([f"- {item}" for item in filled])
                missing_desc = ", ".join(missing)
                bot_response = (
                    "I've updated the form fields. Here is what we have filled in so far:\n"
                    f"{filled_str}\n\n"
                    f"What should I fill in for the remaining fields: {missing_desc}?"
                )
            else:
                filled_str = "\n".join([f"- {item}" for item in filled])
                bot_response = (
                    "Perfect! I have collected values for all the form fields:\n"
                    f"{filled_str}\n\n"
                    "You can now download the filled PDF using the button below."
                )

            return {"response": bot_response, "intent": "fill_form"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat processing: {str(e)}")


@app.get("/form-fields")
def form_fields(session_id: str):
    """
    Returns list of detected form fields in the PDF (if any).
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found or has expired.")

    session_data = sessions[session_id]
    file_bytes = session_data.get("file_bytes")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Original PDF bytes not found in session.")

    try:
        fields = get_pdf_form_fields(file_bytes)
        return {"fields": fields}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve form fields: {str(e)}")


@app.post("/fill-form")
def fill_form_endpoint(request: FillFormRequest):
    """
    Fills PDF fields and returns the resulting PDF file for download.
    """
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found or has expired.")

    session_data = sessions[session_id]
    file_bytes = session_data.get("file_bytes")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Original PDF bytes not found in session.")

    # Use explicitly passed values, fallback to accumulated form_state in session
    values = request.field_values if request.field_values is not None else session_data.get("form_state", {})

    try:
        filled_bytes = fill_pdf_form(file_bytes, values)
        filename = f"filled_{session_data['metadata'].get('title', 'form')}.pdf"
        # Clean filename of standard special chars
        filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        return StreamingResponse(
            io.BytesIO(filled_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=\"{filename}\"",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fill form: {str(e)}")

