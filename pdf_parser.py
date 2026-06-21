import fitz  # PyMuPDF
from typing import List, Dict, Any

class PDFParsingError(Exception):
    """Custom exception raised when PDF 
    parsing fails or validation fails (e.g. OCR needed, empty)."""
    pass

def parse_pdf(file_bytes: bytes) -> Dict[str, Any]:
    """
    Parses a PDF file from bytes.
    Extracts text page-by-page and checks if the PDF contains readable text.
    
    Raises PDFParsingError if the PDF cannot be parsed, is encrypted, is empty, 
    or contains no extractable text.
    """
    try:
        # Open PDF from memory stream
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise PDFParsingError(f"Could not open PDF file. It might be corrupted. Details: {str(e)}")
        
    if doc.is_encrypted:
        raise PDFParsingError("Password-protected or encrypted PDFs are not supported.")
        
    page_count = len(doc)
    if page_count == 0:
        raise PDFParsingError("The uploaded PDF contains no pages.")
        
    text_by_page = []
    total_text_length = 0
    
    for page_num in range(page_count):
        page = doc[page_num]
        text = page.get_text()
        text_clean = text.strip()
        text_by_page.append({
            "page": page_num + 1,
            "text": text
        })
        total_text_length += len(text_clean)
        
    # If total extracted text is empty or near empty, OCR is likely required.
    if total_text_length < 10:
        raise PDFParsingError(
            "This PDF contains no readable text. Scanned or image-only PDFs are not supported (OCR is not supported)."
        )
        
    metadata = {
        "title": doc.metadata.get("title", "") or "Unknown Title",
        "author": doc.metadata.get("author", "") or "Unknown Author",
    }
    
    # Concatenate all page texts for full text operations
    full_text = "\n".join([p["text"] for p in text_by_page])
    
    return {
        "text_by_page": text_by_page,
        "page_count": page_count,
        "metadata": metadata,
        "full_text": full_text
    }

def get_pdf_form_fields(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extracts all fillable form fields from the PDF bytes.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise PDFParsingError(f"Could not open PDF file. Details: {str(e)}")
        
    fields = []
    seen = set()
    for page in doc:
        for widget in page.widgets():
            name = widget.field_name
            if name and name not in seen:
                seen.add(name)
                ftype = "Text"
                if hasattr(widget, "field_type_string") and widget.field_type_string:
                    ftype = widget.field_type_string
                elif hasattr(widget, "field_type"):
                    types = {1: "Button", 2: "Choice", 3: "Text", 4: "Signature"}
                    ftype = types.get(widget.field_type, "Text")
                fields.append({
                    "name": name,
                    "type": ftype,
                    "value": widget.field_value or ""
                })
    doc.close()
    return fields

def fill_pdf_form(file_bytes: bytes, field_values: Dict[str, Any]) -> bytes:
    """
    Fills PDF form fields with provided values and returns the new PDF as bytes.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise PDFParsingError(f"Could not open PDF file for writing. Details: {str(e)}")
        
    for page in doc:
        for widget in page.widgets():
            name = widget.field_name
            if name in field_values:
                widget.field_value = str(field_values[name])
                widget.update()
                
    output_bytes = doc.tobytes()
    doc.close()
    return output_bytes

