import os
import fitz
from pdf_parser import get_pdf_form_fields, fill_pdf_form

def create_test_pdf():
    """
    Creates a temporary PDF file with fillable form fields (widgets) for testing.
    """
    doc = fitz.open()
    page = doc.new_page()
    
    # Draw some text
    page.insert_text((50, 30), "Serea Test Form PDF", fontsize=16)
    
    # Add a Text widget for Name
    w_name = fitz.Widget()
    w_name.rect = fitz.Rect(50, 50, 250, 75)
    w_name.field_name = "Name"
    w_name.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    w_name.field_value = ""
    page.add_widget(w_name)
    
    # Add a Text widget for Email
    w_email = fitz.Widget()
    w_email.rect = fitz.Rect(50, 90, 250, 115)
    w_email.field_name = "Email"
    w_email.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    w_email.field_value = ""
    page.add_widget(w_email)
    
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def main():
    print("--- Starting PyMuPDF Form-Field Tests ---")
    
    # 1. Create a mock form PDF in memory
    pdf_bytes = create_test_pdf()
    print("Created mock PDF in-memory.")
    
    # 2. Extract fields
    fields = get_pdf_form_fields(pdf_bytes)
    print("Extracted fields:")
    for f in fields:
        print(f" - Name: {f['name']}, Type: {f['type']}, Current Value: '{f['value']}'")
        
    assert len(fields) == 2, f"Expected 2 fields, got {len(fields)}"
    assert fields[0]["name"] == "Name", f"Expected first field to be 'Name', got {fields[0]['name']}"
    assert fields[1]["name"] == "Email", f"Expected second field to be 'Email', got {fields[1]['name']}"
    print("Form field extraction test PASSED.")
    
    # 3. Fill fields
    test_values = {"Name": "John Serea", "Email": "john@serea.ai"}
    filled_bytes = fill_pdf_form(pdf_bytes, test_values)
    print("Filled mock PDF fields in-memory.")
    
    # 4. Verify filled values
    filled_fields = get_pdf_form_fields(filled_bytes)
    print("Verified filled fields:")
    for f in filled_fields:
        print(f" - Name: {f['name']}, Value: '{f['value']}'")
        
    assert filled_fields[0]["value"] == "John Serea", f"Expected 'John Serea', got {filled_fields[0]['value']}"
    assert filled_fields[1]["value"] == "john@serea.ai", f"Expected 'john@serea.ai', got {filled_fields[1]['value']}"
    print("Form field filling test PASSED.")
    print("--- All Tests Completed Successfully ---")

if __name__ == "__main__":
    main()
