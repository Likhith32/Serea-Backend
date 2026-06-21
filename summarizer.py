import re
from typing import List, Dict, Any
import google.generativeai as genai
from config import settings

def chunk_text(text: str, chunk_size: int = 6000, overlap: int = 500) -> List[str]:
    """
    Splits the input text into overlapping chunks.
    Default chunk size: 6000 characters (approx 1000-1500 words).
    Default overlap: 500 characters.
    """
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start += (chunk_size - overlap)
    return chunks

def call_gemini(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """
    Calls the Gemini API to generate text completions.
    """
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in the backend environment.")
        
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            system_instruction=system_prompt
        )
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Error calling Gemini API: {str(e)}")

def summarize_chunks_map_reduce(chunks: List[str]) -> str:
    """
    Summarizes long documents by summarizing individual chunks first (Map),
    then synthesizing the intermediate summaries into a final cohesive summary (Reduce).
    """
    if not chunks:
        return "No text available to summarize."
        
    if len(chunks) == 1:
        prompt = (
            "Please provide a concise summary of the following text. "
            "Highlight the key points as clear bullet points:\n\n"
            f"{chunks[0]}"
        )
        return call_gemini(
            prompt, 
            system_prompt="You are a precise document summarization assistant. Always respond with a concise summary followed by key bullet points."
        )

    print(f"Starting Map-Reduce summary for {len(chunks)} chunks...")
    
    # Step 1: Map (Summarize each chunk)
    chunk_summaries = []
    for idx, chunk in enumerate(chunks):
        print(f"Calling Gemini for chunk {idx+1}/{len(chunks)}...")
        prompt = f"Summarize the following section of a larger document:\n\n{chunk}"
        summary = call_gemini(prompt, system_prompt="Summarize the text concisely.")
        chunk_summaries.append(summary)
        print(f"Finished chunk {idx+1}/{len(chunks)}!")
        
    # Step 2: Reduce
    print("Synthesizing final combined summary...")
    combined_summary = "\n\n".join(chunk_summaries)
    
    prompt = (
        "Please synthesize the following section summaries into a single, cohesive, concise final summary. "
        "Highlight the key points as clear bullet points:\n\n"
        f"{combined_summary}"
    )
    return call_gemini(
        prompt, 
        system_prompt="You are a precise document summarization assistant. Always respond with a cohesive, concise summary followed by key bullet points."
    )

def summarize_by_keyword(chunks: List[str], keyword: str) -> str:
    """
    Ranks chunks based on keyword matches, selects the most relevant ones, 
    and generates a summary focused strictly on the keyword topic.
    """
    if not chunks:
        return "No text available to summarize."
        
    keyword_lower = keyword.lower().strip()
    if not keyword_lower:
        return "Please enter a valid keyword to summarize."
        
    scored_chunks = []
    for chunk in chunks:
        # Search for keyword (case-insensitive substring count)
        escaped_keyword = re.escape(keyword_lower)
        matches = re.findall(escaped_keyword, chunk.lower())
        score = len(matches)
        if score > 0:
            scored_chunks.append((score, chunk))
            
    if not scored_chunks:
        return f"No relevant content was found containing the keyword '{keyword}'."
        
    # Sort chunks by relevance score descending
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Select the top 3 most relevant chunks to prevent context overflow
    top_chunks = [item[1] for item in scored_chunks[:3]]
    relevant_text = "\n\n--- SECTION ---\n\n".join(top_chunks)
    
    prompt = (
        f"Summarize only the content related to '{keyword}' from the following text. "
        "Do not include unrelated details. Present the summary clearly and concisely, using bullet points for key findings:\n\n"
        f"{relevant_text}"
    )
    
    print(f"Calling Gemini for keyword '{keyword}' using {len(top_chunks)} relevant chunks...")
    result = call_gemini(
        prompt, 
        system_prompt=f"You are an assistant that specializes in extracting and summarizing content related to the keyword: '{keyword}'."
    )
    print("Keyword summary generated!")
    return result

def search_text(text_by_page: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
    """
    Performs case-insensitive search for sentences containing the search term.
    Returns list of dicts with 1-indexed page number and matching sentence.
    """
    results = []
    term_lower = search_term.lower().strip()
    if not term_lower:
        return results
        
    for page_data in text_by_page:
        page_num = page_data["page"]
        page_text = page_data["text"]
        
        # Split text into sentences using regex boundary lookbehind
        sentences = re.split(r'(?<=[.!?])\s+', page_text)
        
        for sentence in sentences:
            sentence_clean = sentence.strip().replace('\n', ' ')
            if not sentence_clean:
                continue
            # Case-insensitive substring match
            if term_lower in sentence_clean.lower():
                results.append({
                    "page": page_num,
                    "text": sentence_clean
                })
    return results
