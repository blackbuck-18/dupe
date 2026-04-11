import requests
import logging
import config

# The default local port where Ollama runs
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def ask_local_ai(prompt: str, context_text: str = "") -> str:
    """
    Sends a prompt and optional file context to your local Ollama LLM.
    Returns the AI's generated text.
    """
    # 1. Combine the user's question with the context from ChromaDB
    if context_text.strip():
        # Truncate context if it's too massive (protects CPU memory)
        words = context_text.split()
        if len(words) > config.MAX_CONTEXT_WORDS:
            context_text = " ".join(words[:config.MAX_CONTEXT_WORDS]) + "..."
            
        system_prompt = (
            f"You are the FileSense AI. Use the following extracted file snippets "
            f"to answer the user's question accurately. If the answer is not in the "
            f"snippets, just say you don't know based on the provided files.\n\n"
            f"--- FILE CONTEXT ---\n{context_text}\n\n"
            f"--- USER QUESTION ---\n{prompt}"
        )
    else:
        system_prompt = prompt

    # 2. Prepare the payload for Ollama
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": system_prompt,
        "stream": False,  # Set to True later if you want typewriter text effects
        "options": {
            "temperature": 0.2  # Keep it low so the AI is factual, not overly creative
        }
    }

    # 3. Make the connection
    try:
        response = requests.post(
            OLLAMA_API_URL, 
            json=payload, 
            timeout=config.INFERENCE_TIMEOUT
        )
        
        # --- NEW: Catch the exact 404 Missing Model Error ---
        if response.status_code == 404:
            return (
                "🚨 **AI Model Not Found!**<br><br>"
                "Ollama is running, but the AI brain hasn't been downloaded yet.<br>"
                "Open your terminal and run: <code>ollama run llama3</code>"
            )
            
        response.raise_for_status()
        
        data = response.json()
        return data.get("response", "Error: Empty response from AI.")
        
    except requests.exceptions.ConnectionError:
        return (
            "🚨 **Connection Failed:** Could not find Ollama. \n\n"
            "1. Is the Ollama app installed? (Download from ollama.com)\n"
            "2. Is the Ollama app currently running in your system tray?\n"
            f"3. Have you downloaded the model? (Run `ollama run {config.OLLAMA_MODEL}` in terminal)"
        )
    except requests.exceptions.Timeout:
        return "⚠️ **Timeout:** The local AI took too long to respond. Try asking a simpler question."
    except Exception as e:
        logging.error(f"Ollama API Error: {str(e)}")
        return f"⚠️ **Unexpected Error:** {str(e)}"
