import streamlit as st
import os
import time
import config

# ==========================================
# 1. INTEGRATION SAFETY (The "Don't Crash" Rule)
# ==========================================
try:
    from backend.vector_engine import VectorDB
    from backend.parser import extract_text_from_file
    import ollama # Phase 2 requirement
    BACKEND_READY = True
except ImportError:
    BACKEND_READY = False

# ==========================================
# 2. STATE MANAGEMENT (Remembering data)
# ==========================================
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
    
# NEW: Memory for our AI Chat history
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hello! I am your local AI. How can I help you draft or edit your files today?"}
    ]

# ==========================================
# 3. MAIN PAGE SETUP
# ==========================================
st.set_page_config(page_title="FileSense", page_icon="📂", layout="wide")
st.title("📂 FileSense: AI File Organizer")

if not BACKEND_READY:
    st.warning("⚠️ Backend modules not found. UI running in standalone mode (using dummy data).")

# ==========================================
# 4. SIDEBAR LAYOUT (Scanning)
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings & Scanning")
    target_folder = st.text_input("Target Folder Path", value=str(config.DATA_DIR))
    
    if st.button("Scan Directory"):
        if os.path.exists(target_folder) and os.path.isdir(target_folder):
            progress_text = "Scanning files. Please wait..."
            my_bar = st.progress(0, text=progress_text)
            
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            
            st.success(f"Scanned {target_folder} successfully!")
            
            st.session_state.scan_results = [
                {'filename': 'machine_learning_notes.pdf', 'filepath': f'{target_folder}/machine_learning_notes.pdf', 'text_content': 'Placeholder text about AI models...', 'metadata': {'size': '2MB'}},
                {'filename': 'project_requirements.docx', 'filepath': f'{target_folder}/project_requirements.docx', 'text_content': 'Placeholder text for requirements...', 'metadata': {'size': '1MB'}}
            ]
        else:
            st.error("Invalid folder path. Please check and try again.")

# ==========================================
# 5. TABS LAYOUT (Search, Cluster, AI Editor)
# ==========================================
# NEW: Added the 3rd tab for Phase 2
tab1, tab2, tab3 = st.tabs(["🔍 Search Files", "🧠 Smart Clusters", "🤖 AI Editor"])

# --- TAB 1: SEARCH ---
with tab1:
    st.header("Search Your Offline Files")
    search_query = st.text_input("What are you looking for?")
    
    if search_query:
        if len(st.session_state.scan_results) == 0:
            st.info("Please scan a directory first using the sidebar!")
        else:
            st.write(f"Showing results for: **{search_query}**")
            for file in st.session_state.scan_results:
                with st.container(border=True):
                    st.subheader(file['filename'])
                    st.caption(f"Path: {file['filepath']} | Size: {file['metadata']['size']}")
                    st.write(file['text_content'])

# --- TAB 2: CLUSTERING ---
with tab2:
    st.header("Group Similar Files")
    st.write("Use AI to automatically group your files by topic.")
    
    if st.button("Group Similar Files"):
        if BACKEND_READY:
            st.info("Backend connected. Running AI clustering...")
        else:
            st.success("Simulated Clusters generated successfully!")
            st.write("### 🤖 Topic: AI & Tech")
            st.write("- machine_learning_notes.pdf")
            st.write("### 📝 Topic: Documentation")
            st.write("- project_requirements.docx")

# --- TAB 3: AI EDITOR (The Local LLM Interface) ---
with tab3:
    st.header("Offline AI Editor")
    st.caption("Strictly routes to local Ollama engine. Bypasses ChromaDB.")
    
    # 1. Display all past messages in the chat
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # 2. Wait for the user to type a new prompt
    if prompt := st.chat_input("Ask the AI to analyze, summarize, or draft new text..."):
        
        # Show the user's prompt instantly
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        # Create the AI's response logic
        with st.chat_message("assistant"):
            if BACKEND_READY:
                # In the future, this will connect directly to Ollama
                st.info("Connecting to local hardware AI...")
            else:
                # TEMPORARY BYPASS: Dummy AI Response
                dummy_reply = f"I am running locally! You asked: '{prompt}'. Once my backend is hooked up, I will use Llama 3 to help you edit your files securely."
                st.write(dummy_reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": dummy_reply})