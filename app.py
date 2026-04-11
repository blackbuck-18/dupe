import streamlit as st
import os
import time
import pandas as pd
import concurrent.futures # <-- Added for High-Speed Multi-Threading

# 1. LOCAL CONFIG & BACKEND IMPORTS
try:
    import config
    from backend.vector_engine import VectorDB
    from backend.parser import extract_text_from_file
    BACKEND_READY = True
except ImportError as e:
    BACKEND_READY = False
    print(f"Import Error: {e}")

# ==========================================
# 0. PAGE CONFIG (Must be the absolute first Streamlit command!)
# ==========================================
st.set_page_config(page_title="FileSense", page_icon="📂", layout="wide")

# ==========================================
# 2. STATE MANAGEMENT 
# ==========================================
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
    
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hello! I am your local AI. How can I help you analyze or summarize your files today?"}
    ]

if 'vector_db' not in st.session_state:
    if BACKEND_READY:
        st.session_state.vector_db = VectorDB()
    else:
        st.session_state.vector_db = None

if 'undo_stack' not in st.session_state:
    st.session_state.undo_stack = []
if 'pending_deletes' not in st.session_state:
    st.session_state.pending_deletes = []

# ==========================================
# 3. MAIN UI HEADER
# ==========================================
st.title("📂 FileSense: AI File Organizer")

if not BACKEND_READY:
    st.warning("⚠️ Backend modules not found. UI running in standalone mode.")
else:
    st.success("✅ Backend fully connected. Local AI Engine is active.")

# ==========================================
# 4. SIDEBAR: SCANNING & SETTINGS
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings & Scanning")
    
    default_path = str(config.DATA_DIR) if BACKEND_READY else ""
    target_folder = st.text_input(
        "Target Folder Path", 
        value=default_path, 
        key="unique_folder_input"
    ).strip()
    
    # NEW: Checkbox to preserve existing folder structures!
    preserve_structure = st.checkbox("📁 Keep Folder Structure (Skip AI Clustering)")
    
    if st.button("🚀 Fast Scan Directory"):
        if os.path.exists(target_folder) and os.path.isdir(target_folder):
            
            valid_exts = ['.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg', '.mp3', '.mp4', '.wav']
            current_filepaths = []
            total_files_seen = 0
            
            for root, dirs, files in os.walk(target_folder):
                total_files_seen += len(files)
                for file in files:
                    if os.path.splitext(file)[1].lower() in valid_exts:
                        full_path = os.path.join(root, file)
                        current_filepaths.append(full_path)
            
            st.warning(f"🔍 DEBUG: Found {total_files_seen} total files across all sub-folders.")
            st.info(f"🔍 DEBUG: {len(current_filepaths)} files matched supported formats.")

            if BACKEND_READY:
                st.info("Syncing entire directory tree with AI memory...")
                
                db_files = st.session_state.vector_db.get_file_metadata()
                memorized_filepaths = list(db_files.keys())
                
                # 1. DELETE GHOSTS 
                ghosts_removed = 0
                for ghost_path in memorized_filepaths:
                    if ghost_path not in current_filepaths:
                        st.session_state.vector_db.remove_file(ghost_path)
                        ghosts_removed += 1
                        
                # 2. FIND CHANGES
                files_to_process = []
                for filepath in current_filepaths:
                    if filepath not in memorized_filepaths:
                        files_to_process.append(filepath) 
                    else:
                        current_mtime = os.path.getmtime(filepath)
                        saved_mtime = db_files.get(filepath, 0.0)
                        if current_mtime > saved_mtime:
                            files_to_process.append(filepath) 
                
                if not files_to_process and ghosts_removed == 0:
                    st.success("Everything is already up to date! No new changes detected.")
                else:
                    # 3. PYTORCH-OPTIMIZED SCANNING
                    my_bar = st.progress(0, text="Initializing AI analysis...")
                    processed_count = 0
                    total_to_process = len(files_to_process)
                    
                    for idx, filepath in enumerate(files_to_process):
                        # 1. Read the text (Fast I/O)
                        parsed_data = extract_text_from_file(filepath)
                        
                        if not parsed_data.get('error'):
                            # Find the actual name of the folder this file is sitting in
                            actual_folder_name = os.path.basename(os.path.dirname(filepath))
                            
                            # 2. Let PyTorch's native OpenMP handle the C++ parallel embedding
                            st.session_state.vector_db.add_file(
                                filename=parsed_data['filename'],
                                filepath=parsed_data['filepath'],
                                text=parsed_data['text_content'],
                                mtime=os.path.getmtime(filepath),
                                preserve_structure=preserve_structure,  # <--- Passes the UI checkbox flag
                                parent_folder=actual_folder_name        # <--- Passes the actual Windows folder name
                            )
                            processed_count += 1
                        
                        # Update progress bar
                        percent = int(((idx + 1) / total_to_process) * 100)
                        my_bar.progress(percent, text=f"Vectorizing: {os.path.basename(filepath)}")
                    
                    st.success(f"✅ Sync complete! Scanned {processed_count} files and removed {ghosts_removed} deleted files.")
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("AI Backend failed to load. Check terminal.")
        else:
            st.error(f"❌ Path Error: The folder '{target_folder}' does not exist.")

    st.divider()
    st.subheader("Danger Zone")
    if st.button("🚨 Wipe AI Memory", type="primary", use_container_width=True):
        if BACKEND_READY:
            with st.spinner("Erasing knowledge..."):
                st.session_state.vector_db.clear_database()
                st.session_state.undo_stack.clear()
                st.session_state.pending_deletes.clear()
                st.session_state.scan_results = []
            st.success("✅ AI Memory wiped!")
            time.sleep(1)
            st.rerun()

# ==========================================
# 5. TABS LAYOUT 
# ==========================================
tab_search, tab_cluster, tab_editor, tab_manage, tab_insights = st.tabs([
    "🔍 Search Files", "🧠 Smart Clusters", "🤖 AI Editor", "🗄️ Manage Files", "📊 Insights"
])

# --- TAB 1: SEARCH ---
with tab_search:
    st.header("Search Your Offline Files")
    search_query = st.text_input("What are you looking for?", placeholder="e.g. machine learning project ideas")
    
    if search_query:
        if not BACKEND_READY:
            st.info(f"Dummy search results for: {search_query}")
        else:
            with st.spinner("Searching vector space..."):
                search_results = st.session_state.vector_db.search_documents(query_text=search_query)
                
                if "error" in search_results:
                    st.info(search_results["error"])
                else:
                    st.success(f"Found {len(search_results['matches'])} relevant matches.")
                    for match in search_results['matches']:
                        with st.container(border=True):
                            st.markdown(f"**📄 {match['filename']}**")
                            st.caption(f"Path: {match['filepath']} | Match Score: {match['distance']}")
                            st.write(match['snippet'])

# --- TAB 2: CLUSTERING ---
with tab_cluster:
    st.header("Group Similar Files")
    st.write("Use AI to automatically group your files by topic.")
    
    if st.button("Group Similar Files"):
        if BACKEND_READY:
            cluster_results = st.session_state.vector_db.cluster_files()
            if 'error' in cluster_results:
                st.error(cluster_results['error'])
            elif 'warning' in cluster_results:
                st.warning(cluster_results['warning'])
            else:
                st.success("Clusters generated successfully!")
                for cluster_id, files in cluster_results.items():
                    with st.expander(f"🤖 {cluster_id}", expanded=True):
                        for f in files:
                            st.write(f"- {f}")
        else:
            st.success("Simulated Clusters generated successfully!")

# --- TAB 3: AI EDITOR ---
with tab_editor:
    st.header("Offline AI Editor")
    st.caption(f"Powered by local {config.OLLAMA_MODEL} engine.")
    
    # Render previous chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # When the user types a question...
    if prompt := st.chat_input("Ask the AI about your scanned files..."):
        
        # 1. Save and display the user's prompt
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        # 2. Generate the AI response
        with st.chat_message("assistant"):
            with st.spinner("🧠 AI is thinking..."):
                
                # OPTIONAL: Search your database first to give the AI context!
                context_str = ""
                if BACKEND_READY and st.session_state.vector_db:
                    # We silently search ChromaDB for the 3 most relevant paragraphs
                    search_results = st.session_state.vector_db.search_documents(prompt, top_k=3)
                    if "matches" in search_results:
                        for match in search_results["matches"]:
                            context_str += f"[From {match['filename']}]: {match['snippet']}\n\n"
                
                # Import the bridge and ask the LLM
                from backend.ollama_bridge import ask_local_ai
                ai_reply = ask_local_ai(prompt=prompt, context_text=context_str)
                
                # Display the answer and save it to history
                st.write(ai_reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_reply})

# --- TAB 4: MANAGE FILES ---
with tab_manage:
    st.header("Database File Management")
    if BACKEND_READY and st.session_state.vector_db:
        db_files = st.session_state.vector_db.get_file_metadata()
        active_filepaths = [fp for fp in db_files.keys() if fp not in st.session_state.pending_deletes]

        if not active_filepaths:
            st.info("No active files found in the database.")
        else:
            for filepath in active_filepaths:
                filename = os.path.basename(filepath)
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**📄 {filename}**")
                    st.caption(filepath)
                with col2:
                    if st.button("Delete", key=f"del_{filepath}"):
                        st.session_state.vector_db.remove_file(filepath)
                        st.toast(f"Removed '{filename}' from AI memory.")
                        time.sleep(0.5)
                        st.rerun()

# --- TAB 5: INSIGHTS ---
with tab_insights:
    st.header("📊 Database Insights")
    if BACKEND_READY and st.session_state.vector_db:
        db_files = st.session_state.vector_db.get_file_metadata()
        if not db_files:
            st.info("No data to visualize. Scan a folder first!")
        else:
            file_types = {}
            for fp in db_files.keys():
                ext = os.path.splitext(fp)[1].lower().replace(".", "").upper() or "UNKNOWN"
                file_types[ext] = file_types.get(ext, 0) + 1
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Distribution by File Type")
                chart_data = pd.DataFrame({"Count": list(file_types.values())}, index=list(file_types.keys()))
                st.bar_chart(chart_data, color="#1E88E5")
            with col2:
                st.subheader("Summary")
                st.metric("Total Files", len(db_files))
                st.write(f"Your primary format is **{max(file_types, key=file_types.get)}**.")
