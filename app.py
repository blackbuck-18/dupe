import streamlit as st
import os
import time
import config
import pandas as pd

# ==========================================
# 0. PAGE CONFIG (Must be the absolute first command!)
# ==========================================
st.set_page_config(page_title="FileSense", page_icon="📂", layout="wide")

# ==========================================
# 1. INTEGRATION SAFETY & BACKEND LOADING
# ==========================================
try:
    from backend.vector_engine import VectorDB
    from backend.parser import extract_text_from_file
    BACKEND_READY = True
except ImportError:
    BACKEND_READY = False

# ==========================================
# 2. STATE MANAGEMENT 
# ==========================================
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
    
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hello! I am your local AI. How can I help you analyze or summarize your files today?"}
    ]

# Initialize the AI Database exactly once to save memory
if 'vector_db' not in st.session_state:
    if BACKEND_READY:
        st.session_state.vector_db = VectorDB()
    else:
        st.session_state.vector_db = None

# Initialize the Soft-Delete Memory Stack
if 'undo_stack' not in st.session_state:
    st.session_state.undo_stack = []
if 'pending_deletes' not in st.session_state:
    st.session_state.pending_deletes = []

# ==========================================
# 3. MAIN PAGE SETUP
# ==========================================
st.title("📂 FileSense: AI File Organizer")

if not BACKEND_READY:
    st.warning("⚠️ Backend modules not found. UI running in standalone mode (using dummy data).")
else:
    st.success("✅ Backend fully connected. Local AI Engine is active.")

# ==========================================
# 4. SIDEBAR LAYOUT (Smart Sync Logic!)
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings & Scanning")
    target_folder = st.text_input("Target Folder Path", value=str(config.DATA_DIR))
    
    if st.button("Scan Directory"):
        if os.path.exists(target_folder) and os.path.isdir(target_folder):
            if BACKEND_READY:
                st.info("Syncing folder with AI memory...")
                
                valid_extensions = ['.pdf', '.docx', '.txt']
                files_to_scan = [f for f in os.listdir(target_folder) if os.path.splitext(f)[1].lower() in valid_extensions]
                current_filepaths = [os.path.join(target_folder, f) for f in files_to_scan]
                
                # 1. Roll Call
                db_files = st.session_state.vector_db.get_file_metadata()
                memorized_filepaths = list(db_files.keys())
                
                # 2. Forget the Ghosts
                ghosts_removed = 0
                for ghost_path in memorized_filepaths:
                    if ghost_path not in current_filepaths:
                        st.session_state.vector_db.remove_file(ghost_path)
                        ghosts_removed += 1
                        
                # 3. Find New OR Modified Files
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
                    st.success("Everything is already up to date! No new or modified files to scan.")
                else:
                    if files_to_process:
                        progress_text = "Parsing new and updated files..."
                        my_bar = st.progress(0, text=progress_text)
                        total_files = len(files_to_process)
                        
                        st.session_state.scan_results = [] 
                        
                        for idx, filepath in enumerate(files_to_process):
                            filename = os.path.basename(filepath)
                            parsed_data = extract_text_from_file(filepath)
                            
                            if not parsed_data.get('error'):
                                st.session_state.vector_db.add_file(
                                    filename=parsed_data['filename'],
                                    filepath=parsed_data['filepath'],
                                    text=parsed_data['text_content'],
                                    mtime=os.path.getmtime(filepath)
                                )
                                st.session_state.scan_results.append(parsed_data)
                                
                            percent_complete = int(((idx + 1) / total_files) * 100)
                            my_bar.progress(percent_complete, text=f"Processed {filename}...")
                    
                    st.success(f"Sync complete! Removed {ghosts_removed} deleted files and updated/scanned {len(files_to_process)} files.")
            else:
                time.sleep(1)
                st.success("Scanned successfully (Dummy Data mode).")
                st.session_state.scan_results = [{'filename': 'dummy.pdf', 'filepath': 'path', 'text_content': 'dummy', 'metadata': {'size': '1MB'}}]
        else:
            st.error("Invalid folder path. Please check and try again.")

# ==========================================
# 5. TABS LAYOUT 
# ==========================================
# ADDED a 5th tab for Database Insights!
tab_search, tab_cluster, tab_editor, tab_manage, tab_insights = st.tabs(["🔍 Search Files", "🧠 Smart Clusters", "🤖 AI Editor", "🗄️ Manage Files", "📊 Insights"])

# --- TAB 1: SEARCH ---
with tab_search:
    st.header("Search Your Offline Files")
    search_query = st.text_input("What are you looking for?")
    
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
                            st.markdown(f"**📄 {match['filename']}** (Distance: `{match['distance']}`)")
                            st.caption(f"Path: {match['filepath']}")
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
                    st.write(f"### 🤖  {cluster_id}")
                    for f in files:
                        st.write(f"- {f}")
        else:
            st.success("Simulated Clusters generated successfully!")

# --- TAB 3: AI EDITOR ---
with tab_editor:
    st.header("Offline AI Editor")
    st.caption("Strictly routes to local Ollama engine.")
    
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    if prompt := st.chat_input("Ask the AI to analyze, summarize, or draft new text..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            dummy_reply = f"I am running locally! You asked: '{prompt}'. (Ollama LLM connection coming in Phase 2!)"
            st.write(dummy_reply)
            st.session_state.chat_messages.append({"role": "assistant", "content": dummy_reply})

# --- TAB 4: MANAGE FILES ---
with tab_manage:
    st.header("Database File Management")
    st.write("Safely remove indexed files from your AI's memory.")

    if BACKEND_READY and st.session_state.vector_db:
        if st.session_state.undo_stack:
            st.warning(f"🗑️ You have {len(st.session_state.undo_stack)} file(s) pending permanent deletion.")
            col_undo, col_confirm = st.columns(2)
            
            with col_undo:
                if st.button("↩️ Undo Last Deletion", use_container_width=True):
                    recovered_filepath = st.session_state.undo_stack.pop()
                    st.session_state.pending_deletes.remove(recovered_filepath)
                    
                    filename = os.path.basename(recovered_filepath)
                    st.success(f"✅ Recovered '{filename}' back to active database!")
                    time.sleep(1)
                    st.rerun()
            
            with col_confirm:
                if st.button("⚠️ Permanently Delete All", type="primary", use_container_width=True):
                    for filepath in st.session_state.pending_deletes:
                        st.session_state.vector_db.remove_file(filepath)
                    
                    st.session_state.undo_stack.clear()
                    st.session_state.pending_deletes.clear()
                    
                    st.error("Data permanently wiped from database.")
                    time.sleep(1)
                    st.rerun()
            
            st.divider()

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
                        st.session_state.pending_deletes.append(filepath)
                        st.session_state.undo_stack.append(filepath)
                        
                        st.toast(f"Moved '{filename}' to trash.", icon="🗑️")
                        st.rerun()

# --- TAB 5: INSIGHTS & ANALYTICS ---
with tab_insights:
    st.header("📊 Database Insights")
    st.write("A real-time overview of your local AI knowledge base.")

    if BACKEND_READY and st.session_state.vector_db:
        db_files = st.session_state.vector_db.get_file_metadata()
        active_filepaths = [fp for fp in db_files.keys() if fp not in st.session_state.pending_deletes]

        if not active_filepaths:
            st.info("No data available to visualize. Scan a folder first!")
        else:
            # 1. Calculate Metrics
            total_files = len(active_filepaths)
            file_types = {}
            total_size_bytes = 0
            
            for fp in active_filepaths:
                # Get file extension (e.g., '.pdf') and remove the dot
                ext = os.path.splitext(fp)[1].lower().replace(".", "").upper()
                if not ext:
                    ext = "UNKNOWN"
                    
                file_types[ext] = file_types.get(ext, 0) + 1
                
                # Check actual file size on the hard drive
                if os.path.exists(fp):
                    total_size_bytes += os.path.getsize(fp)

            # Convert bytes to Megabytes (MB)
            total_size_mb = total_size_bytes / (1024 * 1024)

            # 2. Render Top KPI Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Indexed Files", total_files)
            with col2:
                st.metric("Total Storage Tracked", f"{total_size_mb:.2f} MB")
            with col3:
                st.metric("Document Types", len(file_types))

            st.divider()

            # 3. Render Visual Charts and Insights
            col_chart1, col_chart2 = st.columns([2, 1])

            with col_chart1:
                st.subheader("Distribution by File Type")
                # Create a Pandas DataFrame to perfectly format the Streamlit Bar Chart
                chart_data = pd.DataFrame(
                    {"Count": list(file_types.values())}, 
                    index=list(file_types.keys())
                )
                st.bar_chart(chart_data, color="#1E88E5")

            with col_chart2:
                st.subheader("AI Knowledge Base Summary")
                st.write(f"- Your local AI engine currently has instant memory access to **{total_files} distinct documents**.")
                st.write(f"- **{max(file_types, key=file_types.get)}** files make up the majority of your dataset.")
                st.write("- **Vector Compression:** By turning text into embeddings, ChromaDB consumes a fraction of the original storage space, allowing hyper-fast semantic search.")
