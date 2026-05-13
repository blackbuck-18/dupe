import sys
import os
import keyboard
import time
import re
# ==========================================
# 1. BACKEND IMPORTS MUST GO FIRST
# (This prevents PySide6 shiboken hooks from crashing pandas/sklearn)
# ==========================================
try:
    import config
    from backend.vector_engine import VectorDB
    from backend.ollama_bridge import ask_local_ai
    from backend.parser import extract_text_from_file
    BACKEND_READY = True
except ImportError as e:
    BACKEND_READY = False
    print(f"Backend Import Error: {e}")

# ==========================================
# 2. PYSIDE6 IMPORTS MUST GO SECOND
# ==========================================
from PySide6.QtWidgets import (QApplication, QMainWindow, QSystemTrayIcon, QMenu,
                               QVBoxLayout, QHBoxLayout, QWidget, QStyle, QLineEdit,
                               QTextBrowser, QPushButton, QLabel, QStackedWidget,
                               QFileDialog, QProgressBar, QFileSystemModel, QTreeView,
                               QHeaderView, QScrollArea, QCheckBox)
from PySide6.QtGui import QAction, QFont, QTextCursor, QIcon, QColor
from PySide6.QtCore import QObject, Signal, Qt, QThread, QPoint


# ==========================================
# THREADING: Background Scanner
# FIX: Now uses config.SUPPORTED_EXTENSIONS (was hardcoded 9-type list).
# FIX: Ghost removal — detects and removes deleted files from DB before scanning.
# FIX: Accepts preserve_structure param and passes it to db.add_file().
# ==========================================
class ScanWorker(QThread):
    progress_signal = Signal(int, str)
    finished_signal = Signal(int, int)  # processed_count, ghosts_removed

    def __init__(self, db, folder, preserve_structure=False):
        super().__init__()
        self.db = db
        self.folder = folder
        self.preserve_structure = preserve_structure

    def run(self):
        # PROBLEM 1 FIX: Use config.SUPPORTED_EXTENSIONS as the single source of truth.
        valid_exts = config.SUPPORTED_EXTENSIONS
        current_filepaths = []
        for root, dirs, files in os.walk(self.folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    current_filepaths.append(os.path.join(root, file))

        # SYNC FIX: Ghost removal — remove DB entries for files deleted from disk
        db_files = self.db.get_file_metadata()
        memorized_filepaths = list(db_files.keys())
        ghosts_removed = 0
        for ghost_path in memorized_filepaths:
            if ghost_path not in current_filepaths:
                self.db.remove_file(ghost_path)
                ghosts_removed += 1

        # SYNC FIX: Only process new or modified files, not the entire directory
        files_to_process = []
        for filepath in current_filepaths:
            if filepath not in memorized_filepaths:
                files_to_process.append(filepath)
            else:
                current_mtime = os.path.getmtime(filepath)
                saved_mtime = db_files.get(filepath, 0.0)
                if current_mtime > saved_mtime:
                    files_to_process.append(filepath)

        total = len(files_to_process)
        processed = 0

        if total == 0:
            self.finished_signal.emit(0, ghosts_removed)
            return

        for idx, filepath in enumerate(files_to_process):
            parsed_data = extract_text_from_file(filepath)
            if not parsed_data.get('error'):
                actual_folder_name = os.path.basename(os.path.dirname(filepath))
                self.db.add_file(
                    filename=parsed_data['filename'],
                    filepath=filepath,
                    text=parsed_data['text_content'],
                    mtime=os.path.getmtime(filepath),
                    preserve_structure=self.preserve_structure,
                    parent_folder=actual_folder_name
                )
                processed += 1
            prog = int(((idx + 1) / total) * 100)
            self.progress_signal.emit(prog, os.path.basename(filepath))

        self.finished_signal.emit(processed, ghosts_removed)


# ==========================================
# THREADING: AI Chat
# ==========================================
class AIWorker(QThread):
    finished_signal = Signal(str)
    def __init__(self, prompt, context_str):
        super().__init__()
        self.prompt = prompt
        self.context_str = context_str
    def run(self):
        reply = ask_local_ai(prompt=self.prompt, context_text=self.context_str)
        self.finished_signal.emit(reply)


# ==========================================
# MAIN WINDOW
# ==========================================
class FileSenseDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db = VectorDB() if BACKEND_READY else None
        self.old_pos = None

        self.setWindowTitle("FileSense AI")
        self.resize(1050, 700)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.setStyleSheet("""
            #MainFrame { background-color: #1A1A1D; border: 1px solid #333; border-radius: 12px; }
            #Sidebar { background-color: #111111; border-right: 1px solid #222; border-top-left-radius: 12px; border-bottom-left-radius: 12px; }
            QPushButton { background: transparent; border: none; color: #888; padding: 12px; text-align: left; font-size: 14px; border-radius: 5px; margin: 2px 10px; }
            QPushButton:hover { background-color: #222; color: white; }
            QPushButton#ActiveNav { color: white; font-weight: bold; background-color: #007ACC; }
            QLineEdit { background: #2D2D30; border: 1px solid #3E3E42; border-radius: 8px; padding: 10px; color: white; font-size: 14px; }
            QProgressBar { border: 1px solid #333; border-radius: 5px; text-align: center; color: white; background: #111; height: 20px; }
            QProgressBar::chunk { background-color: #007ACC; width: 10px; }
            QTreeView { background: #111; color: #EEE; border-radius: 5px; border: 1px solid #333; outline: 0;}
            QTreeView::item:selected { background: #007ACC; color: white; }
            QHeaderView::section { background-color: #222; color: white; border: none; padding: 5px; }
            QScrollArea { background: transparent; border: none; }
            QCheckBox { color: #888; padding: 5px; font-size: 13px; }
            QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #555; border-radius: 3px; background: #2D2D30; }
            QCheckBox::indicator:checked { background: #007ACC; border-color: #007ACC; }
        """)

        # Main UI Structure
        self.main_widget = QWidget()
        self.main_widget.setObjectName("MainFrame")
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- SIDEBAR ---
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(210)
        sidebar_layout = QVBoxLayout(self.sidebar)

        logo = QLabel("📂 FileSense")
        logo.setStyleSheet("font-size: 20px; font-weight: bold; color: white; margin: 20px 10px; padding-bottom: 10px;")

        # Backend status indicator (mirrors web's green/warning banner)
        if BACKEND_READY:
            status_label = QLabel("✅ Backend Active")
            status_label.setStyleSheet("color: #4CAF50; font-size: 11px; margin: 0px 10px 10px 10px;")
        else:
            status_label = QLabel("⚠️ Backend Offline")
            status_label.setStyleSheet("color: #E81123; font-size: 11px; margin: 0px 10px 10px 10px;")

        # 7 nav buttons — page order: 0=Search, 1=AI Editor, 2=Smart Clusters,
        #                              3=System Explorer, 4=Scan & Index, 5=Manage Files, 6=Insights
        self.btn_search   = QPushButton(" 🔍 Search Files")
        self.btn_editor   = QPushButton(" 🤖 AI Editor")
        self.btn_cluster  = QPushButton(" 🧠 Smart Clusters")
        self.btn_explorer = QPushButton(" 🗂️ System Explorer")
        self.btn_scan     = QPushButton(" 🚀 Scan & Index")
        self.btn_db       = QPushButton(" 🗄️ Manage Files")
        self.btn_insights = QPushButton(" 📊 Insights")

        self.btn_search.clicked.connect(lambda: self.switch_page(0))
        self.btn_editor.clicked.connect(lambda: self.switch_page(1))
        self.btn_cluster.clicked.connect(lambda: self.switch_page(2))
        self.btn_explorer.clicked.connect(lambda: self.switch_page(3))
        self.btn_scan.clicked.connect(lambda: self.switch_page(4))
        self.btn_db.clicked.connect(lambda: self.switch_page(5))
        self.btn_insights.clicked.connect(lambda: self.switch_page(6))

        sidebar_layout.addWidget(logo)
        sidebar_layout.addWidget(status_label)
        sidebar_layout.addWidget(self.btn_search)
        sidebar_layout.addWidget(self.btn_editor)
        sidebar_layout.addWidget(self.btn_cluster)
        sidebar_layout.addWidget(self.btn_explorer)
        sidebar_layout.addWidget(self.btn_scan)
        sidebar_layout.addWidget(self.btn_db)
        sidebar_layout.addWidget(self.btn_insights)
        sidebar_layout.addStretch()

        # --- CONTENT AREA ---
        self.pages = QStackedWidget()
        self.init_search_page()    # page 0
        self.init_editor_page()    # page 1
        self.init_cluster_page()   # page 2
        self.init_explorer_page()  # page 3
        self.init_scan_page()      # page 4
        self.init_db_page()        # page 5
        self.init_insights_page()  # page 6

        right_container = QVBoxLayout()
        right_container.addWidget(self.init_title_bar())
        right_container.addWidget(self.pages)

        self.main_layout.addWidget(self.sidebar)
        self.main_layout.addLayout(right_container)
        self.setCentralWidget(self.main_widget)

        self.init_tray()
        self.switch_page(0)

    def init_title_bar(self):
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.addStretch()

        min_btn = QPushButton("—")
        min_btn.setFixedSize(30, 30)
        min_btn.setStyleSheet("""
            QPushButton { background: transparent; color: #888; padding: 0px; margin: 0px; text-align: center; font-weight: bold; font-size: 14px; border-radius: 5px; }
            QPushButton:hover { background-color: #333; color: white; }
        """)
        min_btn.clicked.connect(self.showMinimized)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet("""
            QPushButton { background: transparent; color: #888; padding: 0px; margin: 0px; text-align: center; font-weight: bold; font-size: 14px; border-radius: 5px; }
            QPushButton:hover { background-color: #E81123; color: white; }
        """)
        close_btn.clicked.connect(self.hide)

        layout.addWidget(min_btn)
        layout.addWidget(close_btn)
        return bar

    # --- PAGE 0: SEARCH FILES ---
    def init_search_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(15)

        title = QLabel("🔍 Search Your Files")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("e.g. machine learning project ideas")
        self.search_input.returnPressed.connect(self.handle_search)

        btn_search = QPushButton("🔍 Search")
        btn_search.setStyleSheet("background: #007ACC; color: white; font-weight: bold; padding: 12px; margin: 0px;")
        btn_search.clicked.connect(self.handle_search)

        self.search_results = QTextBrowser()
        self.search_results.setStyleSheet("background: #111; color: #EEE; border-radius: 10px; padding: 10px;")
        self.search_results.setOpenExternalLinks(True)
        self.search_results.setHtml("<i style='color:#555;'>Enter a query above to search your indexed files.</i>")

        layout.addWidget(title)
        layout.addWidget(self.search_input)
        layout.addWidget(btn_search)
        layout.addWidget(self.search_results)
        self.pages.addWidget(page)

    # --- PAGE 1: AI EDITOR ---
    def init_editor_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("🤖 AI Editor")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        subtitle = QLabel(f"Powered by local {config.OLLAMA_MODEL if BACKEND_READY else 'AI'} engine.")
        subtitle.setStyleSheet("color: #555; font-size: 12px; margin-bottom: 5px;")

        self.chat_display = QTextBrowser()
        self.chat_display.setStyleSheet("background: transparent; border: none; color: #EEE;")
        self.chat_display.setOpenExternalLinks(True)
        self.chat_display.append("<b style='color: #007ACC;'>AI:</b> Hello! Ask me about your files.")

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Search or ask a question...")
        self.chat_input.returnPressed.connect(self.handle_chat)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.chat_display)
        layout.addWidget(self.chat_input)
        self.pages.addWidget(page)

    # --- PAGE 2: SMART CLUSTERS ---
    def init_cluster_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("🧠 Smart Clusters")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")

        self.cluster_display = QTextBrowser()
        self.cluster_display.setStyleSheet("background: #111; border-radius: 10px; color: #AAA; padding: 10px;")

        btn_run_cluster = QPushButton("🧠 Group Similar Files")
        btn_run_cluster.setStyleSheet("background: #007ACC; color: white; font-weight: bold; padding: 12px; margin: 0px;")
        btn_run_cluster.clicked.connect(self.run_clustering)

        layout.addWidget(title)
        layout.addWidget(self.cluster_display)
        layout.addWidget(btn_run_cluster)
        self.pages.addWidget(page)

    # --- PAGE 3: SYSTEM EXPLORER ---
    def init_explorer_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("🗂️ Local System Explorer")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: white; margin-bottom: 5px;")
        subtitle = QLabel("<i style='color:#888;'>Right-click any folder to send it to the AI Scanner.</i>")
        subtitle.setStyleSheet("margin-bottom: 10px;")

        self.file_model = QFileSystemModel()
        self.file_model.setRootPath("")

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setColumnWidth(0, 350)
        self.tree_view.hideColumn(1)
        self.tree_view.hideColumn(2)
        self.tree_view.hideColumn(3)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_tree_menu)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.tree_view)
        self.pages.addWidget(page)

    def open_tree_menu(self, position):
        index = self.tree_view.indexAt(position)
        if not index.isValid():
            return
        filepath = self.file_model.filePath(index)
        menu = QMenu()
        menu.setStyleSheet("""
            QMenu { background-color: #2D2D30; color: white; border: 1px solid #555; padding: 5px; }
            QMenu::item:selected { background-color: #007ACC; border-radius: 3px; }
        """)
        if os.path.isdir(filepath):
            scan_action = menu.addAction("🚀 Send Folder to AI Scanner")
            action = menu.exec(self.tree_view.viewport().mapToGlobal(position))
            if action == scan_action:
                self.path_input.setText(filepath)
                self.switch_page(4)

    # --- PAGE 4: SCAN & INDEX ---
    def init_scan_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(15)

        title = QLabel("🚀 Folder Parser & Vectorizer")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")

        self.path_input = QLineEdit()
        self.path_input.setText(str(config.DATA_DIR) if BACKEND_READY else "")

        btn_browse = QPushButton("📁 Select Folder to Scan")
        btn_browse.setStyleSheet("background: #333; color: white; margin: 0px;")
        btn_browse.clicked.connect(self.browse_folder)

        # Keep Folder Structure checkbox — matches web sidebar option
        self.preserve_structure_check = QCheckBox("📁 Keep Folder Structure (Skip AI Clustering)")
        self.preserve_structure_check.setStyleSheet("color: #888; padding: 5px; font-size: 13px;")

        self.btn_start_scan = QPushButton("🚀 Run Deep AI Indexing")
        self.btn_start_scan.setStyleSheet("background: #007ACC; color: white; font-weight: bold; padding: 15px; margin: 0px;")
        self.btn_start_scan.clicked.connect(self.run_scan)

        self.scan_progress = QProgressBar()
        self.scan_label = QLabel("Waiting for input...")
        self.scan_label.setStyleSheet("color: #888;")

        layout.addWidget(title)
        layout.addWidget(self.path_input)
        layout.addWidget(btn_browse)
        layout.addWidget(self.preserve_structure_check)
        layout.addWidget(self.btn_start_scan)
        layout.addWidget(self.scan_progress)
        layout.addWidget(self.scan_label)
        layout.addStretch()
        self.pages.addWidget(page)

    # --- PAGE 5: MANAGE FILES ---
    def init_db_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        title = QLabel("🗄️ Manage Files")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")

        # Scroll area for per-file list with Delete buttons
        self.file_list_widget = QWidget()
        self.file_list_widget.setStyleSheet("background: transparent;")
        self.file_list_layout = QVBoxLayout(self.file_list_widget)
        self.file_list_layout.setSpacing(4)
        self.file_list_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidget(self.file_list_widget)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: #111; border-radius: 10px; border: 1px solid #222;")

        btn_refresh = QPushButton("🔄 Refresh Database Status")
        btn_refresh.setStyleSheet("margin: 0px; background: #333; color: white;")
        btn_refresh.clicked.connect(self.refresh_db_stats)

        btn_wipe = QPushButton("🚨 Wipe AI Memory")
        btn_wipe.setStyleSheet("color: #E81123; margin: 0px; border: 1px solid #E81123;")
        btn_wipe.clicked.connect(self.wipe_db)

        layout.addWidget(title)
        layout.addWidget(scroll)
        layout.addWidget(btn_refresh)
        layout.addWidget(btn_wipe)
        self.pages.addWidget(page)

    # --- PAGE 6: INSIGHTS ---
    def init_insights_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        title = QLabel("📊 Database Insights")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")

        self.insights_display = QTextBrowser()
        self.insights_display.setStyleSheet("background: #111; color: #EEE; border-radius: 10px; padding: 15px;")
        self.insights_display.setHtml("<i style='color:#555;'>Click Refresh to load insights.</i>")

        btn_refresh_insights = QPushButton("🔄 Refresh Insights")
        btn_refresh_insights.setStyleSheet("background: #333; color: white; margin: 0px;")
        btn_refresh_insights.clicked.connect(self.refresh_insights)

        layout.addWidget(title)
        layout.addWidget(self.insights_display)
        layout.addWidget(btn_refresh_insights)
        self.pages.addWidget(page)

    # ==========================================
    # LOGIC FUNCTIONS
    # ==========================================
    def switch_page(self, index):
        self.pages.setCurrentIndex(index)
        btns = [self.btn_search, self.btn_editor, self.btn_cluster,
                self.btn_explorer, self.btn_scan, self.btn_db, self.btn_insights]
        for i, btn in enumerate(btns):
            btn.setObjectName("ActiveNav" if i == index else "")
        self.sidebar.style().unpolish(self.sidebar)
        self.sidebar.style().polish(self.sidebar)

    def handle_search(self):
        query = self.search_input.text().strip()
        if not query or not self.db:
            return
        self.search_results.setHtml("<i style='color:#888;'>Searching...</i>")
        res = self.db.search_documents(query, top_k=5)
        if "matches" in res and res["matches"]:
            html = f"<b style='color:#007ACC;'>Found {len(res['matches'])} matches for: \"{query}\"</b><br><br>"
            for m in res["matches"]:
                html += "<div style='background:#1E1E1E; padding:10px; border-radius:5px; margin-bottom:10px;'>"
                html += f"<b style='color:#EEE;'>📄 {m['filename']}</b><br>"
                html += f"<span style='color:#555; font-size:11px;'>{m['filepath']}</span><br>"
                html += f"<span style='color:#AAA;'>{m['snippet']}</span>"
                html += "</div>"
            self.search_results.setHtml(html)
        else:
            self.search_results.setHtml("<i style='color:#888;'>No matches found.</i>")

    def handle_chat(self):
        query = self.chat_input.text().strip()
        if not query: return
        self.chat_display.append(f"<b style='color: #4CAF50;'>You:</b> {query}")
        self.chat_input.clear()

        context = ""
        sources = []
        if self.db:
            res = self.db.search_documents(query, top_k=10)
            if "matches" in res:
                for m in res['matches']:
                    context += f"[From {m['filename']}]: {m['snippet']}\n\n"
                    sources.append(m)

        if sources:
            source_html = "<div style='color: #888; font-size: 12px; background: #222; padding: 8px; border-radius: 5px;'>"
            source_html += "<b>📂 Matching Files Found:</b><br>"
            for s in sources:
                source_html += f"📄 {s['filename']} <br><span style='font-size: 10px; color: #555;'>Path: {s['filepath']}</span><br>"
            source_html += "</div>"
            self.chat_display.append(source_html)

        self.ai_thread = AIWorker(query, context)
        self.ai_thread.finished_signal.connect(lambda r: self.chat_display.append(f"<b style='color: #007ACC;'>AI:</b> {r}<hr>"))
        self.ai_thread.start()

    def run_clustering(self):
        if not self.db: return
        self.cluster_display.setText("Processing semantic relationships...")
        clusters = self.db.cluster_files()
        if "error" in clusters:
            self.cluster_display.setText(clusters["error"])
        else:
            text = ""
            for cid, files in clusters.items():
                text += f"<b style='color: #007ACC;'>{cid}</b><br>"
                for f in files: text += f"&nbsp;&nbsp;• {f}<br>"
                text += "<br>"
            self.cluster_display.setHtml(text)

    def browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path: self.path_input.setText(path)

    def run_scan(self):
        folder = self.path_input.text()
        if not os.path.exists(folder): return
        self.btn_start_scan.setDisabled(True)
        preserve = self.preserve_structure_check.isChecked()
        self.scan_worker = ScanWorker(self.db, folder, preserve_structure=preserve)
        self.scan_worker.progress_signal.connect(self.update_scan_ui)
        self.scan_worker.finished_signal.connect(self.scan_done)
        self.scan_worker.start()

    def update_scan_ui(self, val, filename):
        self.scan_progress.setValue(val)
        self.scan_label.setText(f"Vectorizing: {filename}")

    def scan_done(self, count, ghosts_removed):
        self.btn_start_scan.setDisabled(False)
        msg = f"✅ Finished! Indexed {count} files"
        if ghosts_removed > 0:
            msg += f", removed {ghosts_removed} deleted files."
        else:
            msg += "."
        self.scan_label.setText(msg)
        self.refresh_db_stats()

    def refresh_db_stats(self):
        if not self.db:
            return
        files = self.db.get_file_metadata()

        # Clear existing rows
        while self.file_list_layout.count():
            item = self.file_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not files:
            empty = QLabel("No files indexed yet. Scan a folder first.")
            empty.setStyleSheet("color: #555; padding: 15px;")
            self.file_list_layout.addWidget(empty)
        else:
            for filepath in files.keys():
                row = QWidget()
                row.setStyleSheet("border-bottom: 1px solid #1E1E1E;")
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(12, 6, 12, 6)

                info = QLabel(f"<b style='color:#EEE;'>📄 {os.path.basename(filepath)}</b><br>"
                              f"<span style='color:#555; font-size:11px;'>{filepath}</span>")
                info.setTextFormat(Qt.RichText)
                info.setWordWrap(True)

                del_btn = QPushButton("Delete")
                del_btn.setFixedWidth(70)
                del_btn.setStyleSheet(
                    "background: transparent; color: #E81123; border: 1px solid #E81123; "
                    "padding: 4px; margin: 0px; border-radius: 4px;"
                )
                del_btn.clicked.connect(lambda checked, fp=filepath: self.delete_file(fp))

                row_layout.addWidget(info, 1)
                row_layout.addWidget(del_btn)
                self.file_list_layout.addWidget(row)

        self.file_list_layout.addStretch()

        # Keep insights in sync
        if hasattr(self, 'insights_display'):
            self.refresh_insights()

    def delete_file(self, filepath):
        if self.db:
            self.db.remove_file(filepath)
            self.refresh_db_stats()

    def refresh_insights(self):
        if not self.db:
            return
        files = self.db.get_file_metadata()
        if not files:
            self.insights_display.setHtml("<i style='color:#555;'>No data. Scan a folder first.</i>")
            return

        file_types = {}
        for fp in files.keys():
            ext = os.path.splitext(fp)[1].lower().replace(".", "").upper() or "UNKNOWN"
            file_types[ext] = file_types.get(ext, 0) + 1

        primary = max(file_types, key=file_types.get)
        max_count = max(file_types.values())

        html = (f"<b style='color:white; font-size:16px;'>Total Files: {len(files)}</b><br>"
                f"<span style='color:#888;'>Primary format: <b style='color:#007ACC;'>{primary}</b></span>"
                f"<br><br><b style='color:#888;'>Distribution by File Type:</b><br><br>")

        for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            bar_len = int((count / max_count) * 28)
            bar = "█" * bar_len
            html += (f"<span style='color:#007ACC; font-family:monospace;'>{ext:<8}</span>"
                     f"<span style='color:#007ACC;'>{bar}</span>"
                     f"<span style='color:white;'> {count}</span><br>")

        self.insights_display.setHtml(html)

    def wipe_db(self):
        if self.db:
            self.db.clear_database()
            self.refresh_db_stats()
            self.chat_display.append("<i style='color: #E81123;'>AI Memory Cleared.</i>")

    def init_tray(self):
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        menu = QMenu()
        q = menu.addAction("Exit FileSense")
        q.triggered.connect(QApplication.instance().quit)
        self.tray.setContextMenu(menu)
        self.tray.show()
        keyboard.add_hotkey('ctrl+shift+space', self.toggle_window)

    def toggle_window(self):
        if self.isVisible(): self.hide()
        else: self.showNormal(); self.activateWindow()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: self.old_pos = e.globalPosition().toPoint()
    def mouseMoveEvent(self, e):
        if self.old_pos:
            delta = e.globalPosition().toPoint() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = e.globalPosition().toPoint()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    win = FileSenseDesktop()
    win.show()
    sys.exit(app.exec())
