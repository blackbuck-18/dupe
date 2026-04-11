import sys
import os
import keyboard
import time
import re
from PySide6.QtWidgets import (QApplication, QMainWindow, QSystemTrayIcon, QMenu, 
                               QVBoxLayout, QHBoxLayout, QWidget, QStyle, QLineEdit, 
                               QTextBrowser, QPushButton, QLabel, QStackedWidget, 
                               QFileDialog, QProgressBar, QFileSystemModel, QTreeView, QHeaderView)
from PySide6.QtGui import QAction, QFont, QTextCursor, QIcon, QColor
from PySide6.QtCore import QObject, Signal, Qt, QThread, QPoint

# --- BACKEND IMPORTS ---
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
# THREADING: Background Scanner
# ==========================================
class ScanWorker(QThread):
    progress_signal = Signal(int, str)
    finished_signal = Signal(int)

    def __init__(self, db, folder):
        super().__init__()
        self.db = db
        self.folder = folder

    def run(self):
        valid_exts = ['.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg', '.mp3', '.mp4', '.wav']
        files_to_process = []
        for root, dirs, files in os.walk(self.folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    files_to_process.append(os.path.join(root, file))
        
        total = len(files_to_process)
        processed = 0
        for idx, filepath in enumerate(files_to_process):
            parsed_data = extract_text_from_file(filepath)
            if not parsed_data.get('error'):
                self.db.add_file(
                    filename=parsed_data['filename'],
                    filepath=filepath,
                    text=parsed_data['text_content'],
                    mtime=os.path.getmtime(filepath)
                )
                processed += 1
            prog = int(((idx + 1) / total) * 100)
            self.progress_signal.emit(prog, os.path.basename(filepath))
        self.finished_signal.emit(processed)

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
        """)

        # Main UI Structure
        self.main_widget = QWidget()
        self.main_widget.setObjectName("MainFrame")
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 1. Sidebar
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(210)
        sidebar_layout = QVBoxLayout(self.sidebar)
        
        logo = QLabel("📂 FileSense")
        logo.setStyleSheet("font-size: 20px; font-weight: bold; color: white; margin: 20px 10px; padding-bottom: 10px;")
        
        self.btn_chat = QPushButton(" 💬 AI Chat")
        self.btn_cluster = QPushButton(" 🧠 Smart Clusters")
        self.btn_explorer = QPushButton(" 🗂️ System Explorer") # <-- NEW BUTTON
        self.btn_scan = QPushButton(" 🚀 Scan & Index")
        self.btn_db = QPushButton(" 🗄️ Manage Data")
        
        self.btn_chat.clicked.connect(lambda: self.switch_page(0))
        self.btn_cluster.clicked.connect(lambda: self.switch_page(1))
        self.btn_explorer.clicked.connect(lambda: self.switch_page(2)) # <-- NEW LINK
        self.btn_scan.clicked.connect(lambda: self.switch_page(3))
        self.btn_db.clicked.connect(lambda: self.switch_page(4))
        
        sidebar_layout.addWidget(logo)
        sidebar_layout.addWidget(self.btn_chat)
        sidebar_layout.addWidget(self.btn_cluster)
        sidebar_layout.addWidget(self.btn_explorer)
        sidebar_layout.addWidget(self.btn_scan)
        sidebar_layout.addWidget(self.btn_db)
        sidebar_layout.addStretch()
        
        # 2. Content Area
        self.pages = QStackedWidget()
        self.init_chat_page()
        self.init_cluster_page()
        self.init_explorer_page() # <-- NEW PAGE RENDERED
        self.init_scan_page()
        self.init_db_page()
        
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

    # --- PAGE 1: CHAT ---
    def init_chat_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.chat_display = QTextBrowser()
        self.chat_display.setStyleSheet("background: transparent; border: none; color: #EEE;")
        self.chat_display.setOpenExternalLinks(True)
        self.chat_display.append("<b style='color: #007ACC;'>AI:</b> Hello! Ask me about your files.")
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Search or ask a question...")
        self.chat_input.returnPressed.connect(self.handle_chat)
        
        layout.addWidget(self.chat_display)
        layout.addWidget(self.chat_input)
        self.pages.addWidget(page)

    # --- PAGE 2: CLUSTERS ---
    def init_cluster_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.cluster_display = QTextBrowser()
        self.cluster_display.setStyleSheet("background: #111; border-radius: 10px; color: #AAA; padding: 10px;")
        btn_run_cluster = QPushButton("🧠 Generate Smart Clusters")
        btn_run_cluster.setStyleSheet("background: #007ACC; color: white; font-weight: bold; padding: 12px; margin: 0px;")
        btn_run_cluster.clicked.connect(self.run_clustering)
        
        layout.addWidget(QLabel("🧠 AI Semantic Grouping"))
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
        
        # 1. Create the File System Model (Points to Root / All Drives)
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath("") 
        
        # 2. Create the Tree View
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        
        # Clean up the columns (Hide size, type, date modified for cleaner look)
        self.tree_view.setColumnWidth(0, 350)
        self.tree_view.hideColumn(1)
        self.tree_view.hideColumn(2)
        self.tree_view.hideColumn(3)
        
        # 3. Enable Right-Click Menus
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_tree_menu)
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.tree_view)
        self.pages.addWidget(page)

    def open_tree_menu(self, position):
        """Creates the right-click menu for the file explorer."""
        index = self.tree_view.indexAt(position)
        if not index.isValid():
            return
            
        filepath = self.file_model.filePath(index)
        
        menu = QMenu()
        menu.setStyleSheet("""
            QMenu { background-color: #2D2D30; color: white; border: 1px solid #555; padding: 5px; } 
            QMenu::item:selected { background-color: #007ACC; border-radius: 3px; }
        """)
        
        # Only allow scanning if they right-clicked a directory (not a single file)
        if os.path.isdir(filepath):
            scan_action = menu.addAction("🚀 Send Folder to AI Scanner")
            action = menu.exec(self.tree_view.viewport().mapToGlobal(position))
            
            if action == scan_action:
                self.path_input.setText(filepath) # Pre-fill the scan tab
                self.switch_page(3)               # Jump to the scan tab

    # --- PAGE 4: SCAN ---
    def init_scan_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(15)
        
        self.path_input = QLineEdit()
        self.path_input.setText(str(config.DATA_DIR))
        btn_browse = QPushButton("📁 Select Folder to Scan")
        btn_browse.setStyleSheet("background: #333; color: white; margin: 0px;")
        btn_browse.clicked.connect(self.browse_folder)
        
        self.btn_start_scan = QPushButton("🚀 Run Deep AI Indexing")
        self.btn_start_scan.setStyleSheet("background: #007ACC; color: white; font-weight: bold; padding: 15px; margin: 0px;")
        self.btn_start_scan.clicked.connect(self.run_scan)
        
        self.scan_progress = QProgressBar()
        self.scan_label = QLabel("Waiting for input...")
        
        layout.addWidget(QLabel("🚀 Folder Parser & Vectorizer"))
        layout.addWidget(self.path_input)
        layout.addWidget(btn_browse)
        layout.addWidget(self.btn_start_scan)
        layout.addWidget(self.scan_progress)
        layout.addWidget(self.scan_label)
        layout.addStretch()
        self.pages.addWidget(page)

    # --- PAGE 5: MANAGE DATA ---
    def init_db_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.db_table = QTextBrowser()
        self.db_table.setStyleSheet("background: #111; color: #888; border-radius: 10px;")
        
        btn_refresh = QPushButton("🔄 Refresh Database Status")
        btn_refresh.setStyleSheet("margin: 0px; background: #333;")
        btn_refresh.clicked.connect(self.refresh_db_stats)
        
        btn_wipe = QPushButton("🚨 Wipe AI Memory")
        btn_wipe.setStyleSheet("color: #E81123; margin: 0px; border: 1px solid #E81123;")
        btn_wipe.clicked.connect(self.wipe_db)
        
        layout.addWidget(QLabel("🗄️ Knowledge Base Management"))
        layout.addWidget(self.db_table)
        layout.addWidget(btn_refresh)
        layout.addWidget(btn_wipe)
        self.pages.addWidget(page)

    # ==========================================
    # LOGIC FUNCTIONS
    # ==========================================
    def switch_page(self, index):
        self.pages.setCurrentIndex(index)
        # Update Nav Styles for all 5 buttons
        btns = [self.btn_chat, self.btn_cluster, self.btn_explorer, self.btn_scan, self.btn_db]
        for i, btn in enumerate(btns):
            btn.setObjectName("ActiveNav" if i == index else "")
        self.sidebar.style().unpolish(self.sidebar)
        self.sidebar.style().polish(self.sidebar)

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
        self.scan_worker = ScanWorker(self.db, folder)
        self.scan_worker.progress_signal.connect(self.update_scan_ui)
        self.scan_worker.finished_signal.connect(self.scan_done)
        self.scan_worker.start()

    def update_scan_ui(self, val, filename):
        self.scan_progress.setValue(val)
        self.scan_label.setText(f"AI Indexing: {filename}")

    def scan_done(self, count):
        self.btn_start_scan.setDisabled(False)
        self.scan_label.setText(f"✅ Finished! Indexed {count} files.")
        self.refresh_db_stats()

    def refresh_db_stats(self):
        if self.db:
            files = self.db.get_file_metadata()
            text = f"<b>Total Files Indexed: {len(files)}</b><br><br>"
            for f, mtime in files.items():
                text += f"📄 {os.path.basename(f)}<br><span style='font-size: 10px; color: #555;'>{f}</span><br><br>"
            self.db_table.setHtml(text)

    def wipe_db(self):
        if self.db:
            self.db.clear_database()
            self.refresh_db_stats()
            self.chat_display.append("<i style='color: red;'>AI Memory Cleared.</i>")

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
