import os
import numpy as np
import pytest
import tempfile
from unittest.mock import patch, MagicMock
import json

# --- IMPORT YOUR BACKEND MODULES ---
from backend.parser import extract_text_from_file
from backend.vector_engine import VectorDB
from backend.ollama_bridge import ask_local_ai
import config

# ==========================================
# 1. PARSER: DEEP TESTING
# ==========================================
class TestParserDeep:
    def test_file_not_found(self):
        result = extract_text_from_file("non_existent_file_12345.pdf")
        assert result['error'] == "File not found or inaccessible"

    def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = extract_text_from_file(tmp_path)
            assert result['error'] == "Unsupported format"
            assert result['metadata']['extension'] == '.jpg'
        finally:
            os.remove(tmp_path)

    @patch('os.path.getsize')
    def test_file_size_exceeds_limit(self, mock_getsize):
        mock_getsize.return_value = 105 * 1024 * 1024  # 105 MB
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = extract_text_from_file(tmp_path)
            assert result['error'] == "File too large"
            assert result['metadata']['file_size'] == 105 * 1024 * 1024
        finally:
            os.remove(tmp_path)

    def test_valid_txt_parsing(self):
        content = "This is a test document.\nIt has multiple lines."
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w', encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = extract_text_from_file(tmp_path)
            assert result['error'] is None
            assert result['text_content'] == content
            assert result['metadata']['extension'] == '.txt'
            assert result['filename'] == os.path.basename(tmp_path)
        finally:
            os.remove(tmp_path)

    def test_txt_parsing_encoding_fallback(self):
        # Write bytes that aren't valid UTF-8
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='wb') as tmp:
            tmp.write(b"Good text \xff\xfe Bad text")
            tmp_path = tmp.name
        try:
            result = extract_text_from_file(tmp_path)
            assert result['error'] is None
            # The 'ignore' flag should drop the bad bytes without crashing
            assert "Good text" in result['text_content']
        finally:
            os.remove(tmp_path)

    @patch('backend.parser.fitz.open')
    def test_pdf_parsing_success(self, mock_fitz_open):
        # Mock PyMuPDF behavior
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page 1 Text"
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = extract_text_from_file(tmp_path)
            assert result['error'] is None
            assert result['text_content'] == "Page 1 Text"
            mock_doc.close.assert_called_once()
        finally:
            os.remove(tmp_path)

    @patch('backend.parser.fitz.open')
    def test_pdf_parsing_corrupt(self, mock_fitz_open):
        import fitz
        mock_fitz_open.side_effect = fitz.FileDataError("Corrupted file")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = extract_text_from_file(tmp_path)
            assert result['error'] == "Corrupt PDF"
        finally:
            os.remove(tmp_path)


# ==========================================
# 2. OLLAMA BRIDGE: DEEP TESTING
# ==========================================
class TestOllamaBridgeDeep:
    @patch('backend.ollama_bridge.requests.post')
    def test_successful_inference(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "I am an AI."}
        mock_post.return_value = mock_resp

        result = ask_local_ai("Who are you?")
        assert result == "I am an AI."

    @patch('backend.ollama_bridge.requests.post')
    def test_context_truncation(self, mock_post):
        # Create a massive context string
        massive_context = "word " * (config.MAX_CONTEXT_WORDS + 500)
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Truncated"}
        mock_post.return_value = mock_resp

        ask_local_ai("Query", context_text=massive_context)
        
        # Verify the payload sent to Ollama was truncated
        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        # The prompt should be roughly the length of the system prompt + MAX_CONTEXT_WORDS + "..."
        word_count = len(payload['prompt'].split())
        assert word_count <= config.MAX_CONTEXT_WORDS + 50 # 50 words for system prompt padding

    @patch('backend.ollama_bridge.requests.post')
    def test_model_not_found(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_post.return_value = mock_resp

        result = ask_local_ai("Query")
        assert "AI Model Not Found!" in result

    @patch('backend.ollama_bridge.requests.post')
    def test_timeout(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Timed out")
        
        result = ask_local_ai("Query")
        assert "Timeout" in result

    @patch('backend.ollama_bridge.requests.post')
    def test_connection_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Refused")
        
        result = ask_local_ai("Query")
        assert "Connection Failed" in result


# ==========================================
# 3. VECTOR ENGINE: DEEP TESTING
# ==========================================
class TestVectorEngineDeep:
    @pytest.fixture(autouse=True)
    def setup_mock_db(self):
        """Sets up a fully mocked ChromaDB and SentenceTransformer."""
        with patch('backend.vector_engine.chromadb.PersistentClient') as mock_client:
            self.mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = self.mock_collection
            
            with patch('backend.vector_engine.SentenceTransformer') as mock_model:
                # Return a dummy vector
                self.mock_encoder = mock_model.return_value
                self.mock_encoder.encode.return_value = np.array([0.1, 0.2])
                
                self.db = VectorDB()
                yield

    def test_add_file_success(self):
        result = self.db.add_file("test.txt", "/path/test.txt", "Text", 10.0)
        assert result is True
        self.mock_collection.upsert.assert_called_once()
        args, kwargs = self.mock_collection.upsert.call_args
        assert kwargs['ids'] == ["/path/test.txt"]
        assert kwargs['metadatas'] == [{"filename": "test.txt", "filepath": "/path/test.txt", "mtime": 10.0}]

    def test_add_file_empty_text(self):
        result = self.db.add_file("test.txt", "/path/test.txt", "   ", 10.0)
        assert result is False
        self.mock_collection.upsert.assert_not_called()

    def test_remove_file(self):
        result = self.db.remove_file("/path/test.txt")
        assert result is True
        self.mock_collection.delete.assert_called_once_with(ids=["/path/test.txt"])

    def test_get_file_metadata(self):
        # Mock ChromaDB returning metadata
        self.mock_collection.get.return_value = {
            'metadatas': [
                {'filepath': '/doc1.pdf', 'mtime': 100.0},
                {'filepath': '/doc2.txt', 'mtime': 200.0}
            ]
        }
        
        result = self.db.get_file_metadata()
        assert len(result) == 2
        assert result['/doc1.pdf'] == 100.0
        assert result['/doc2.txt'] == 200.0

    def test_clear_database(self):
        result = self.db.clear_database()
        assert result is True
        # Verify it deletes and recreates
        self.db.chroma_client.delete_collection.assert_called_with("filesense_docs")
        self.db.chroma_client.get_or_create_collection.assert_called()

    # --- CLUSTERING TESTS ---
    def test_cluster_not_enough_files(self):
        # Mock DB returning only 1 file (default min_cluster_size is 2)
        self.mock_collection.get.return_value = {
            'embeddings': [[0.1, 0.2]],
            'metadatas': [{'filename': 'one.txt'}]
        }
        result = self.db.cluster_files()
        assert 'warning' in result

    @patch('backend.vector_engine.HDBSCAN')
    def test_cluster_success(self, mock_hdbscan):
        # Mock DB returning 3 files
        self.mock_collection.get.return_value = {
            'embeddings': [[0.1, 0.1], [0.12, 0.12], [0.9, 0.9]],
            'metadatas': [{'filename': 'a.txt'}, {'filename': 'b.txt'}, {'filename': 'outlier.txt'}]
        }
        
        # Mock HDBSCAN assigning a, b to Cluster 0, and outlier to -1 (Noise)
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = [0, 0, -1]
        mock_hdbscan.return_value = mock_instance

        result = self.db.cluster_files()
        assert "Cluster 0" in result
        assert "Uncategorized / Noise" in result
        assert "a.txt" in result["Cluster 0"]
        assert "outlier.txt" in result["Uncategorized / Noise"]

    # --- SEARCH TESTS ---
    def test_search_empty_query(self):
        result = self.db.search_documents("   ")
        assert "error" in result

    def test_hybrid_search_lexical_match(self):
        # Test the direct keyword substring match
        self.mock_collection.get.return_value = {
            'documents': ["This is about python programming."],
            'metadatas': [{'filename': 'code.txt', 'filepath': '/code.txt'}]
        }
        # Mock the vector search to return nothing relevant to isolate lexical
        self.mock_collection.query.return_value = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        result = self.db.search_documents("python")
        assert "matches" in result
        assert len(result["matches"]) == 1
        assert result["matches"][0]["distance"] == "Keyword Match"

    def test_hybrid_search_fuzzy_match(self):
        # Test the fuzzy spelling fallback
        self.mock_collection.get.return_value = {
            'documents': ["This is about python programming."],
            'metadatas': [{'filename': 'code.txt', 'filepath': '/code.txt'}]
        }
        self.mock_collection.query.return_value = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        # "pyhton" is misspelled
        result = self.db.search_documents("pyhton")
        assert "matches" in result
        assert len(result["matches"]) == 1
        assert result["matches"][0]["distance"] == "Keyword Match"

    def test_hybrid_search_semantic_match(self):
        # Test the vector search fallback
        self.mock_collection.get.return_value = {'documents': [], 'metadatas': []} # No lexical matches
        
        self.mock_collection.query.return_value = {
            'documents': [["A document about ML."]],
            'metadatas': [[{'filename': 'ml.txt', 'filepath': '/ml.txt'}]],
            'distances': [[0.5]] # Good distance score
        }

        result = self.db.search_documents("machine learning", distance_threshold=1.0)
        assert "matches" in result
        assert len(result["matches"]) == 1
        assert result["matches"][0]["score"] == 0.5