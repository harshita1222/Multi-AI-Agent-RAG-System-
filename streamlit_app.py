# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #667eea;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #718096;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .result-container {
        background: #f7fafc;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .chunk-preview {
        background: #edf2f7;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        border: 1px solid #e2e8f0;
    }
    
    .confidence-high { color: #38a169; font-weight: bold; }
    .confidence-medium { color: #d69e2e; font-weight: bold; }
    .confidence-low { color: #e53e3e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class StreamlitRAGSystem:
    def __init__(self):
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'chunks' not in st.session_state:
            st.session_state.chunks = []
        if 'vectorizer' not in st.session_state:
            st.session_state.vectorizer = None
        if 'doc_vectors' not in st.session_state:
            st.session_state.doc_vectors = None
        if 'processed' not in st.session_state:
            st.session_state.processed = False
    
    def read_file_content(self, uploaded_file):
        """Extract text content from uploaded file"""
        try:
            if uploaded_file.type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "application/pdf":
                # Simple PDF text extraction (you might want to use PyPDF2 or pdfplumber)
                return f"PDF content from {uploaded_file.name} (PDF parsing not implemented in this demo)"
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # Simple DOCX handling (you might want to use python-docx)
                return f"DOCX content from {uploaded_file.name} (DOCX parsing not implemented in this demo)"
            else:
                return f"Unsupported file type: {uploaded_file.type}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_documents(self, uploaded_files, chunk_size, overlap):
        """Process uploaded documents and create chunks"""
        st.session_state.documents = []
        st.session_state.chunks = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Read file content
            content = self.read_file_content(uploaded_file)
            
            # Store document
            doc = {
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'content': content
            }
            st.session_state.documents.append(doc)
            
            # Create chunks
            file_chunks = self.chunk_text(content, chunk_size, overlap)
            
            for j, chunk in enumerate(file_chunks):
                chunk_data = {
                    'id': f"{i}_{j}",
                    'text': chunk,
                    'source': uploaded_file.name,
                    'doc_index': i,
                    'chunk_index': j
                }
                st.session_state.chunks.append(chunk_data)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Create TF-IDF vectors
        status_text.text("Creating vector embeddings...")
        chunk_texts = [chunk['text'] for chunk in st.session_state.chunks]
        
        st.session_state.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        st.session_state.doc_vectors = st.session_state.vectorizer.fit_transform(chunk_texts)
        st.session_state.processed = True
        
        status_text.text("✅ Processing complete!")
        progress_bar.progress(1.0)
        
        return len(st.session_state.documents), len(st.session_state.chunks)
    
    def search_documents(self, query, top_k=3):
        """Search documents using vector similarity"""
        if not st.session_state.processed:
            return []
        
        # Transform query using the same vectorizer
        query_vector = st.session_state.vectorizer.transform([query])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(query_vector, st.session_state.doc_vectors)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarity_scores[idx] > 0:  # Only include relevant results
                chunk = st.session_state.chunks[idx]
                result = {
                    'chunk': chunk,
                    'score': similarity_scores[idx],
                    'text': chunk['text'],
                    'source': chunk['source']
                }
                results.append(result)
        
        return results
    
    def generate_answer(self, query, results):
        """Generate answer from search results"""
        if not results:
            return "I couldn't find relevant information in the documents to answer your question."
        
        # Simple answer generation based on most relevant chunk
        top_result = results[0]
        context = top_result['text']
        
        # Extract most relevant sentence
        sentences = re.split(r'[.!?]+', context)
        query_words = set(query.lower().split())
        
        best_sentence = ""
        max_overlap = 0
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sentence.strip()
        
        if best_sentence and max_overlap > 0:
            return best_sentence + "."
        else:
            # Return first sentence of most relevant chunk
            first_sentence = sentences[0].strip()
            return first_sentence + "." if first_sentence else context[:200] + "..."

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Document Q&A System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload documents and ask intelligent questions using Retrieval-Augmented Generation</p>', unsafe_allow_html=True)
    
    # Initialize RAG system
    rag = StreamlitRAGSystem()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=100,
            max_value=2000,
            value=500,
            help="Size of text chunks for processing"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=500,
            value=50,
            help="Overlap between adjacent chunks"
        )
        
        top_k = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of most relevant chunks to retrieve"
        )
        
        st.divider()
        
        # Display statistics
        if st.session_state.processed:
            st.markdown("### 📊 Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", len(st.session_state.documents))
            with col2:
                st.metric("Chunks", len(st.session_state.chunks))
            
            # Document list
            st.markdown("### 📄 Uploaded Documents")
            for i, doc in enumerate(st.session_state.documents):
                file_size = f"{doc['size'] / 1024:.1f} KB" if doc['size'] < 1024*1024 else f"{doc['size'] / (1024*1024):.1f} MB"
                st.text(f"📄 {doc['name']} ({file_size})")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Processing")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx'],
            help="Supported formats: TXT, PDF, DOCX"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) selected")
            
            # Process documents button
            if st.button("🔄 Process Documents", type="primary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    num_docs, num_chunks = rag.process_documents(uploaded_files, chunk_size, chunk_overlap)
                
                st.success(f"✅ Processed {num_docs} documents into {num_chunks} chunks!")
                st.rerun()
    
    with col2:
        st.header("❓ Query & Results")
        
        # Query input
        query = st.text_input(
            "Ask a question about your documents:",
            placeholder="What is the main topic discussed?",
            disabled=not st.session_state.processed
        )
        
        # Search button
        search_button = st.button(
            "🔍 Search",
            type="primary",
            disabled=not st.session_state.processed or not query,
            use_container_width=True
        )
        
        if search_button and query:
            with st.spinner("Searching documents..."):
                results = rag.search_documents(query, top_k)
            
            if results:
                # Generate answer
                answer = rag.generate_answer(query, results)
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(f'<div class="result-container">{answer}</div>', unsafe_allow_html=True)
                
                # Display search results
                st.markdown("### 🔍 Search Results")
                
                for i, result in enumerate(results):
                    score = result['score']
                    confidence_class = "confidence-high" if score > 0.3 else "confidence-medium" if score > 0.1 else "confidence-low"
                    
                    with st.expander(f"📄 {result['source']} (Relevance: {score:.1%})"):
                        st.markdown(f'<span class="{confidence_class}">Confidence: {score:.1%}</span>', unsafe_allow_html=True)
                        st.markdown(f'<div class="chunk-preview">{result["text"]}</div>', unsafe_allow_html=True)
                
                # Metadata
                st.markdown("### 📊 Search Metadata")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Results Found", len(results))
                with col2:
                    st.metric("Best Match", f"{results[0]['score']:.1%}")
                with col3:
                    unique_sources = len(set(r['source'] for r in results))
                    st.metric("Sources", unique_sources)
            
            else:
                st.warning("No relevant results found. Try rephrasing your question.")
    
    # Instructions
    if not st.session_state.processed:
        st.markdown("---")
        st.markdown("### 🚀 How to Use")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1. Upload Documents**
            - Click "Browse files" or drag & drop
            - Supports TXT, PDF, DOCX files
            - Multiple files allowed
            """)
        
        with col2:
            st.markdown("""
            **2. Configure Settings**
            - Adjust chunk size (default: 500)
            - Set overlap (default: 50)
            - Choose top-K results (default: 3)
            """)
        
        with col3:
            st.markdown("""
            **3. Ask Questions**
            - Process documents first
            - Type your question
            - Get AI-generated answers
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | [View Source Code](https://github.com/yourusername/rag-document-qa)")

if __name__ == "__main__":
    main()
