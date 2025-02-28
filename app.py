# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://cdp-frontend.vercel.app"}})

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize model and index storage
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CDP_DOCS = {
    'segment': {
        'base_url': 'https://segment.com/docs/',
        'start_urls': ['https://segment.com/docs/?ref=nav'],
        'allowed_domains': ['segment.com']
    },
    'mparticle': {
        'base_url': 'https://docs.mparticle.com/',
        'start_urls': ['https://docs.mparticle.com/'],
        'allowed_domains': ['docs.mparticle.com']
    },
    'lytics': {
        'base_url': 'https://docs.lytics.com/',
        'start_urls': ['https://docs.lytics.com/'],
        'allowed_domains': ['docs.lytics.com']
    },
    'zeotap': {
        'base_url': 'https://docs.zeotap.com/',
        'start_urls': ['https://docs.zeotap.com/home/en-us/'],
        'allowed_domains': ['docs.zeotap.com']
    }
}

class DocScraper:
    def __init__(self, config):
        self.base_url = config['base_url']
        self.start_urls = config['start_urls']
        self.allowed_domains = config['allowed_domains']
        self.visited_urls = set()
        self.url_queue = list(self.start_urls)
        self.documents = []
        self.max_pages = 50  # Limit to prevent extensive crawling

    def is_allowed_url(self, url):
        parsed_url = urlparse(url)
        return any(domain in parsed_url.netloc for domain in self.allowed_domains)

    def extract_text_from_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get main content area - this might need customization per site
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup
        
        # Extract headings and paragraphs
        headings = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        paragraphs = main_content.find_all('p')
        list_items = main_content.find_all('li')
        code_blocks = main_content.find_all(['pre', 'code'])
        
        # Combine relevant content
        content_parts = []
        
        for heading in headings:
            content_parts.append(f"## {heading.get_text().strip()}")
        
        for para in paragraphs:
            text = para.get_text().strip()
            if text:
                content_parts.append(text)
        
        for item in list_items:
            text = item.get_text().strip()
            if text:
                content_parts.append(f"- {text}")
        
        for code in code_blocks:
            text = code.get_text().strip()
            if text:
                content_parts.append(f"```\n{text}\n```")
        
        return "\n\n".join(content_parts)

    def extract_links(self, html_content, current_url):
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urljoin(current_url, href)
            
            # Skip non-http links, already visited links, and external links
            if (absolute_url.startswith('http') and 
                absolute_url not in self.visited_urls and 
                self.is_allowed_url(absolute_url)):
                links.append(absolute_url)
        
        return links

    def scrape(self):
        count = 0
        
        while self.url_queue and count < self.max_pages:
            current_url = self.url_queue.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            self.visited_urls.add(current_url)
            logger.info(f"Scraping {current_url}")
            
            try:
                response = requests.get(current_url, timeout=10)
                if response.status_code == 200:
                    html_content = response.text
                    
                    # Extract text content
                    text = self.extract_text_from_html(html_content)
                    if text:
                        self.documents.append({
                            'url': current_url,
                            'text': text
                        })
                    
                    # Extract and add new links to queue
                    new_links = self.extract_links(html_content, current_url)
                    for link in new_links:
                        if link not in self.visited_urls and link not in self.url_queue:
                            self.url_queue.append(link)
                    
                    count += 1
                    
                    # Small delay to be respectful to the server
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error scraping {current_url}: {str(e)}")
        
        return self.documents

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def chunk_document(self, document, chunk_size=300, overlap=50):
        """Split document into chunks with some overlap"""
        text = document['text']
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = ' '.join(words[i:i + chunk_size])
            if len(chunk_text.split()) > 20:  # Only keep substantial chunks
                chunks.append({
                    'url': document['url'],
                    'text': chunk_text,
                    'chunk_id': f"{document['url']}_{i}"
                })
                
        return chunks
    
    def create_embeddings(self, chunks):
        """Create vector embeddings for document chunks"""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts)
        
        return embeddings, chunks
    
    def process_documents(self, documents):
        """Process all documents into chunks and embeddings"""
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        embeddings, chunks = self.create_embeddings(all_chunks)
        
        return embeddings, chunks

class DocumentIndex:
    def __init__(self, cdp_name):
        self.cdp_name = cdp_name
        self.processor = DocumentProcessor()
        self.index_path = os.path.join(DATA_DIR, f"{cdp_name}_index.faiss")
        self.chunks_path = os.path.join(DATA_DIR, f"{cdp_name}_chunks.pkl")
        
    def build_index(self, documents):
        """Build FAISS index from documents"""
        embeddings, chunks = self.processor.process_documents(documents)
        
        # Create and save FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        faiss.write_index(index, self.index_path)
        
        # Save chunks
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
            
        return index, chunks
    
    def load_index(self):
        """Load existing FAISS index and chunks"""
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            index = faiss.read_index(self.index_path)
            
            with open(self.chunks_path, 'rb') as f:
                chunks = pickle.load(f)
                
            return index, chunks
        
        return None, None
    
    def search(self, query, k=5):
        """Search for relevant chunks"""
        index, chunks = self.load_index()
        
        if index is None or chunks is None:
            return []
        
        # Encode query
        query_vector = self.processor.model.encode([query])
        
        # Search
        distances, indices = index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx < len(chunks):
                results.append(chunks[idx])
                
        return results

def format_response(chunks, query):
    """Format a response from relevant document chunks"""
    if not chunks:
        return "I don't have information about that in my knowledge base. Please try asking about a different topic or rephrasing your question."
    
    # Extract and join text from most relevant chunks
    combined_text = "\n".join([chunk['text'] for chunk in chunks[:3]])
    
    # Check if this is a "how-to" question
    is_how_to = any(phrase in query.lower() for phrase in ["how to", "how do i", "how can i", "steps to", "guide for"])
    
    if is_how_to:
        # Extract step-by-step instructions if available
        steps = []
        step_pattern = re.compile(r'(?:^|\n)(?:\d+\.|[\-\*])\s+(.*?)(?=(?:\n(?:\d+\.|[\-\*]))|$)', re.DOTALL)
        step_matches = step_pattern.findall(combined_text)
        
        if step_matches:
            for i, step in enumerate(step_matches):
                steps.append(f"{i+1}. {step.strip()}")
            
            response = (
                "Here's how you can do that:\n\n" + 
                "\n".join(steps) + "\n\n" + 
                f"Source: {chunks[0]['url']}"
            )
        else:
            # If no clear steps found, provide a summary
            response = (
                "Based on the documentation, here's what I found:\n\n" +
                combined_text[:800] + "...\n\n" +
                f"For more details, check: {chunks[0]['url']}"
            )
    else:
        # For non-how-to questions, provide a direct answer
        response = (
            "Here's what I found about that:\n\n" +
            combined_text[:800] + "...\n\n" +
            f"Source: {chunks[0]['url']}"
        )
    
    return response

def compare_cdps(query, cdp1, cdp2):
    """Compare functionality between two CDPs"""
    index1 = DocumentIndex(cdp1)
    index2 = DocumentIndex(cdp2)
    
    # Extract the feature to compare
    feature_pattern = re.compile(r'(?:how does|compare|difference between).*?([\w\s]+)')
    feature_match = feature_pattern.search(query.lower())
    
    if feature_match:
        feature = feature_match.group(1).strip()
        # Create specific queries for each platform
        specific_query = f"how to {feature}"
    else:
        specific_query = query
    
    # Get relevant chunks from both CDPs
    chunks1 = index1.search(specific_query, k=3)
    chunks2 = index2.search(specific_query, k=3)
    
    if not chunks1 or not chunks2:
        return f"I don't have enough information to compare {cdp1} and {cdp2} on this topic."
    
    # Format comparison response
    response = f"# Comparing {cdp1.title()} vs {cdp2.title()}\n\n"
    
    response += f"## {cdp1.title()} Approach:\n"
    response += "\n".join([chunk['text'][:300] + "..." for chunk in chunks1[:2]])
    response += f"\n\nSource: {chunks1[0]['url']}\n\n"
    
    response += f"## {cdp2.title()} Approach:\n"
    response += "\n".join([chunk['text'][:300] + "..." for chunk in chunks2[:2]])
    response += f"\n\nSource: {chunks2[0]['url']}"
    
    return response

@app.route('/scrape', methods=['POST'])
def scrape_docs():
    """Endpoint to initiate scraping of documentation"""
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for cdp_name, config in CDP_DOCS.items():
                # Check if index already exists
                index_path = os.path.join(DATA_DIR, f"{cdp_name}_index.faiss")
                if not os.path.exists(index_path):
                    logger.info(f"Scraping documentation for {cdp_name}")
                    scraper = DocScraper(config)
                    futures[executor.submit(scraper.scrape)] = cdp_name
            
            # Process results as they complete
            for future in futures:
                cdp_name = futures[future]
                try:
                    documents = future.result()
                    if documents:
                        logger.info(f"Building index for {cdp_name} with {len(documents)} documents")
                        indexer = DocumentIndex(cdp_name)
                        indexer.build_index(documents)
                        logger.info(f"Index built for {cdp_name}")
                except Exception as e:
                    logger.error(f"Error processing {cdp_name}: {str(e)}")
        
        return jsonify({'status': 'success', 'message': 'Documentation scraped and indexed'})
    
    except Exception as e:
        logger.error(f"Error in scrape_docs: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/search', methods=['POST'])
def search():
    """Endpoint to search for answers to user questions"""
    data = request.json
    query = data.get('query', '')
    platform = data.get('platform', 'all')
    
    if not query:
        return jsonify({'status': 'error', 'message': 'No query provided'})
    
    # Check if query is related to CDP
    cdp_related = any(cdp in query.lower() for cdp in ['segment', 'mparticle', 'lytics', 'zeotap', 'cdp', 'customer data platform'])
    
    # Also check for how-to patterns
    how_to_patterns = ['how to', 'how do i', 'how can i', 'steps to', 'guide for', 'integrate', 'setup', 'configure', 'create']
    is_how_to = any(pattern in query.lower() for pattern in how_to_patterns)
    
    # If it's not a how-to question and not explicitly about CDPs, provide a friendly response
    if not is_how_to and not cdp_related:
        return jsonify({
            'status': 'success',
            'response': "I'm a specialized assistant for CDP platforms (Segment, mParticle, Lytics, and Zeotap). For the best results, please ask me how-to questions about these platforms. For example: 'How do I set up a new source in Segment?' or 'How can I create a user profile in mParticle?'"
        })
    
    try:
        if platform == 'all':
            # Try each platform and use the one with best results
            all_results = []
            for cdp_name in CDP_DOCS.keys():
                index = DocumentIndex(cdp_name)
                chunks = index.search(query)
                if chunks:
                    all_results.append({
                        'cdp': cdp_name,
                        'chunks': chunks,
                        'score': sum(1/(i+1) for i in range(len(chunks)))  # Simple relevance score
                    })
            
            if all_results:
                # Use results from most relevant platform
                all_results.sort(key=lambda x: x['score'], reverse=True)
                best_match = all_results[0]
                response = f"Based on {best_match['cdp'].title()} documentation:\n\n" + format_response(best_match['chunks'], query)
            else:
                response = "I couldn't find information about that in any of the CDP documentation. Please try rephrasing your question."
        else:
            # Search specific platform
            index = DocumentIndex(platform)
            chunks = index.search(query)
            response = format_response(chunks, query)
        
        return jsonify({'status': 'success', 'response': response})
    
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/compare', methods=['POST'])
def compare():
    """Endpoint to compare functionality between CDPs"""
    data = request.json
    query = data.get('query', '')
    cdp1 = data.get('cdp1', 'segment')
    cdp2 = data.get('cdp2', 'mparticle')
    
    if not query:
        return jsonify({'status': 'error', 'message': 'No query provided'})
    
    try:
        response = compare_cdps(query, cdp1, cdp2)
        return jsonify({'status': 'success', 'response': response})
    
    except Exception as e:
        logger.error(f"Error in compare: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/status', methods=['GET'])
def status():
    """Check if documentation has been scraped and indexed"""
    status = {}
    
    for cdp_name in CDP_DOCS.keys():
        index_path = os.path.join(DATA_DIR, f"{cdp_name}_index.faiss")
        status[cdp_name] = os.path.exists(index_path)
    
    return jsonify({'status': 'success', 'data': status})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns a dynamic port
    app.run(host="0.0.0.0", port=port)