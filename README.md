# CDP Support Agent - Backend

This is the backend API for the CDP Support Agent Chatbot, a specialized application that answers "how-to" questions about Customer Data Platforms (CDPs): Segment, mParticle, Lytics, and Zeotap by extracting relevant information from their official documentation.

## Features

- **Documentation Scraping**: Automatically scrapes official documentation from all four CDP platforms
- **Semantic Search**: Uses embeddings and vector similarity to find relevant information
- **Cross-Platform Queries**: Search across all platforms or focus on a specific one
- **Comparison Support**: API endpoints for comparing functionality between platforms

## Tech Stack

- **Flask**: Python web framework for the API
- **BeautifulSoup**: For scraping documentation websites
- **Sentence Transformers**: For creating semantic embeddings of documentation content
- **FAISS**: For efficient vector similarity search
- **Flask-CORS**: For cross-origin resource sharing

## Directory Structure

```
backend/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation and Setup

1. Make sure you have Python installed (version 3.8 or higher recommended)

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. The backend API will be available at http://localhost:5000

## API Endpoints

- **POST /api/query**
  - Accepts a question and returns relevant information from CDP documentation
  - Parameters:
    - `question` (string): The how-to question to answer
    - `platform` (string, optional): Specific platform to search ("segment", "mparticle", "lytics", "zeotap", or "all")
  - Response: JSON with answer, source links, and confidence score

- **POST /api/compare**
  - Compares functionality between two CDP platforms
  - Parameters:
    - `question` (string): The functionality to compare
    - `platform1` (string): First platform
    - `platform2` (string): Second platform
  - Response: JSON with comparison results and source links

## Initial Setup and Data Indexing

On first launch, the application will:

1. Scrape documentation from the official CDP websites
2. Process and clean the documentation content
3. Generate embeddings for efficient semantic search
4. Store the indexed data for future use

This initial process may take several minutes. Subsequent launches will use the cached data.

## Development

To enable debug mode during development:

```bash
export FLASK_ENV=development
python app.py
```

## Customization

- Update the list of CDP platforms by modifying the `PLATFORMS` constant in `app.py`
- Adjust scraping parameters in the `scrape_documentation` function

## Notes

- The application requires internet access for the initial documentation scraping
- The quality of responses depends on the quality and structure of the official documentation
- This is a demonstration project and not affiliated with any of the CDP platforms