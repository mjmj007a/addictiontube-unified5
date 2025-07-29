from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import chromadb
import logging
from dotenv import load_dotenv
import os
from openai import OpenAI
import httpx

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/search_content": {"origins": "https://addictiontube.com"}})
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Chroma client (in-memory)
def get_chroma_client():
    for attempt in range(3):
        try:
            client = chromadb.Client()
            logger.info(f"Chroma client initialized, attempt {attempt + 1}")
            try:
                collection = client.get_collection("Content")
                logger.info("Content collection found")
                return client, collection
            except Exception as e:
                logger.warning(f"'Content' collection not found: {str(e)}")
                try:
                    logger.info("Creating 'Content' collection in Chroma...")
                    collection = client.create_collection(name="Content", embedding_function=None)
                    logger.info("Content collection created successfully.")
                    return client, collection
                except Exception as inner_e:
                    logger.error(f"Failed to create Content collection: {str(inner_e)}")
                    client.close()
                    raise
        except Exception as e:
            logger.error(f"Chroma client initialization attempt {attempt + 1} failed: {str(e)}")
    logger.error("Failed to initialize Chroma client after 3 attempts")
    raise EnvironmentError("Chroma client initialization failed")

client, collection = get_chroma_client()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

@app.route('/')
def health_check():
    try:
        collection.count()
        return jsonify({"status": "healthy", "chroma": "connected"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/search_content', methods=['GET'])
@limiter.limit("50 per hour")
def search_content():
    try:
        query = request.args.get('q', '')
        content_type = request.args.get('content_type', 'all')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 5))

        if not query:
            return jsonify({"error": "Query parameter 'q' is required"}), 400

        where_clause = {"type": content_type} if content_type != 'all' else {}
        results = collection.query(
            query_texts=[query],
            n_results=per_page,
            where=where_clause,
            include=["metadatas", "documents", "distances"]
        )

        formatted_results = [
            {
                "id": results["ids"][0][i],
                "title": results["metadatas"][0][i]["title"],
                "description": results["metadatas"][0][i]["description"],
                "url": results["metadatas"][0][i]["url"],
                "type": results["metadatas"][0][i]["type"],
                "category": results["metadatas"][0][i]["category"],
                "date": results["metadatas"][0][i]["date"],
                "distance": results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]

        return jsonify({
            "results": formatted_results,
            "page": page,
            "per_page": per_page,
            "total": len(formatted_results)
        }), 200
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/rag_answer_content', methods=['GET'])
@limiter.limit("10 per hour")
def rag_answer_content():
    try:
        query = request.args.get('q', '')
        content_type = request.args.get('content_type', 'all')
        reroll = request.args.get('reroll', 'no').lower() == 'yes'

        if not query:
            return jsonify({"error": "Query parameter 'q' is required"}), 400

        where_clause = {"type": content_type} if content_type != 'all' else {}
        results = collection.query(
            query_texts=[query],
            n_results=5,
            where=where_clause,
            include=["metadatas", "documents"]
        )

        context = "\n".join([doc for doc in results["documents"][0]])
        prompt = f"Based on the following context, provide a concise and accurate answer to the question: {query}\n\nContext:\n{context}"

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing concise answers based on given context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        answer = response.choices[0].message.content.strip()
        return jsonify({
            "answer": answer,
            "context_ids": results["ids"][0],
            "context_titles": [meta["title"] for meta in results["metadatas"][0]]
        }), 200
    except Exception as e:
        logger.error(f"RAG answer error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
