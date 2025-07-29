import chromadb
import json
import os
import logging
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from dateutil import parser
import pytz
import tiktoken
from bs4 import BeautifulSoup
import httpx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = chromadb.Client()
logger.info("Connected to Chroma in-memory client")

CATEGORY_MAP = {
    "1074": "Songs - Recovery",
    "1082": "Poems - Recovery",
    "1073": "Stories - Recovery",
    "1028": "Stories - Recovery",
    "1042": "Stories - Support"
}

def extract_tags(description):
    keywords = ["addiction", "recovery", "hope", "resilience", "empathy", "homelessness", "redemption", "chaos", "despair", "healing", "drug", "drugs"]
    return [kw for kw in keywords if kw.lower() in description.lower()]

def format_date(date_str):
    try:
        if not date_str:
            return datetime.now(tz=pytz.UTC).isoformat()
        parsed = parser.parse(date_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tz=pytz.UTC)
        return parsed.isoformat()
    except:
        return datetime.now(tz=pytz.UTC).isoformat()

def strip_html(html_text):
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        logger.error(f"HTML stripping error: {e}")
        return html_text

def truncate_text(text, max_tokens=7500):
    try:
        enc = tiktoken.encoding_for_model("text-embedding-3-small")
        tokens = enc.encode(text)
        tokens = tokens[:min(len(tokens), max_tokens)]
        truncated_text = enc.decode(tokens)
        logger.debug(f"Truncated text to {len(enc.encode(truncated_text))} tokens")
        return truncated_text
    except Exception as e:
        logger.error(f"Tokenizer error: {e}")
        return text[:30000]

def split_text(text, max_tokens=7500):
    try:
        enc = tiktoken.encoding_for_model("text-embedding-3-small")
        tokens = enc.encode(text)
        parts = []
        for i in range(0, len(tokens), max_tokens):
            part_tokens = tokens[i:i + max_tokens]
            part_text = enc.decode(part_tokens)
            part_text = truncate_text(part_text, max_tokens)
            part_token_count = len(enc.encode(part_text))
            if part_token_count > 8191:
                logger.error(f"Part still exceeds 8191 tokens: {part_token_count}")
                continue
            parts.append(part_text)
        return parts
    except Exception as e:
        logger.error(f"Split text error: {e}")
        return [truncate_text(text, max_tokens)]

def compute_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_embedding(text):
    try:
        response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        embedding = response.data[0].embedding
        if not isinstance(embedding, list):
            raise ValueError(f"Invalid embedding type: expected list, got {type(embedding)}")
        if len(embedding) != 1536:
            raise ValueError(f"Invalid embedding length: expected 1536, got {len(embedding)}")
        if not all(isinstance(x, (int, float)) for x in embedding):
            raise ValueError(f"Invalid embedding values: expected floats, got {type(embedding[0])}")
        logger.debug(f"Embedding type: {type(embedding)}, length: {len(embedding)}, sample: {embedding[:5]}")
        return [float(x) for x in embedding]
    except Exception as e:
        logger.error(f"OpenAI embedding error: {e}")
        return None

skipped_items = []
failed_objs_serializable = []

try:
    try:
        client.delete_collection("Content")
        logger.info("Existing 'Content' collection deleted.")
    except:
        logger.info("No existing 'Content' collection found.")

    collection = client.create_collection(name="Content", embedding_function=None)
    logger.info("✅ 'Content' collection created")

    json_files = [
        "songs_revised_with_songs-july06.json",
        "videos_revised_with_poems-july04.json",
        "stories.json"
    ]

    total_uploaded = 0
    total_skipped = 0
    point_id_counter = 1

    for json_file in json_files:
        if not os.path.exists(json_file):
            logger.warning(f"{json_file} not found")
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        embeddings = []
        documents = []
        metadatas = []
        ids = []

        for item in data:
            content_type = "stories" if json_file == "stories.json" else ("poems" if "poem" in item else "songs")
            raw_html = item.get("poem") or item.get("song") or item.get("text")
            if not raw_html:
                logger.warning(f"Skipping: No content for {item.get('title', '[no title]')}")
                skipped_items.append(item)
                total_skipped += 1
                continue

            clean_text = strip_html(raw_html)
            enc = tiktoken.encoding_for_model("text-embedding-3-small")
            token_count = len(enc.encode(clean_text))
            logger.debug(f"Processing {item.get('title', '[no title]')} - {token_count} tokens before splitting")

            text_parts = split_text(clean_text, max_tokens=7500)
            for idx, part in enumerate(text_parts):
                if not part.strip():
                    logger.warning(f"Skipping: Empty part for {item.get('title', '[no title]')}")
                    skipped_items.append({"title": item.get("title", "[no title]"), "part": idx + 1})
                    total_skipped += 1
                    continue

                part_token_count = len(enc.encode(part))
                logger.debug(f"Part {idx + 1} of {item.get('title', '[no title]')} - {part_token_count} tokens")
                if part_token_count > 8191:
                    logger.error(f"Part {idx + 1} of {item.get('title', '[no title]')} exceeds 8191 tokens: {part_token_count}")
                    skipped_items.append({"title": item.get("title", "[no title]"), "part": idx + 1, "tokens": part_token_count})
                    total_skipped += 1
                    continue

                vector = get_embedding(part)
                if vector is None:
                    logger.error(f"Skipping: Failed to generate embedding for {item.get('title', '[no title]')}")
                    skipped_items.append({"title": item.get("title", "[no title]"), "part": idx + 1})
                    total_skipped += 1
                    continue

                content_id = item.get('id' if json_file == 'stories.json' else 'video_id', '')
                point_id = str(point_id_counter)
                point_id_counter += 1

                metadata = {
                    "text": part,
                    "html": raw_html,
                    "html_sha256": compute_sha256(raw_html),
                    "title": f"{item.get('title', '')} (Part {idx + 1})" if len(text_parts) > 1 else item.get("title", ""),
                    "description": item.get("description", ""),
                    "category": CATEGORY_MAP.get(item.get("category" if json_file == "stories.json" else "category_id", ""), "Unknown"),
                    "date": format_date(item.get("date", "")),
                    "url": item.get("video_location") or item.get("image") or f"https://addictiontube.com/articles/read/{item['title'].lower().replace(' ', '-')}_{content_id}.html",
                    "type": content_type,
                    "tags": ", ".join(extract_tags(item.get("description", ""))),
                    "author": item.get("author", "Unknown"),
                    "schema_version": "v1.1",
                    "content_id": content_id
                }

                embeddings.append(vector)
                documents.append(part)
                metadatas.append(metadata)
                ids.append(point_id)

        if embeddings:
            try:
                collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"✅ {len(embeddings)} points inserted from {json_file}")
                total_uploaded += len(embeddings)
            except Exception as e:
                logger.error(f"Failed to insert points from {json_file}: {e}")
                failed_objs_serializable.extend([{
                    "title": metadata["title"],
                    "error": str(e),
                    "metadata": metadata
                } for metadata in metadatas])
                total_skipped += len(embeddings)
        else:
            logger.warning(f"No points prepared for {json_file}")

    logger.info(f"TOTAL uploaded: {total_uploaded}")
    logger.info(f"TOTAL skipped: {total_skipped}")

    try:
        for content_type in ["songs", "poems", "stories"]:
            results = collection.query(
                query_texts=["test"],
                where={"type": content_type},
                n_results=1,
                include=["metadatas"]
            )
            if results["metadatas"][0]:
                logger.info(f"✅ Filtering works for 'type: {content_type}' - found {len(results['metadatas'][0])} points")
            else:
                logger.warning(f"⚠️ No points found for 'type: {content_type}' - filtering may not work")
    except Exception as e:
        logger.error(f"Failed to verify filtering for 'type': {e}")

    if skipped_items:
        logger.info(f"📝 Skipped {len(skipped_items)} items (not saved due to in-memory mode)")
    if failed_objs_serializable:
        logger.info(f"⚠️ Failed {len(failed_objs_serializable)} inserts (not saved due to in-memory mode)")

except Exception as e:
    logger.error(f"❗ Error: {e}")
    raise

