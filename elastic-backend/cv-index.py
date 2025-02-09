import os
import pandas as pd
from elasticsearch import Elasticsearch, helpers
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ES_HOST = "http://localhost:9200"
INDEX_NAME = "cv-transcriptions"

def create_es_client():
    try:
        es = Elasticsearch([ES_HOST])
        if not es.ping():
            raise Exception("Connection failed")
        return es
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {e}")
        raise

def create_index(es):
    mapping = {
        "mappings": {
            "properties": {
                "filename": {"type": "keyword"},
                "text": {"type": "text"},
                "up_votes": {"type": "integer"},
                "down_votes": {"type": "integer"},
                "age": {"type": "keyword"},
                "gender": {"type": "keyword"},
                "accent": {"type": "keyword"},
                "duration": {"type": "float"},
                "generated_text": {"type": "text"}
            }
        },
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1
        }
    }
    
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=mapping)
        logger.info(f"Created index: {INDEX_NAME}")

def generate_actions(df):
    for index, row in df.iterrows():
        try:
            up_votes = int(row.get("up_votes", 0)) if pd.notna(row.get("up_votes")) else 0
            down_votes = int(row.get("down_votes", 0)) if pd.notna(row.get("down_votes")) else 0
            duration = float(row.get("duration", 0)) if pd.notna(row.get("duration")) else None
        except (ValueError, TypeError) as e:
            logger.warning(f"Error converting numeric values for row {index}: {e}")
            up_votes = 0
            down_votes = 0
            duration = None

        doc = {
            "filename": str(row.get("filename", "")) if pd.notna(row.get("filename")) else "",
            "text": str(row.get("text", "")) if pd.notna(row.get("text")) else "",
            "up_votes": up_votes,
            "down_votes": down_votes,
            "age": None if pd.isna(row.get("age")) else str(row.get("age")),
            "gender": None if pd.isna(row.get("gender")) else str(row.get("gender")),
            "accent": None if pd.isna(row.get("accent")) else str(row.get("accent")),
            "duration": duration,
            "generated_text": str(row.get("generated_text", "")) if pd.notna(row.get("generated_text")) else ""
        }

        doc = {k: v for k, v in doc.items() if v is not None and v != ""}
        
        yield {
            "_index": INDEX_NAME,
            "_id": str(index),
            "_source": doc
        }

def main():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "data/cv-valid-dev.csv")
        logger.info("Reading CSV file...")
        df = pd.read_csv(csv_path)
        
        es = create_es_client()
        
        create_index(es)
        
        logger.info("Starting bulk indexing...")
        try:
            success, errors = helpers.bulk(
                es,
                generate_actions(df),
                raise_on_error=False,
                raise_on_exception=False
            )
            
            logger.info(f"Indexed {success} documents successfully")
            if errors:
                logger.error("Indexing errors occurred:")
                for error in errors:
                    logger.error(f"Document error: {error}")
                    
                with open('indexing_errors.log', 'w') as f:
                    for error in errors:
                        f.write(f"{str(error)}\n")
                logger.info("Errors have been saved to indexing_errors.log")
        except Exception as bulk_error:
            logger.error(f"Bulk indexing error: {bulk_error}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()