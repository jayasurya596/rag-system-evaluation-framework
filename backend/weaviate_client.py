import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import weaviate
from weaviate.classes.init import Auth
import weaviate.classes.config as wvc_config
from backend.config import WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_COLLECTION, logger, WORKSPACE_DIR

class MockWeaviateCollection:
    """A mock Weaviate collection stored in a JSON file for robust offline execution without sqlite3 DLL dependencies."""
    _cache = None
    _cache_path = None

    def __init__(self, json_path: str = None):
        if json_path is None:
            json_path = str(WORKSPACE_DIR / "mock_weaviate.json")
        self.json_path = json_path
        self._init_db()

    @classmethod
    def _load_data(cls, json_path: str) -> Dict[str, Any]:
        if cls._cache is not None and cls._cache_path == json_path:
            return cls._cache
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cls._cache = data
            cls._cache_path = json_path
            return data
        except Exception:
            return {}

    @classmethod
    def _save_data(cls, json_path: str, data: Dict[str, Any]):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        cls._cache = data
        cls._cache_path = json_path

    def _init_db(self):
        if not os.path.exists(self.json_path):
            self._save_data(self.json_path, {})

    def insert_many(self, objects: List[Dict[str, Any]], vectors: List[List[float]]):
        # Load existing
        data = self._load_data(self.json_path)

        for obj, vec in zip(objects, vectors):
            props = obj["properties"]
            uuid_val = obj.get("uuid", props["document_id"])
            data[uuid_val] = {
                "content": props["content"],
                "company_name": props["company_name"],
                "filing_type": props["filing_type"],
                "filing_date": props["filing_date"],
                "document_id": props["document_id"],
                "source_url": props["source_url"],
                "vector": vec
            }
            
        self._save_data(self.json_path, data)

    def vector_search(self, query_vector: List[float], limit: int = 15, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        data = self._load_data(self.json_path)
        if not data:
            return []

        # Filter
        filtered_items = []
        for uuid_val, doc in data.items():
            match = True
            if filters:
                for k, v in filters.items():
                    if k == "year":
                        doc_year = ""
                        if doc.get("filing_date"):
                            doc_year = doc["filing_date"].split("-")[0]
                        elif doc.get("document_id"):
                            parts = doc["document_id"].split("_")
                            if len(parts) >= 3:
                                doc_year = parts[2]
                        if isinstance(v, list):
                            if doc_year not in v:
                                match = False
                                break
                        else:
                            if doc_year != v:
                                match = False
                                break
                    else:
                        if doc.get(k) != v:
                            match = False
                            break
            if match:
                filtered_items.append((uuid_val, doc))

        if not filtered_items:
            return []

        # Compute Cosine Similarity
        q_vec = np.array(query_vector, dtype=np.float32)
        results = []
        for uuid_val, doc in filtered_items:
            db_vec = np.array(doc["vector"], dtype=np.float32)
            similarity = float(np.dot(q_vec, db_vec))
            
            results.append({
                "properties": {
                    "content": doc["content"],
                    "company_name": doc["company_name"],
                    "filing_type": doc["filing_type"],
                    "filing_date": doc["filing_date"],
                    "document_id": doc["document_id"],
                    "source_url": doc["source_url"]
                },
                "score": similarity,
                "uuid": uuid_val
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def get_all_documents(self) -> List[Dict[str, Any]]:
        data = self._load_data(self.json_path)
        if not data:
            return []
            
        return [{
            "content": doc["content"],
            "company_name": doc["company_name"],
            "filing_type": doc["filing_type"],
            "filing_date": doc["filing_date"],
            "document_id": doc["document_id"],
            "source_url": doc["source_url"]
        } for doc in data.values()]

    def count(self) -> int:
        data = self._load_data(self.json_path)
        return len(data)


class WeaviateClientWrapper:
    def __init__(self):
        self.client = None
        self.mock_db = None
        self.is_mock = False
        self._init_client()

    def _init_client(self):
        if WEAVIATE_URL:
            try:
                logger.info(f"Connecting to Weaviate at: {WEAVIATE_URL}...")
                if WEAVIATE_API_KEY:
                    self.client = weaviate.connect_to_wcd(
                        connection_string=WEAVIATE_URL,
                        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
                        skip_init_checks=True
                    )
                else:
                    self.client = weaviate.connect_to_local(
                        uri=WEAVIATE_URL,
                        skip_init_checks=True
                    )
                logger.info("Connected to Weaviate successfully.")
            except Exception as e:
                logger.warning(f"Failed to connect to Weaviate. Falling back to local mock database: {e}")
                self.is_mock = True
                self.mock_db = MockWeaviateCollection()
        else:
            logger.info("No WEAVIATE_URL provided. Operating in OFFLINE MOCK database mode.")
            self.is_mock = True
            self.mock_db = MockWeaviateCollection()

    def create_collection(self):
        """Create collection in Weaviate or log schema for Mock."""
        if self.is_mock:
            self.mock_db._init_db()
            logger.info(f"Initialized mock Weaviate collection '{WEAVIATE_COLLECTION}' in JSON.")
            return

        try:
            if self.client.collections.exists(WEAVIATE_COLLECTION):
                self.client.collections.delete(WEAVIATE_COLLECTION)
                logger.info(f"Deleted existing Weaviate collection '{WEAVIATE_COLLECTION}'.")

            self.client.collections.create(
                name=WEAVIATE_COLLECTION,
                vectorizer_config=None,
                properties=[
                    wvc_config.Property(name="content", data_type=wvc_config.DataType.TEXT),
                    wvc_config.Property(name="company_name", data_type=wvc_config.DataType.TEXT),
                    wvc_config.Property(name="filing_type", data_type=wvc_config.DataType.TEXT),
                    wvc_config.Property(name="filing_date", data_type=wvc_config.DataType.TEXT),
                    wvc_config.Property(name="document_id", data_type=wvc_config.DataType.TEXT),
                    wvc_config.Property(name="source_url", data_type=wvc_config.DataType.TEXT),
                ]
            )
            logger.info(f"Created Weaviate collection '{WEAVIATE_COLLECTION}' successfully.")
        except Exception as e:
            logger.error(f"Error creating Weaviate collection: {e}. Switching to Mock.")
            self.is_mock = True
            self.mock_db = MockWeaviateCollection()

    def upload_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Upload chunks and their corresponding embeddings in batch."""
        if self.is_mock:
            objects = []
            for c in chunks:
                objects.append({
                    "properties": {
                        "content": c["content"],
                        "company_name": c["company_name"],
                        "filing_type": c["filing_type"],
                        "filing_date": c["filing_date"],
                        "document_id": c["document_id"],
                        "source_url": c["source_url"]
                    }
                })
            self.mock_db.insert_many(objects, embeddings)
            logger.info(f"Uploaded {len(chunks)} chunks to Mock JSON Weaviate database.")
            return

        try:
            collection = self.client.collections.get(WEAVIATE_COLLECTION)
            with collection.batch.dynamic() as batch:
                for chunk, embedding in zip(chunks, embeddings):
                    batch.add_object(
                        properties={
                            "content": chunk["content"],
                            "company_name": chunk["company_name"],
                            "filing_type": chunk["filing_type"],
                            "filing_date": chunk["filing_date"],
                            "document_id": chunk["document_id"],
                            "source_url": chunk["source_url"]
                        },
                        vector=embedding
                    )
            logger.info(f"Uploaded {len(chunks)} chunks to Weaviate Cloud collection '{WEAVIATE_COLLECTION}'.")
        except Exception as e:
            logger.error(f"Failed Weaviate upload: {e}. Uploading to fallback JSON Mock.")
            self.is_mock = True
            self.mock_db = MockWeaviateCollection()
            self.upload_chunks(chunks, embeddings)

    def vector_search(self, query_vector: List[float], limit: int = 15, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform dense vector search with optional filters."""
        if self.is_mock:
            return self.mock_db.vector_search(query_vector, limit, filters)

        try:
            collection = self.client.collections.get(WEAVIATE_COLLECTION)
            weaviate_filter = None
            if filters:
                from weaviate.classes.query import Filter
                filter_list = []
                for k, v in filters.items():
                    if k == "year":
                        if isinstance(v, list):
                            or_filters = [Filter.by_property("filing_date").like(f"{y}*") for y in v]
                            if len(or_filters) > 1:
                                filter_list.append(Filter.any_of(or_filters))
                            elif or_filters:
                                filter_list.append(or_filters[0])
                        else:
                            filter_list.append(Filter.by_property("filing_date").like(f"{v}*"))
                    else:
                        filter_list.append(Filter.by_property(k).equal(v))
                if len(filter_list) > 1:
                    weaviate_filter = Filter.all_of(filter_list)
                elif filter_list:
                    weaviate_filter = filter_list[0]

            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                filters=weaviate_filter,
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
            )

            results = []
            for obj in response.objects:
                dist = obj.metadata.distance if obj.metadata.distance is not None else 1.0
                score = 1.0 - dist
                results.append({
                    "properties": obj.properties,
                    "score": score,
                    "uuid": str(obj.uuid)
                })
            return results
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}. Falling back to JSON mock search.")
            if not self.mock_db:
                self.mock_db = MockWeaviateCollection()
            return self.mock_db.vector_search(query_vector, limit, filters)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        if self.is_mock:
            return self.mock_db.get_all_documents()
        try:
            collection = self.client.collections.get(WEAVIATE_COLLECTION)
            response = collection.query.fetch_objects(limit=10000)
            return [obj.properties for obj in response.objects]
        except Exception as e:
            logger.error(f"Failed to fetch from Weaviate: {e}")
            if not self.mock_db:
                self.mock_db = MockWeaviateCollection()
            return self.mock_db.get_all_documents()

    def count_documents(self) -> int:
        if self.is_mock:
            return self.mock_db.count()
        try:
            collection = self.client.collections.get(WEAVIATE_COLLECTION)
            response = collection.aggregate.over_all(total_count=True)
            return response.total_count
        except Exception as e:
            logger.error(f"Failed to count in Weaviate: {e}")
            if not self.mock_db:
                self.mock_db = MockWeaviateCollection()
            return self.mock_db.count()

    def close(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
