import chromadb
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Union
import os


class VectorDBManager:
    def __init__(
        self,
        persist_directory: str = "./vector_db",
        provider: str = "sentence_transformer",
        model_name: str = "all-mpnet-base-v2",
    ):
        """
        Initialize the Vector Database Manager.

        Args:
            persist_directory (str): Directory to persist the database
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        if provider == "sentence_transformer":
            self.model = SentenceTransformer(model_name)

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def create_database(
        self,
        csv_path: str,
        collection_name: str,
        distance_metric: str = "l2",
        collection_metadata: Dict = None,
    ) -> bool:
        """
        Create a new database from a CSV file.

        Args:
            csv_path (str): Path to the CSV file containing 'id' and 'text' columns
            collection_name (str): Name of the collection to create
            distance_metric (str): Distance metric (l2, cosine, ip)
            collection_metadata (Dict, optional): Additional metadata for the collection

        Returns:
            bool: True if successful, False otherwise
        """
        try:

            df = pd.read_csv(csv_path)

            if not {"id", "text"}.issubset(df.columns):
                self.logger.error("CSV must contain 'id' and 'text' columns")
                return False

            collection_meta = {
                "hnsw:space": distance_metric,
                "description": f"Collection created from {csv_path}",
            }

            if collection_metadata:
                collection_meta.update(collection_metadata)

            collection = self.client.create_collection(
                name=collection_name, metadata=collection_meta
            )

            embeddings = self.model.encode(df["text"].tolist()).tolist()

            collection.add(
                ids=[str(id_) for id_ in df["id"]],
                documents=df["text"].tolist(),
                embeddings=embeddings,
            )

            self.logger.info(f"Successfully created collection '{collection_name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error creating database: {str(e)}")
            return False

    def add_values_from_csv(
        self, csv_path: str, collection_name: str
    ) -> Dict[str, int]:
        """
        Add values from CSV file to existing collection, skipping existing IDs.

        Args:
            csv_path (str): Path to the CSV file
            collection_name (str): Name of the target collection

        Returns:
            Dict[str, int]: Statistics about the operation
        """
        try:

            df = pd.read_csv(csv_path)

            collection = self.client.get_collection(collection_name)

            existing_ids = set(collection.get()["ids"])

            new_df = df[~df["id"].astype(str).isin(existing_ids)]

            if not new_df.empty:

                embeddings = self.model.encode(new_df["text"].tolist()).tolist()

                collection.add(
                    ids=[str(id_) for id_ in new_df["id"]],
                    documents=new_df["text"].tolist(),
                    embeddings=embeddings,
                )

            stats = {
                "total_entries": len(df),
                "new_entries_added": len(new_df),
                "skipped_entries": len(df) - len(new_df),
            }

            self.logger.info(
                f"Added {stats['new_entries_added']} new entries to '{collection_name}'"
            )
            return stats

        except Exception as e:
            self.logger.error(f"Error adding values from CSV: {str(e)}")
            return {"error": str(e)}

    def add_single_row(self, id_: str, text: str, collection_name: str) -> bool:
        """
        Add a single entry to the collection.

        Args:
            id_ (str): ID for the new entry
            text (str): Text content
            collection_name (str): Target collection name

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            collection = self.client.get_collection(collection_name)

            if str(id_) in collection.get()["ids"]:
                self.logger.warning(f"ID {id_} already exists in collection")
                return False

            embedding = self.model.encode([text]).tolist()

            collection.add(ids=[str(id_)], documents=[text], embeddings=embedding)

            self.logger.info(f"Successfully added entry with ID {id_}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding single row: {str(e)}")
            return False

    def delete_entry_by_id(self, id_: str, collection_name: str) -> bool:
        """
        Delete an entry by its ID.

        Args:
            id_ (str): ID of the entry to delete
            collection_name (str): Collection name

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            collection = self.client.get_collection(collection_name)

            if str(id_) not in collection.get()["ids"]:
                self.logger.warning(f"ID {id_} not found in collection")
                return False

            collection.delete(ids=[str(id_)])

            self.logger.info(f"Successfully deleted entry with ID {id_}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting entry: {str(e)}")
            return False

    def find_nearby_texts(
        self,
        text: str,
        collection_name: str,
        search_string: Optional[str] = None,
        k: int = 5,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Find nearby texts using similarity search with scores.

        Args:
            text (str): Query text
            collection_name (str): Collection to search in
            k (int): Number of results to return

        Returns:
            List[Dict[str, Union[str, float]]]: List of nearby texts with their IDs and similarity scores
        """
        try:
            collection = self.client.get_collection(collection_name)
            print("Metadata:", collection.metadata)

            distance_metric = collection.metadata["hnsw:space"]

            query_embedding = self.model.encode([text]).tolist()

            if search_string:
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=k,
                    include=["documents", "distances", "metadatas"],
                    where_document={"$contains": search_string},
                )
            else:
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=k,
                    include=["documents", "distances", "metadatas"],
                )

            distances = results["distances"][0]
            if not distances:
                return []

            similarities = []
            for dist in distances:
                if distance_metric == "cosine":

                    similarity = 1 - dist
                elif distance_metric == "ip":

                    min_dist = min(distances)
                    max_dist = max(distances)
                    similarity = (
                        (dist - min_dist) / (max_dist - min_dist)
                        if max_dist > min_dist
                        else 1.0
                    )
                elif distance_metric == "l1":

                    max_dist = max(distances)
                    similarity = 1 - (dist / max_dist) if max_dist > 0 else 1.0
                elif distance_metric == "l2":

                    max_dist = max(distances)
                    similarity = 1 - (dist / max_dist) if max_dist > 0 else 1.0

                similarities.append(similarity)

            nearby_texts = [
                {
                    "id": id_,
                    "text": text_,
                    "similarity": round(similarity * 100, 4),
                    "raw_distance": dist,
                    "metric": distance_metric,
                }
                for id_, text_, similarity, dist in zip(
                    results["ids"][0], results["documents"][0], similarities, distances
                )
            ]

            return nearby_texts

        except Exception as e:
            self.logger.error(f"Error finding nearby texts: {str(e)}")
            return []

    def delete_collection(self, collection_name: str, confirmation: str = "no") -> bool:
        """
        Delete an entire collection.

        Args:
            collection_name (str): Name of collection to delete
            confirmation (str): Must be 'yes' to proceed

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if confirmation.lower() != "yes":
                self.logger.warning("Deletion cancelled - confirmation not provided")
                return False

            self.client.delete_collection(collection_name)
            self.logger.info(f"Successfully deleted collection '{collection_name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting collection: {str(e)}")
            return False

    def get_collection_info(self, collection_name: str) -> Dict:
        """
        Get information about a collection.

        Args:
            collection_name (str): Name of the collection

        Returns:
            Dict: Collection information and statistics
        """
        try:
            collection = self.client.get_collection(collection_name)
            collection_data = collection.get()

            info = {
                "name": collection_name,
                "count": len(collection_data["ids"]),
                "metadata": collection.metadata,
            }

            return info

        except Exception as e:
            self.logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}

    def get_by_ids(self, ids: List[str], collection_name: str) -> Dict[str, str]:
        """
        Get texts for given IDs in batch.

        Args:
            ids (List[str]): List of IDs to fetch
            collection_name (str): Name of the collection

        Returns:
            Dict[str, str]: Dictionary mapping IDs to their corresponding texts
        """
        try:
            collection = self.client.get_collection(collection_name)

            str_ids = [str(id_) for id_ in ids]

            results = collection.get(ids=str_ids, include=["documents"])

            id_to_text = {
                id_: text for id_, text in zip(results["ids"], results["documents"])
            }

            return id_to_text

        except Exception as e:
            self.logger.error(f"Error getting texts by IDs: {str(e)}")
            return {}

    def get_by_texts(self, texts: List[str], collection_name: str) -> Dict[str, str]:
        """
        Get IDs for given texts in batch.
        Note: For exact text matching. For similar texts, use find_nearby_texts.

        Args:
            texts (List[str]): List of texts to fetch
            collection_name (str): Name of the collection

        Returns:
            Dict[str, str]: Dictionary mapping texts to their corresponding IDs
        """
        try:
            collection = self.client.get_collection(collection_name)

            all_data = collection.get()

            text_to_id = {
                text: id_
                for text, id_ in zip(all_data["documents"], all_data["ids"])
                if text in texts
            }

            return text_to_id

        except Exception as e:
            self.logger.error(f"Error getting IDs by texts: {str(e)}")
            return {}
