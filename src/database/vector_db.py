from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Dict, Any, Optional

import chromadb


@dataclass
class ObjectRecord:
    # define base object schema
    object_id: str
    object_xyz: Sequence[float]
    object_image_ref: str
    object_embedding: Sequence[float]
    location_embedding: Sequence[float]


class ObjectVectorDB:
    def __init__(self, persist_directory: str = "chroma_db") -> None:
        # define local chroma client
        self._client = chromadb.PersistentClient(path=persist_directory)

        self._image_col = self._client.get_or_create_collection(
            name="objects_image",
            metadata={"hnsw:space": "cosine"},
        )
        self._location_col = self._client.get_or_create_collection(
            name="objects_location",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, record: ObjectRecord) -> None:
        # define shared metadata
        xyz = list(record.object_xyz)
        if len(xyz) != 3:
            raise ValueError("object_xyz must have length 3")

        base_meta = {
            "object_id": record.object_id,
            "object_x": float(xyz[0]),
            "object_y": float(xyz[1]),
            "object_z": float(xyz[2]),
            "object_image_ref": record.object_image_ref,
        }

        # image embedding
        self._image_col.upsert(
            ids=[f"{record.object_id}::image"],
            embeddings=[list(record.object_embedding)],
            metadatas=[{**base_meta, "embedding_type": "image"}],
            documents=[""],
        )

        # location embedding
        self._location_col.upsert(
            ids=[f"{record.object_id}::location"],
            embeddings=[list(record.location_embedding)],
            metadatas=[{**base_meta, "embedding_type": "location"}],
            documents=[""],
        )

    def delete_object(self, object_id: str) -> None:
        self._image_col.delete(ids=[f"{object_id}::image"])
        self._location_col.delete(ids=[f"{object_id}::location"])

    def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        result = self._image_col.get(
            ids=[f"{object_id}::image"],
            include=["metadatas"],
        )

        if not result or not result.get("metadatas"):
            return None

        meta = dict(result["metadatas"][0])
        # rebuild xyz for convenience
        meta["object_xyz"] = [
            meta.get("object_x"),
            meta.get("object_y"),
            meta.get("object_z"),
        ]
        return meta

    def _query(
        self,
        query_embedding: Sequence[float],
        embedding_type: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        col = self._image_col if embedding_type == "image" else self._location_col

        result = col.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            where={"embedding_type": embedding_type},
            include=["metadatas", "distances"],
        )

        # returns batched results
        metadatas_batch = result.get("metadatas", [[]])[0]
        distances_batch = result.get("distances", [[]])[0]

        hits: List[Dict[str, Any]] = []
        for meta, dist in zip(metadatas_batch, distances_batch):
            hit = dict(meta)
            
            # rebuild xyz field for caller
            hit["object_xyz"] = [
                hit.get("object_x"),
                hit.get("object_y"),
                hit.get("object_z"),
            ]
            hit["distance"] = float(dist)
            hits.append(hit)

        return hits

    def query_by_image_embedding(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        return self._query(
            query_embedding=query_embedding,
            embedding_type="image",
            n_results=n_results,
        )

    def query_by_location_embedding(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        return self._query(
            query_embedding=query_embedding,
            embedding_type="location",
            n_results=n_results,
        )


__all__ = ["ObjectRecord", "ObjectVectorDB"]