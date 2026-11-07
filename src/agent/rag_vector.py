from sentence_transformers import SentenceTransformer
from psycopg import Connection
from typing import Optional, Any
from agent.rag_helper import RAGBase
from helper.helpers import vec_to_str


class RAGPgVector(RAGBase):
    embedder: SentenceTransformer
    conn: Connection

    def __init__(self, embedder: SentenceTransformer, conn: Connection, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(
        self,
        query,
        num_results=5,
        filter_dict: Optional[dict[str, Any]] = {"enabled": True},
    ):
        query_vector = self.embedder.encode(query)
        query_str = vec_to_str(query_vector)

        query_string = """
        SELECT source_id, title, url, cca_f_domain, text
        FROM documents
        WHERE enabled = TRUE AND cca_f_domain = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """

        if filter_dict is not None:
            query_string += " WHERE " + " AND ".join(
                [f"{k} = %s" for k in filter_dict.keys()]
            )

        query_string += " ORDER BY embedding <=> %s::vector LIMIT %s"
        query_params = [*list(filter_dict.values()), query_str, num_results]

        rows = self.conn.execute(query_string, query_params).fetchall()

        return [
            {
                "source_id": r[0],
                "title": r[1],
                "url": r[2],
                "cca_f_domain": r[3],
                "text": r[4],
            }
            for r in rows
        ]
