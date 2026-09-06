import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGEngine:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.documents = []
        self.embeddings = None
        self.index = None


    # ==========================================
    # 1. ADD MULTIPLE DOCUMENTS
    # ==========================================

    def add_documents(self, documents):

        """
        Add processed documents to the RAG system.

        Each document should contain:

        {
            "text": "...",
            "metadata": {
                "filename": "...",
                "subject": "...",
                "chapter": "..."
            }
        }
        """

        if not documents:
            return


        self.documents.extend(documents)

        self._rebuild_index()


    # ==========================================
    # 2. BUILD / REBUILD FAISS INDEX
    # ==========================================

    def _rebuild_index(self):

        if not self.documents:
            return


        texts = [
            document["text"]
            for document in self.documents
        ]


        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )


        # FAISS works with float32
        self.embeddings = self.embeddings.astype(
            "float32"
        )


        dimension = self.embeddings.shape[1]


        self.index = faiss.IndexFlatL2(
            dimension
        )


        self.index.add(
            self.embeddings
        )


        print(
            "FAISS index updated:",
            self.index.ntotal,
            "vectors"
        )


    # ==========================================
    # 3. RETRIEVE RELEVANT DOCUMENTS
    # ==========================================

    def retrieve(self, query, k=5):

        if self.index is None:
            return []


        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )


        query_embedding = query_embedding.astype(
            "float32"
        )


        k = min(
            k,
            len(self.documents)
        )


        distances, indices = self.index.search(
            query_embedding,
            k
        )


        results = []


        for distance, idx in zip(
            distances[0],
            indices[0]
        ):

            if idx < 0:
                continue


            document = self.documents[idx]


            results.append({

                "text": document["text"],

                "distance": float(distance),

                "metadata": document["metadata"]

            })


        return results


    # ==========================================
    # 4. BUILD CONTEXT
    # ==========================================

    def build_context(self, results):

        context_parts = []


        for result in results:

            metadata = result["metadata"]


            context_parts.append(
                f"""
Source: {metadata["filename"]}
Subject: {metadata["subject"]}
Chapter: {metadata["chapter"]}

Content:
{result["text"]}
"""
            )


        return "\n".join(
            context_parts
        )


    # ==========================================
    # 5. GET DOCUMENT COUNT
    # ==========================================

    def get_document_count(self):

        return len(
            self.documents
        )


    # ==========================================
    # 6. GET VECTOR COUNT
    # ==========================================

    def get_vector_count(self):

        if self.index is None:
            return 0

        return self.index.ntotal


    # ==========================================
    # 7. GET AVAILABLE SUBJECTS
    # ==========================================

    def get_subjects(self):

        subjects = set()


        for document in self.documents:

            subject = document["metadata"].get(
                "subject",
                "General"
            )

            subjects.add(subject)


        return sorted(
            subjects
        )


    # ==========================================
    # 8. GET AVAILABLE CHAPTERS
    # ==========================================

    def get_chapters(self, subject=None):

        chapters = set()


        for document in self.documents:

            metadata = document["metadata"]


            if subject is not None:

                if metadata.get("subject") != subject:
                    continue


            chapters.add(
                metadata.get(
                    "chapter",
                    "General"
                )
            )


        return sorted(
            chapters
        )