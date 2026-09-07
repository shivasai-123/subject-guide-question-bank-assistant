import faiss
import numpy as np
import re
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
    # ADD DOCUMENTS
    # ==========================================

    def add_documents(self, documents):

        if not documents:
            return

        self.documents.extend(
            documents
        )

        self._rebuild_index()


    # ==========================================
    # BUILD FAISS INDEX
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
    # RETRIEVE DOCUMENTS
    # ==========================================

    def retrieve(
        self,
        query,
        k=5,
        subject=None,
        chapter=None
    ):

        if self.index is None:
            return []


        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype(
            "float32"
        )


        # --------------------------------------
        # Find matching documents
        # --------------------------------------

        candidate_indices = []

        for i, document in enumerate(
            self.documents
        ):

            metadata = document["metadata"]


            if subject is not None:

                if metadata.get(
                    "subject"
                ) != subject:

                    continue


            if chapter is not None:

                if metadata.get(
                    "chapter"
                ) != chapter:

                    continue


            candidate_indices.append(i)


        if not candidate_indices:
            return []


        # --------------------------------------
        # Create filtered FAISS index
        # --------------------------------------

        candidate_embeddings = (
            self.embeddings[
                candidate_indices
            ]
        )


        filtered_index = faiss.IndexFlatL2(
            candidate_embeddings.shape[1]
        )


        filtered_index.add(
            candidate_embeddings
        )


        k = min(
            k,
            len(candidate_indices)
        )


        distances, indices = (
            filtered_index.search(
                query_embedding,
                k
            )
        )


        results = []


        for distance, filtered_idx in zip(
            distances[0],
            indices[0]
        ):

            if filtered_idx < 0:
                continue


            original_idx = (
                candidate_indices[
                    filtered_idx
                ]
            )


            document = self.documents[
                original_idx
            ]


            results.append(
                {
                    "text": document["text"],

                    "distance": float(
                        distance
                    ),

                    "metadata": document[
                        "metadata"
                    ]
                }
            )


        return results


    # ==========================================
    # BUILD CONTEXT
    # ==========================================

    def build_context(self, results):

        context_parts = []


        for result in results:

            metadata = result[
                "metadata"
            ]


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
    # DOCUMENT COUNT
    # ==========================================

    def get_document_count(self):

        return len(
            self.documents
        )


    # ==========================================
    # VECTOR COUNT
    # ==========================================

    def get_vector_count(self):

        if self.index is None:
            return 0

        return self.index.ntotal


    # ==========================================
    # GET SUBJECTS
    # ==========================================

    def get_subjects(self):

        subjects = set()


        for document in self.documents:

            subject = document[
                "metadata"
            ].get(
                "subject",
                "General"
            )


            subjects.add(
                subject
            )


        return sorted(
            subjects,
            key=lambda x: x.lower()
        )


    # ==========================================
    # GET CHAPTERS
    # ==========================================

    def get_chapters(
        self,
        subject=None
    ):

        chapters = set()


        for document in self.documents:

            metadata = document[
                "metadata"
            ]


            if subject is not None:

                if metadata.get(
                    "subject"
                ) != subject:

                    continue


            chapter = metadata.get(
                "chapter",
                "General"
            )


            chapters.add(
                chapter
            )


        # --------------------------------------
        # Numeric chapter sorting
        # --------------------------------------

        def chapter_sort_key(chapter):

            match = re.search(
                r"(?:Chapter\s*)?(\d+)",
                chapter,
                re.IGNORECASE
            )


            if match:

                return (
                    0,
                    int(match.group(1)),
                    chapter.lower()
                )


            return (
                1,
                999999,
                chapter.lower()
            )


        return sorted(
            chapters,
            key=chapter_sort_key
        )