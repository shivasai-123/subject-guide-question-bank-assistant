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
    # DETECT QUESTION BANK / EXAM QUERY
    # ==========================================

    def is_exam_query(self, query):

        query_lower = query.lower()

        exam_keywords = [

            "exam",
            "examination",

            "important question",
            "important questions",

            "question bank",

            "previous year",
            "previous-year",

            "pyq",

            "prepare",
            "preparation",

            "marks",
            "mark",

            "long answer",
            "short answer",

            "2 marks",
            "5 marks",
            "10 marks",
            "16 marks",

            "viva",

            "practice question",
            "practice questions",

            "model question",
            "model questions",

            "expected question",
            "expected questions",

            "external question bank",

            "external questions",

            "questions included",

            "questions are included",

            "programs included",

            "programs are included"
        ]

        for keyword in exam_keywords:

            if keyword in query_lower:
                return True

        return False


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


        # ======================================
        # CREATE QUERY EMBEDDING
        # ======================================

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype(
            "float32"
        )


        # ======================================
        # FIND CANDIDATE DOCUMENTS
        # ======================================

        candidate_indices = []

        for i, document in enumerate(
            self.documents
        ):

            metadata = document["metadata"]


            # ----------------------------------
            # SUBJECT FILTER
            # ----------------------------------

            if (
                subject is not None
                and subject != "All Subjects"
            ):

                if metadata.get(
                    "subject"
                ) != subject:

                    continue


            # ----------------------------------
            # CHAPTER FILTER
            # ----------------------------------

            if (
                chapter is not None
                and chapter != "All Chapters"
            ):

                if metadata.get(
                    "chapter"
                ) != chapter:

                    continue


            candidate_indices.append(
                i
            )


        # ======================================
        # NO CANDIDATES
        # ======================================

        if not candidate_indices:
            return []


        # ======================================
        # NORMAL SEMANTIC SEARCH
        # ======================================

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


        # Search a larger candidate pool
        candidate_count = min(
            max(k * 10, 50),
            len(candidate_indices)
        )


        distances, indices = (
            filtered_index.search(
                query_embedding,
                candidate_count
            )
        )


        # ======================================
        # DETECT EXAM QUERY
        # ======================================

        exam_query = self.is_exam_query(
            query
        )


        results = []


        # ======================================
        # ADD NORMAL SEARCH RESULTS
        # ======================================

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


            metadata = document[
                "metadata"
            ]


            content_type = metadata.get(
                "content_type",
                "Notes"
            )


            original_distance = float(
                distance
            )


            rank_score = (
                original_distance
            )


            # ----------------------------------
            # QUESTION BANK BOOST
            # ----------------------------------

            if (
                exam_query
                and content_type == "Question Bank"
            ):

                rank_score -= 0.75


            results.append(
                {
                    "text":
                        document["text"],

                    "distance":
                        original_distance,

                    "rank_score":
                        rank_score,

                    "metadata":
                        metadata
                }
            )


        # ======================================
        # DIRECT QUESTION BANK SEARCH
        # ======================================

        if exam_query:

            question_bank_indices = []

            for i in candidate_indices:

                metadata = self.documents[
                    i
                ]["metadata"]

                if metadata.get(
                    "content_type",
                    "Notes"
                ) == "Question Bank":

                    question_bank_indices.append(
                        i
                    )


            # ----------------------------------
            # Search Question Bank separately
            # ----------------------------------

            if question_bank_indices:

                question_bank_embeddings = (
                    self.embeddings[
                        question_bank_indices
                    ]
                )


                question_bank_index = (
                    faiss.IndexFlatL2(
                        question_bank_embeddings.shape[1]
                    )
                )


                question_bank_index.add(
                    question_bank_embeddings
                )


                question_bank_k = min(
                    max(k, 5),
                    len(question_bank_indices)
                )


                q_distances, q_indices = (
                    question_bank_index.search(
                        query_embedding,
                        question_bank_k
                    )
                )


                for distance, q_idx in zip(
                    q_distances[0],
                    q_indices[0]
                ):

                    if q_idx < 0:
                        continue


                    original_idx = (
                        question_bank_indices[
                            q_idx
                        ]
                    )


                    document = self.documents[
                        original_idx
                    ]


                    metadata = document[
                        "metadata"
                    ]


                    original_distance = float(
                        distance
                    )


                    # Stronger boost for direct
                    # Question Bank results

                    rank_score = (
                        original_distance - 0.75
                    )


                    results.append(
                        {
                            "text":
                                document["text"],

                            "distance":
                                original_distance,

                            "rank_score":
                                rank_score,

                            "metadata":
                                metadata
                        }
                    )


        # ======================================
        # REMOVE DUPLICATES
        # ======================================

        unique_results = {}


        for result in results:

            metadata = result[
                "metadata"
            ]


            key = (
                metadata.get(
                    "filename",
                    ""
                ),

                metadata.get(
                    "chapter",
                    ""
                ),

                result["text"]
            )


            if (
                key not in unique_results
                or
                result["rank_score"]
                <
                unique_results[key][
                    "rank_score"
                ]
            ):

                unique_results[key] = result


        results = list(
            unique_results.values()
        )


        # ======================================
        # SORT BY FINAL SCORE
        # ======================================

        results.sort(
            key=lambda result:
                result["rank_score"]
        )


        # ======================================
        # RETURN TOP K
        # ======================================

        return results[:k]


    # ==========================================
    # GET ALL QUESTION BANK DOCUMENTS
    # ==========================================

    def get_all_question_bank_documents(
        self,
        subject=None,
        chapter=None
    ):

        results = []


        for document in self.documents:

            metadata = document[
                "metadata"
            ]


            # ----------------------------------
            # SUBJECT FILTER
            # ----------------------------------

            if (
                subject is not None
                and subject != "All Subjects"
            ):

                if metadata.get(
                    "subject"
                ) != subject:

                    continue


            # ----------------------------------
            # CHAPTER FILTER
            # ----------------------------------

            if (
                chapter is not None
                and chapter != "All Chapters"
            ):

                if metadata.get(
                    "chapter"
                ) != chapter:

                    continue


            # ----------------------------------
            # QUESTION BANK ONLY
            # ----------------------------------

            if metadata.get(
                "content_type",
                "Notes"
            ) != "Question Bank":

                continue


            results.append(
                {
                    "text":
                        document["text"],

                    "distance":
                        0.0,

                    "rank_score":
                        0.0,

                    "metadata":
                        metadata
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
Content Type: {metadata.get("content_type", "Notes")}

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
            key=lambda x:
                x.lower()
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


            # ----------------------------------
            # SUBJECT FILTER
            # ----------------------------------

            if (
                subject is not None
                and subject != "All Subjects"
            ):

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


        # ======================================
        # NUMERIC CHAPTER SORTING
        # ======================================

        def chapter_sort_key(chapter):

            match = re.search(
                r"(?:Chapter\s*)?(\d+)",
                chapter,
                re.IGNORECASE
            )


            if match:

                return (
                    0,
                    int(
                        match.group(1)
                    ),
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