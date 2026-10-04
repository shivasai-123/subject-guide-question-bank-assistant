import os
import json
import re

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


class RAGEngine:

    def __init__(
        self,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    ):

        print()
        print("==========================================")
        print("INITIALIZING RAG ENGINE")
        print("==========================================")

        self.embedding_model_name = embedding_model

        print(
            f"Loading embedding model: "
            f"{embedding_model}"
        )

        self.model = SentenceTransformer(
            embedding_model
        )

        self.documents = []
        self.embeddings = None
        self.index = None

        self.embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        print(
            f"Embedding dimension: "
            f"{self.embedding_dimension}"
        )

        print("RAG engine ready.")
        print("==========================================")

    # =========================================================
    # 1. ADD DOCUMENTS
    # =========================================================

    def add_documents(
        self,
        documents,
        rebuild=True,
    ):

        if not documents:
            return

        self.documents.extend(
            documents
        )

        if rebuild:
            self._rebuild_index()

    # =========================================================
    # 2. REBUILD FAISS INDEX
    # =========================================================

    def _rebuild_index(self):

        if not self.documents:

            self.embeddings = None
            self.index = None

            return

        texts = []

        for document in self.documents:

            texts.append(
                document.get(
                    "text",
                    "",
                )
            )

        print(
            f"Encoding {len(texts)} "
            f"document chunks..."
        )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embeddings = embeddings.astype(
            "float32"
        )

        self.embeddings = embeddings

        self.index = faiss.IndexFlatL2(
            self.embedding_dimension
        )

        self.index.add(
            embeddings
        )

        print(
            f"FAISS index built with "
            f"{self.index.ntotal} vectors."
        )

    # =========================================================
    # 3. TEXT NORMALIZATION
    # =========================================================

    def _normalize_text(
        self,
        text,
    ):

        if not text:
            return ""

        text = text.lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text

    # =========================================================
    # 4. METADATA
    # =========================================================

    def _get_metadata(
        self,
        document,
    ):

        metadata = document.get(
            "metadata",
            {},
        )

        if isinstance(
            metadata,
            dict,
        ):
            return metadata

        return {}

    def _get_content_type(
        self,
        document,
    ):

        metadata = self._get_metadata(
            document
        )

        content_type = (
            document.get(
                "content_type"
            )
            or metadata.get(
                "content_type"
            )
            or "Notes"
        )

        if content_type == "QuestionBank":
            content_type = "Question Bank"

        return content_type

    def _get_filename(
        self,
        document,
    ):

        metadata = self._get_metadata(
            document
        )

        return (
            document.get(
                "filename"
            )
            or metadata.get(
                "filename"
            )
            or metadata.get(
                "source"
            )
            or "Unknown file"
        )

    def _get_subject(
        self,
        document,
    ):

        metadata = self._get_metadata(
            document
        )

        return (
            document.get(
                "subject"
            )
            or metadata.get(
                "subject"
            )
            or "General"
        )

    def _get_chapter(
        self,
        document,
    ):

        metadata = self._get_metadata(
            document
        )

        return (
            document.get(
                "chapter"
            )
            or metadata.get(
                "chapter"
            )
            or "General"
        )

    # =========================================================
    # 5. RESULT OBJECT
    # =========================================================

    def _make_result(
        self,
        document,
        distance,
        index=None,
        lexical_score=0,
    ):

        metadata = dict(
            self._get_metadata(
                document
            )
        )

        filename = self._get_filename(
            document
        )

        subject = self._get_subject(
            document
        )

        chapter = self._get_chapter(
            document
        )

        content_type = self._get_content_type(
            document
        )

        metadata["filename"] = filename
        metadata["subject"] = subject
        metadata["chapter"] = chapter
        metadata["content_type"] = content_type

        return {

            "text": document.get(
                "text",
                "",
            ),

            "filename": filename,

            "subject": subject,

            "chapter": chapter,

            "content_type": content_type,

            "distance": float(
                distance
            ),

            "index": index,

            "lexical_score": lexical_score,

            "metadata": metadata,
        }

    # =========================================================
    # 6. EXAM QUERY DETECTION
    # =========================================================

    def is_exam_query(
        self,
        query,
    ):

        query_lower = (
            query.lower().strip()
        )

        exam_keywords = [

            "solve this question",

            "solve question",

            "answer this question",

            "answer question",

            "write a program",

            "write program",

            "implement",

            "implementation",

            "program to",

            "program for",

            "question",

            "exam",

            "examination",

            "previous year",

            "pyq",

            "question bank",

            "give me the answer",

            "provide solution",

            "solve",
        ]

        return any(
            keyword in query_lower
            for keyword in exam_keywords
        )

    # =========================================================
    # 7. QUERY TERMS
    # =========================================================

    def _query_terms(
        self,
        query,
    ):

        query_lower = (
            query.lower().strip()
        )

        aliases = {

            "bfs": [
                "bfs",
                "breadth first search",
                "breadth-first search",
                "breadth first",
            ],

            "dfs": [
                "dfs",
                "depth first search",
                "depth-first search",
                "depth first",
            ],

            "tuple": [
                "tuple",
                "tuples",
            ],

            "tuple packing": [
                "tuple packing",
            ],

            "tuple unpacking": [
                "tuple unpacking",
            ],

            "list": [
                "list",
                "lists",
            ],

            "dictionary": [
                "dictionary",
                "dictionaries",
            ],

            "set": [
                "set",
                "sets",
            ],

            "8 queens": [
                "8 queens",
                "8-queens",
                "eight queens",
            ],

            "water jug": [
                "water jug",
                "water-jug",
            ],

            "hill climbing": [
                "hill climbing",
                "hill-climbing",
            ],

            "tower of hanoi": [
                "tower of hanoi",
                "tower-of-hanoi",
            ],

            "alpha beta": [
                "alpha beta",
                "alpha-beta",
                "alpha beta pruning",
                "alpha-beta pruning",
            ],

            "apriori": [
                "apriori",
                "apriori algorithm",
            ],

            # Core Algorithms
            "binary search": [
                "binary search",
                "binary-search",
                "sorted array",
                "divide and conquer",
            ],

            "linear search": [
                "linear search",
                "sequential search",
            ],

            # DBMS / SQL
            "normalization": [
                "normalization",
                "database normalization",
                "1nf",
                "2nf",
                "3nf",
                "bcnf",
                "normal form",
            ],

            "acid": [
                "acid",
                "acid properties",
                "atomicity",
                "consistency",
                "isolation",
                "durability",
            ],

            "sql": [
                "sql",
                "structured query language",
                "select",
                "insert",
                "create table",
                "join",
            ],

            "transaction": [
                "transaction",
                "transactions",
                "concurrency control",
                "serializability",
            ],

            # Operating Systems
            "process": [
                "process",
                "processes",
                "process management",
                "pcb",
                "context switch",
            ],

            "deadlock": [
                "deadlock",
                "deadlocks",
                "banker's algorithm",
                "mutual exclusion",
                "resource allocation",
            ],

            "cpu scheduling": [
                "cpu scheduling",
                "scheduling algorithm",
                "round robin",
                "fcfs",
                "sjf",
                "priority scheduling",
            ],

            "paging": [
                "paging",
                "page replacement",
                "virtual memory",
                "page table",
                "segmentation",
            ],

            # Computer Networks
            "osi": [
                "osi",
                "osi model",
                "osi layers",
                "physical layer",
                "data link",
                "transport layer",
            ],

            "tcp": [
                "tcp",
                "tcp/ip",
                "transmission control protocol",
                "three-way handshake",
            ],

            "udp": [
                "udp",
                "user datagram protocol",
            ],

            "routing": [
                "routing",
                "routing algorithm",
                "distance vector",
                "link state",
            ],
        }

        phrases = []

        for alias, variations in aliases.items():

            if alias in query_lower:

                phrases.extend(
                    variations
                )

        for variation_list in aliases.values():

            for variation in variation_list:

                if variation in query_lower:

                    phrases.append(
                        variation
                    )

        stop_words = {

            "a",
            "an",
            "the",

            "is",
            "are",
            "was",
            "were",

            "what",
            "why",
            "how",
            "when",
            "where",
            "which",

            "write",
            "program",
            "code",

            "solve",
            "solution",

            "implement",
            "implementation",

            "explain",
            "explanation",

            "give",
            "me",

            "this",
            "that",

            "question",
            "questions",

            "using",
            "use",

            "with",

            "for",
            "to",
            "of",
            "in",
            "on",
            "about",

            "related",
            "answer",

            "please",
            "provide",

            "teach",
            "learn",
            "study",
        }

        words = re.findall(
            r"[A-Za-z0-9]+",
            query_lower,
        )

        useful_words = [

            word

            for word in words

            if (
                word not in stop_words
                and len(word) >= 2
            )
        ]

        return {

            "phrases": list(
                dict.fromkeys(
                    phrases
                )
            ),

            "words": set(
                useful_words
            ),
        }

    # =========================================================
    # 8. LEXICAL SCORE
    # =========================================================

    def _lexical_score(
        self,
        query,
        document,
    ):

        query_lower = (
            query.lower().strip()
        )

        text = self._normalize_text(
            document.get(
                "text",
                "",
            )
        )

        content_type = (
            self._get_content_type(
                document
            ).lower()
        )

        term_info = self._query_terms(
            query
        )

        phrases = term_info[
            "phrases"
        ]

        words = term_info[
            "words"
        ]

        score = 0

        # -----------------------------------------------------
        # Exact technical phrases
        # -----------------------------------------------------

        for phrase in phrases:

            phrase_lower = (
                phrase.lower()
            )

            if phrase_lower in text:

                score += (
                    12
                    + min(
                        len(
                            phrase.split()
                        ),
                        5,
                    )
                    * 2
                )

        # -----------------------------------------------------
        # Exact query
        # -----------------------------------------------------

        if (
            query_lower
            and query_lower in text
        ):

            score += 20

        # -----------------------------------------------------
        # Individual useful words
        # -----------------------------------------------------

        for word in words:

            if re.search(
                rf"\b{re.escape(word)}\b",
                text,
            ):

                score += 2

        # -----------------------------------------------------
        # Question-bank behavior
        # -----------------------------------------------------

        if content_type == "question bank":

            if self.is_exam_query(
                query
            ):

                score += 8

        return score

    # =========================================================
    # 9. STRONG TOPIC
    # =========================================================

    def _strong_topic(
        self,
        query,
    ):

        query_lower = (
            query.lower().strip()
        )

        strong_topics = [

            "bfs",
            "breadth first search",

            "dfs",
            "depth first search",

            "tuple packing",
            "tuple unpacking",
            "tuple",

            "dictionary",

            "8 queens",

            "water jug",

            "hill climbing",

            "tower of hanoi",

            "alpha beta",

            "binary search",

            "linear search",

            "normalization",

            "acid",

            "deadlock",

            "cpu scheduling",

            "osi",

            "tcp",
        ]

        # Prefer the longest exact topic phrase.
        matches = [

            topic

            for topic in strong_topics

            if topic in query_lower
        ]

        if matches:

            return max(
                matches,
                key=len,
            )

        term_info = self._query_terms(
            query
        )

        phrases = term_info[
            "phrases"
        ]

        if phrases:

            return max(
                phrases,
                key=len,
            )

        return None

    # =========================================================
    # 10. STRONG TOPIC SUBSTANTIVE MATCH
    # =========================================================

    def _strong_topic_substantive_match(
        self,
        strong_topic,
        document,
    ):

        text = self._normalize_text(
            document.get(
                "text",
                "",
            )
        )

        topic = (
            strong_topic
            .lower()
            .strip()
        )

        # -----------------------------------------------------
        # BFS / DFS need stronger evidence.
        #
        # A single "DFS" in a comparison table is not enough
        # to treat the entire chunk as DFS study material.
        # -----------------------------------------------------

        if topic in {
            "bfs",
            "breadth first search",
        }:

            short_count = len(
                re.findall(
                    r"\bbfs\b",
                    text,
                    re.IGNORECASE,
                )
            )

            long_count = len(
                re.findall(
                    r"\bbreadth[- ]first search\b",
                    text,
                    re.IGNORECASE,
                )
            )

            return (
                long_count >= 1
                or short_count >= 2
            )

        if topic in {
            "dfs",
            "depth first search",
        }:

            short_count = len(
                re.findall(
                    r"\bdfs\b",
                    text,
                    re.IGNORECASE,
                )
            )

            long_count = len(
                re.findall(
                    r"\bdepth[- ]first search\b",
                    text,
                    re.IGNORECASE,
                )
            )

            return (
                long_count >= 1
                or short_count >= 2
            )

        # -----------------------------------------------------
        # Other strong topics only need one exact appearance.
        # -----------------------------------------------------

        aliases = {
            "tuple": [
                "tuple",
                "tuples",
            ],

            "tuple packing": [
                "tuple packing",
            ],

            "tuple unpacking": [
                "tuple unpacking",
            ],

            "dictionary": [
                "dictionary",
                "dictionaries",
            ],

            "8 queens": [
                "8 queens",
                "8-queens",
                "eight queens",
            ],

            "water jug": [
                "water jug",
                "water-jug",
            ],

            "hill climbing": [
                "hill climbing",
                "hill-climbing",
            ],

            "tower of hanoi": [
                "tower of hanoi",
                "tower-of-hanoi",
            ],

            "alpha beta": [
                "alpha beta",
                "alpha-beta",
                "alpha beta pruning",
                "alpha-beta pruning",
            ],
        }

        topic_aliases = aliases.get(
            topic,
            [topic],
        )

        return any(
            alias in text
            for alias in topic_aliases
        )

    # =========================================================
    # 11. CANDIDATE FILTER
    # =========================================================

    def _candidate_indices(
        self,
        subject=None,
        chapter=None,
    ):

        indices = []

        for i, document in enumerate(
            self.documents
        ):

            document_subject = (
                self._get_subject(
                    document
                )
            )

            document_chapter = (
                self._get_chapter(
                    document
                )
            )

            if (
                subject
                and document_subject.lower()
                != subject.lower()
            ):

                continue

            if (
                chapter
                and document_chapter.lower()
                != chapter.lower()
            ):

                continue

            indices.append(i)

        return indices

    # =========================================================
    # 12. RETRIEVE
    # =========================================================

    def retrieve(
        self,
        query,
        k=5,
        subject=None,
        chapter=None,
    ):

        if not self.documents:
            return []

        query = query.strip()

        if not query:
            return []

        candidate_indices = (
            self._candidate_indices(
                subject=subject,
                chapter=chapter,
            )
        )

        if not candidate_indices:
            return []

        # -----------------------------------------------------
        # Query embedding
        # -----------------------------------------------------

        query_embedding = (
            self.model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            .astype("float32")
        )

        # -----------------------------------------------------
        # Temporary FAISS search over filtered documents
        # -----------------------------------------------------

        candidate_vectors = (
            self.embeddings[
                candidate_indices
            ]
        )

        temp_index = faiss.IndexFlatL2(
            self.embedding_dimension
        )

        temp_index.add(
            candidate_vectors
        )

        search_k = min(
            max(
                k * 5,
                20,
            ),
            len(candidate_indices),
        )

        distances, local_indices = (
            temp_index.search(
                query_embedding,
                search_k,
            )
        )

        candidates = []

        for distance, local_index in zip(
            distances[0],
            local_indices[0],
        ):

            if local_index < 0:
                continue

            actual_index = (
                candidate_indices[
                    int(local_index)
                ]
            )

            document = self.documents[
                actual_index
            ]

            lexical_score = (
                self._lexical_score(
                    query,
                    document,
                )
            )

            candidates.append({

                "document": document,

                "index": actual_index,

                "distance": float(
                    distance
                ),

                "lexical_score":
                    lexical_score,
            })

        if not candidates:
            return []

        # -----------------------------------------------------
        # Detect strong topic
        # -----------------------------------------------------

        exam_query = self.is_exam_query(
            query
        )

        strong_topic = (
            self._strong_topic(
                query
            )
        )

        # -----------------------------------------------------
        # Score candidates
        # -----------------------------------------------------

        for candidate in candidates:

            document = candidate[
                "document"
            ]

            content_type = (
                self._get_content_type(
                    document
                ).lower()
            )

            lexical = candidate[
                "lexical_score"
            ]

            score = lexical

            semantic_score = (
                1.0
                /
                (
                    1.0
                    + candidate["distance"]
                )
            )

            score += (
                semantic_score * 5
            )

            # -------------------------------------------------
            # Strong topic bonus
            # -------------------------------------------------

            if strong_topic:

                strong_topic_lower = (
                    strong_topic.lower()
                )

                text = self._normalize_text(
                    document.get(
                        "text",
                        "",
                    )
                )

                if (
                    strong_topic_lower
                    in text
                ):

                    score += 30

                # BFS aliases
                if strong_topic_lower == "bfs":

                    if (
                        "breadth first search"
                        in text
                    ):

                        score += 35

                # Breadth First Search aliases
                if (
                    strong_topic_lower
                    == "breadth first search"
                ):

                    if "bfs" in text:

                        score += 35

                # DFS aliases
                if strong_topic_lower == "dfs":

                    if (
                        "depth first search"
                        in text
                    ):

                        score += 35

                if (
                    strong_topic_lower
                    == "depth first search"
                ):

                    if "dfs" in text:

                        score += 35

            # -------------------------------------------------
            # Question bank handling
            # -------------------------------------------------

            if content_type == "question bank":

                if exam_query:

                    score += 12

                else:

                    score -= 8

            # -------------------------------------------------
            # Study material preference
            # -------------------------------------------------

            if (
                not exam_query
                and content_type
                != "question bank"
            ):

                score += 3

            candidate[
                "final_score"
            ] = score

        # -----------------------------------------------------
        # Sort
        # -----------------------------------------------------

        candidates.sort(
            key=lambda item: (
                item[
                    "final_score"
                ],
                -item[
                    "distance"
                ],
            ),
            reverse=True,
        )

        # =====================================================
        # IMPORTANT PRECISION GATE
        # =====================================================
        #
        # For a strong topic:
        #
        #   genuine topic material exists
        #       -> use it
        #
        #   no genuine topic material exists
        #       -> return []
        #
        # We DO NOT fall back to unrelated semantic results.
        # =====================================================

        if strong_topic:

            strong_matches = []

            for candidate in candidates:

                if (
                    candidate[
                        "lexical_score"
                    ] <= 0
                ):

                    continue

                if not self._strong_topic_substantive_match(
                    strong_topic,
                    candidate[
                        "document"
                    ],
                ):

                    continue

                strong_matches.append(
                    candidate
                )

            if strong_matches:

                candidates = (
                    strong_matches
                )

            else:

                return []

        # -----------------------------------------------------
        # Return top K
        # -----------------------------------------------------

        top_candidates = candidates[
            :max(
                1,
                k,
            )
        ]

        results = []

        for candidate in top_candidates:

            result = self._make_result(

                document=candidate[
                    "document"
                ],

                distance=candidate[
                    "distance"
                ],

                index=candidate[
                    "index"
                ],

                lexical_score=candidate[
                    "lexical_score"
                ],
            )

            result[
                "score"
            ] = float(
                candidate[
                    "final_score"
                ]
            )

            results.append(
                result
            )

        return results

    # =========================================================
    # 13. ALL QUESTION BANK DOCUMENTS
    # =========================================================

    def get_all_question_bank_documents(
        self,
        subject=None,
        chapter=None,
    ):

        results = []

        for index, document in enumerate(
            self.documents
        ):

            content_type = (
                self._get_content_type(
                    document
                )
            )

            if (
                content_type.lower()
                != "question bank"
            ):

                continue

            document_subject = (
                self._get_subject(
                    document
                )
            )

            document_chapter = (
                self._get_chapter(
                    document
                )
            )

            if (
                subject
                and document_subject.lower()
                != subject.lower()
            ):

                continue

            if (
                chapter
                and document_chapter.lower()
                != chapter.lower()
            ):

                continue

            result = self._make_result(
                document=document,
                distance=0.0,
                index=index,
                lexical_score=0,
            )

            results.append(
                result
            )

        return results

    # =========================================================
    # 14. TOPIC QUESTION BANK DOCUMENTS
    # =========================================================

    def get_topic_question_bank_documents(
        self,
        topic,
        subject=None,
        chapter=None,
        limit=10,
    ):

        if not topic:
            return []

        topic = topic.strip()

        question_bank_documents = []

        candidate_indices = (
            self._candidate_indices(
                subject=subject,
                chapter=chapter,
            )
        )

        for index in candidate_indices:

            document = self.documents[
                index
            ]

            content_type = (
                self._get_content_type(
                    document
                )
            )

            if (
                content_type.lower()
                != "question bank"
            ):

                continue

            text = self._normalize_text(
                document.get(
                    "text",
                    "",
                )
            )

            lexical_score = (
                self._lexical_score(
                    topic,
                    document,
                )
            )

            topic_lower = (
                topic.lower()
            )

            extra_score = 0

            if topic_lower == "bfs":

                if (
                    "bfs" in text
                    or "breadth first search"
                    in text
                    or "breadth-first search"
                    in text
                ):

                    extra_score += 100

            elif topic_lower == "dfs":

                if (
                    "dfs" in text
                    or "depth first search"
                    in text
                    or "depth-first search"
                    in text
                ):

                    extra_score += 100

            elif topic_lower in text:

                extra_score += 80

            final_score = (
                lexical_score
                + extra_score
            )

            if final_score > 0:

                question_bank_documents.append({

                    "document":
                        document,

                    "index":
                        index,

                    "score":
                        final_score,

                    "distance":
                        0.0,
                })

        # -----------------------------------------------------
        # Sort exact topic question-bank results
        # -----------------------------------------------------

        question_bank_documents.sort(
            key=lambda item: (
                item["score"],
                -item["distance"],
            ),
            reverse=True,
        )

        results = []

        for item in question_bank_documents[
            :max(
                1,
                limit,
            )
        ]:

            result = self._make_result(

                document=item[
                    "document"
                ],

                distance=item[
                    "distance"
                ],

                index=item[
                    "index"
                ],

                lexical_score=item[
                    "score"
                ],
            )

            result[
                "score"
            ] = float(
                item[
                    "score"
                ]
            )

            results.append(
                result
            )

        return results

    # =========================================================
    # 15. BUILD CONTEXT
    # =========================================================

    def build_context(
        self,
        results,
    ):

        if not results:
            return ""

        context_parts = []

        for i, result in enumerate(
            results,
            start=1,
        ):

            metadata = result.get(
                "metadata",
                {},
            )

            filename = (
                result.get(
                    "filename"
                )
                or metadata.get(
                    "filename",
                    "Unknown file",
                )
            )

            subject = (
                result.get(
                    "subject"
                )
                or metadata.get(
                    "subject",
                    "General",
                )
            )

            chapter = (
                result.get(
                    "chapter"
                )
                or metadata.get(
                    "chapter",
                    "General",
                )
            )

            content_type = (
                result.get(
                    "content_type"
                )
                or metadata.get(
                    "content_type",
                    "Notes",
                )
            )

            text = result.get(
                "text",
                "",
            ).strip()

            if not text:
                continue

            context_parts.append(

                f"[Source {i}]\n"
                f"File: {filename}\n"
                f"Subject: {subject}\n"
                f"Chapter: {chapter}\n"
                f"Content Type: {content_type}\n"
                f"Content:\n{text}"
            )

        return "\n\n---\n\n".join(
            context_parts
        )

    # =========================================================
    # 16. DOCUMENT COUNT
    # =========================================================

    def get_document_count(self):

        filenames = set()

        for document in self.documents:

            filenames.add(
                self._get_filename(
                    document
                )
            )

        return len(
            filenames
        )

    # =========================================================
    # 17. VECTOR COUNT
    # =========================================================

    def get_vector_count(self):

        if self.index is None:
            return 0

        return int(
            self.index.ntotal
        )

    # =========================================================
    # 18. SUBJECTS
    # =========================================================

    def get_subjects(self):

        subjects = set()

        for document in self.documents:

            subjects.add(
                self._get_subject(
                    document
                )
            )

        return sorted(
            subjects,
            key=lambda item: item.lower(),
        )

    # =========================================================
    # 19. CHAPTERS
    # =========================================================

    @staticmethod
    def _chapter_sort_key(chapter_name):
        clean_name = str(chapter_name).strip()
        if clean_name.lower() == "general":
            return (2, [])

        tokens = []
        for part in re.split(r"(\d+)", clean_name):
            if not part:
                continue
            if part.isdigit():
                tokens.append((0, int(part), ""))
            else:
                tokens.append((1, 0, part.lower()))

        return (0, tokens)

    def get_chapters(
        self,
        subject=None,
    ):

        chapters = set()

        for document in self.documents:

            document_subject = (
                self._get_subject(
                    document
                )
            )

            if (
                subject
                and document_subject.lower()
                != subject.lower()
            ):

                continue

            chapters.add(
                self._get_chapter(
                    document
                )
            )

        return sorted(
            chapters,
            key=self._chapter_sort_key,
        )

    # =========================================================
    # 20. INDEX PERSISTENCE & CACHING
    # =========================================================

    def save_cache(self, cache_dir):
        """Save FAISS index, embeddings, and documents to cache directory."""
        if self.index is None or not self.documents:
            return False
        os.makedirs(cache_dir, exist_ok=True)
        index_file = os.path.join(cache_dir, "faiss.index")
        embeddings_file = os.path.join(cache_dir, "embeddings.npy")
        docs_file = os.path.join(cache_dir, "documents.json")
        try:
            faiss.write_index(self.index, index_file)
            if self.embeddings is not None:
                np.save(embeddings_file, self.embeddings)
            with open(docs_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False)
            print(f"RAG cache successfully saved to {cache_dir}")
            return True
        except Exception as e:
            print(f"Warning: Failed to save RAG cache: {e}")
            return False

    def load_cache(self, cache_dir):
        """Load FAISS index, embeddings, and documents from cache directory."""
        index_file = os.path.join(cache_dir, "faiss.index")
        embeddings_file = os.path.join(cache_dir, "embeddings.npy")
        docs_file = os.path.join(cache_dir, "documents.json")
        if not (os.path.exists(index_file) and os.path.exists(docs_file)):
            return False
        try:
            self.index = faiss.read_index(index_file)
            if os.path.exists(embeddings_file):
                self.embeddings = np.load(embeddings_file)
            with open(docs_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            print(f"Loaded RAG cache: {len(self.documents)} chunks, {self.index.ntotal} vectors.")
            return True
        except Exception as e:
            print(f"Warning: Failed to load RAG cache: {e}")
            return False