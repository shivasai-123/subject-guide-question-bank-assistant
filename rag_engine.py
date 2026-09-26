import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGEngine:

    # ==========================================
    # TECHNICAL TOPIC ALIASES
    # ==========================================

    TOPIC_ALIASES = {
        "bfs": [
            "bfs",
            "breadth first search",
            "breadth-first search",
        ],

        "dfs": [
            "dfs",
            "depth first search",
            "depth-first search",
        ],

        "tuple": [
            "tuple",
        ],

        "tuple packing": [
            "tuple packing",
        ],

        "tuple unpacking": [
            "tuple unpacking",
        ],

        "8 queens": [
            "8 queens",
            "8-queens",
            "eight queens",
        ],

        "water jug": [
            "water jug",
        ],

        "hill climbing": [
            "hill climbing",
        ],

        "tower of hanoi": [
            "tower of hanoi",
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
    }

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(
        self,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    ):

        print()
        print("==========================================")
        print("INITIALIZING RAG ENGINE")
        print("==========================================")

        self.embedding_model_name = (
            embedding_model
        )

        # --------------------------------------
        # Load embedding model
        # --------------------------------------

        print(
            f"Loading embedding model: "
            f"{embedding_model}"
        )

        self.model = SentenceTransformer(
            embedding_model
        )

        # --------------------------------------
        # Storage
        # --------------------------------------

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

        print(
            "RAG engine ready."
        )

        print(
            "=========================================="
        )

    # ==========================================
    # ADD DOCUMENTS
    # ==========================================

    def add_documents(
        self,
        documents,
        rebuild=True
    ):

        if not documents:
            return

        self.documents.extend(
            documents
        )

        if rebuild:
            self._rebuild_index()

    # ==========================================
    # REBUILD FAISS INDEX
    # ==========================================

    def _rebuild_index(self):

        if not self.documents:

            self.embeddings = None
            self.index = None

            return

        texts = []

        for document in self.documents:

            text = document.get(
                "text",
                ""
            )

            texts.append(
                text
            )

        print(
            f"Encoding {len(texts)} "
            f"document chunks..."
        )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
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

    # ==========================================
    # TEXT NORMALIZATION
    # ==========================================

    def _normalize_text(
        self,
        text
    ):

        if not text:
            return ""

        text = text.lower()

        text = text.replace(
            "\u2010",
            "-"
        )

        text = text.replace(
            "\u2011",
            "-"
        )

        text = text.replace(
            "\u2012",
            "-"
        )

        text = text.replace(
            "\u2013",
            "-"
        )

        text = text.replace(
            "\u2014",
            "-"
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    # ==========================================
    # METADATA HELPERS
    # ==========================================

    def _get_metadata(
        self,
        document
    ):

        metadata = document.get(
            "metadata",
            {}
        )

        if isinstance(
            metadata,
            dict
        ):

            return metadata

        return {}

    def _get_filename(
        self,
        document
    ):

        metadata = self._get_metadata(
            document
        )

        return (
            document.get("filename")
            or metadata.get("filename")
            or metadata.get("source")
            or "Unknown file"
        )

    def _get_subject(
        self,
        document
    ):

        metadata = self._get_metadata(
            document
        )

        return (
            document.get("subject")
            or metadata.get("subject")
            or "General"
        )

    def _get_chapter(
        self,
        document
    ):

        metadata = self._get_metadata(
            document
        )

        return (
            document.get("chapter")
            or metadata.get("chapter")
            or "General"
        )

    def _get_content_type(
        self,
        document
    ):

        metadata = self._get_metadata(
            document
        )

        content_type = (
            document.get("content_type")
            or metadata.get("content_type")
            or "Notes"
        )

        if content_type == "QuestionBank":
            content_type = "Question Bank"

        return content_type

    # ==========================================
    # BUILD RESULT
    # ==========================================

    def _make_result(
        self,
        document,
        distance,
        index=None,
        lexical_score=0
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

        # Keep metadata complete
        metadata["filename"] = filename
        metadata["subject"] = subject
        metadata["chapter"] = chapter
        metadata["content_type"] = content_type

        return {

            "text":
                document.get(
                    "text",
                    ""
                ),

            "filename":
                filename,

            "subject":
                subject,

            "chapter":
                chapter,

            "content_type":
                content_type,

            "distance":
                float(distance),

            "index":
                index,

            "lexical_score":
                lexical_score,

            "metadata":
                metadata
        }

    # ==========================================
    # EXAM / PROGRAMMING QUERY DETECTION
    # ==========================================

    def is_exam_query(
        self,
        query
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

            "provide solution",

            "give me the answer",

            "solve"
        ]

        return any(
            keyword in query_lower
            for keyword in exam_keywords
        )

    # ==========================================
    # CANONICAL TOPIC
    # ==========================================

    def _canonical_topic(
        self,
        query
    ):

        query_lower = (
            self._normalize_text(
                query
            )
        )

        # Longer phrases first
        topic_checks = [

            ("breadth first search", "bfs"),
            ("breadth-first search", "bfs"),
            ("bfs", "bfs"),

            ("depth first search", "dfs"),
            ("depth-first search", "dfs"),
            ("dfs", "dfs"),

            ("tuple unpacking", "tuple unpacking"),

            ("tuple packing", "tuple packing"),

            ("8-queens", "8 queens"),
            ("8 queens", "8 queens"),
            ("eight queens", "8 queens"),

            ("water jug", "water jug"),

            ("hill climbing", "hill climbing"),

            ("tower of hanoi", "tower of hanoi"),

            ("alpha-beta pruning", "alpha beta"),
            ("alpha beta pruning", "alpha beta"),
            ("alpha-beta", "alpha beta"),
            ("alpha beta", "alpha beta"),

            ("apriori algorithm", "apriori"),
            ("apriori", "apriori"),

            ("tuple", "tuple")
        ]

        # Longest/most specific patterns first
        topic_checks.sort(
            key=lambda item: len(item[0]),
            reverse=True
        )

        for phrase, canonical in topic_checks:

            if phrase in query_lower:

                return canonical

        return None

    # ==========================================
    # TOPIC ALIASES FOR CANONICAL TOPIC
    # ==========================================

    def _get_topic_aliases(
        self,
        canonical_topic
    ):

        return self.TOPIC_ALIASES.get(
            canonical_topic,
            [canonical_topic]
        )

    # ==========================================
    # EXACT STRONG-TOPIC MATCH
    # ==========================================

    def _matches_strong_topic(
        self,
        document,
        canonical_topic
    ):

        if not canonical_topic:
            return False

        text = self._normalize_text(
            document.get(
                "text",
                ""
            )
        )

        aliases = self._get_topic_aliases(
            canonical_topic
        )

        for alias in aliases:

            alias_normalized = (
                self._normalize_text(
                    alias
                )
            )

            if not alias_normalized:
                continue

            if alias_normalized in text:
                return True

        return False

    # ==========================================
    # QUERY WORDS
    # ==========================================

    def _query_terms(
        self,
        query
    ):

        query_lower = self._normalize_text(
            query
        )

        # --------------------------------------
        # If this is a strong technical topic,
        # don't use generic words such as
        # "first" or "search".
        # --------------------------------------

        canonical_topic = (
            self._canonical_topic(
                query
            )
        )

        if canonical_topic:

            aliases = self._get_topic_aliases(
                canonical_topic
            )

            useful_phrases = list(
                dict.fromkeys(
                    aliases
                )
            )

            return {

                "phrases":
                    useful_phrases,

                "words":
                    set()
            }

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

            "provide"
        }

        words = re.findall(
            r"[A-Za-z0-9]+",
            query_lower
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

            "phrases":
                [],

            "words":
                set(
                    useful_words
                )
        }

    # ==========================================
    # LEXICAL RELEVANCE
    # ==========================================

    def _lexical_score(
        self,
        query,
        document
    ):

        text = self._normalize_text(
            document.get(
                "text",
                ""
            )
        )

        if not text:
            return 0

        content_type = (
            self._get_content_type(
                document
            ).lower()
        )

        # --------------------------------------
        # STRONG TOPIC
        # --------------------------------------

        canonical_topic = (
            self._canonical_topic(
                query
            )
        )

        if canonical_topic:

            if self._matches_strong_topic(
                document,
                canonical_topic
            ):

                return 100

            # CRITICAL:
            # If this is BFS, a DFS document
            # gets ZERO score.
            return 0

        # --------------------------------------
        # NORMAL QUERY
        # --------------------------------------

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

        # Phrase match
        for phrase in phrases:

            phrase_lower = (
                self._normalize_text(
                    phrase
                )
            )

            if phrase_lower in text:

                score += (
                    10
                    + min(
                        len(
                            phrase_lower.split()
                        ),
                        5
                    ) * 2
                )

        # Exact complete query
        normalized_query = (
            self._normalize_text(
                query
            )
        )

        if (
            normalized_query
            and normalized_query in text
        ):

            score += 20

        # Individual words
        for word in words:

            if re.search(
                rf"\b{re.escape(word)}\b",
                text
            ):

                score += 2

        # Question-bank handling
        if content_type == "question bank":

            if self.is_exam_query(query):

                score += 8

            else:

                score -= 8

        return score

    # ==========================================
    # FILTER BY SUBJECT / CHAPTER
    # ==========================================

    def _candidate_indices(
        self,
        subject=None,
        chapter=None
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

    # ==========================================
    # RETRIEVE
    # ==========================================

    def retrieve(
        self,
        query,
        k=5,
        subject=None,
        chapter=None
    ):

        if not self.documents:
            return []

        query = query.strip()

        if not query:
            return []

        candidate_indices = (
            self._candidate_indices(
                subject=subject,
                chapter=chapter
            )
        )

        if not candidate_indices:
            return []

        # ======================================
        # DETECT STRONG TOPIC
        # ======================================

        canonical_topic = (
            self._canonical_topic(
                query
            )
        )

        # ======================================
        # STRICT TOPIC MODE
        # ======================================

        if canonical_topic:

            exact_indices = [

                index

                for index in candidate_indices

                if self._matches_strong_topic(
                    self.documents[index],
                    canonical_topic
                )
            ]

            # ----------------------------------
            # If exact matches exist, ONLY use
            # those documents.
            # ----------------------------------

            if exact_indices:

                candidate_indices = (
                    exact_indices
                )

        # ======================================
        # CREATE TEMPORARY FAISS INDEX
        # ======================================

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype(
            "float32"
        )

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
            max(k * 5, 20),
            len(candidate_indices)
        )

        distances, local_indices = (
            temp_index.search(
                query_embedding,
                search_k
            )
        )

        candidates = []

        for distance, local_index in zip(
            distances[0],
            local_indices[0]
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
                    document
                )
            )

            candidates.append({

                "document":
                    document,

                "index":
                    actual_index,

                "distance":
                    float(distance),

                "lexical_score":
                    lexical_score
            })

        if not candidates:
            return []

        exam_query = self.is_exam_query(
            query
        )

        # ======================================
        # RERANK
        # ======================================

        for candidate in candidates:

            document = candidate[
                "document"
            ]

            content_type = (
                self._get_content_type(
                    document
                ).lower()
            )

            distance = candidate[
                "distance"
            ]

            lexical_score = candidate[
                "lexical_score"
            ]

            # Smaller distance = better.
            semantic_score = (
                1.0
                / (1.0 + distance)
            )

            score = (
                lexical_score
                + semantic_score * 5
            )

            # ----------------------------------
            # Question bank
            # ----------------------------------

            if content_type == "question bank":

                if exam_query:

                    score += 12

                else:

                    score -= 8

            # ----------------------------------
            # Study notes for normal questions
            # ----------------------------------

            if (
                not exam_query
                and content_type != "question bank"
            ):

                score += 3

            candidate[
                "final_score"
            ] = score

        # ======================================
        # STRONG TOPIC SAFETY GATE
        # ======================================

        if canonical_topic:

            strict_candidates = [

                candidate

                for candidate in candidates

                if candidate[
                    "lexical_score"
                ] > 0
            ]

            if strict_candidates:

                candidates = (
                    strict_candidates
                )

        # ======================================
        # SORT
        # ======================================

        candidates.sort(

            key=lambda item: (
                item["final_score"],
                -item["distance"]
            ),

            reverse=True
        )

        # ======================================
        # TOP K
        # ======================================

        top_candidates = candidates[
            :max(1, k)
        ]

        results = []

        for candidate in top_candidates:

            result = self._make_result(

                document=
                    candidate["document"],

                distance=
                    candidate["distance"],

                index=
                    candidate["index"],

                lexical_score=
                    candidate["lexical_score"]
            )

            result["score"] = float(
                candidate["final_score"]
            )

            results.append(
                result
            )

        return results

    # ==========================================
    # ALL QUESTION BANK DOCUMENTS
    # ==========================================

    def get_all_question_bank_documents(
        self,
        subject=None,
        chapter=None
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

            results.append(
                self._make_result(

                    document=document,

                    distance=0.0,

                    index=index,

                    lexical_score=0
                )
            )

        return results

    # ==========================================
    # TOPIC QUESTION BANK
    # ==========================================

    def get_topic_question_bank_documents(
        self,
        topic,
        subject=None,
        chapter=None,
        limit=10
    ):

        if not topic:
            return []

        topic = topic.strip()

        candidate_indices = (
            self._candidate_indices(
                subject=subject,
                chapter=chapter
            )
        )

        matches = []

        canonical_topic = (
            self._canonical_topic(
                topic
            )
        )

        # ======================================
        # STRONG TECHNICAL TOPIC
        # ======================================

        if canonical_topic:

            for index in candidate_indices:

                document = self.documents[
                    index
                ]

                content_type = (
                    self._get_content_type(
                        document
                    ).lower()
                )

                if (
                    content_type
                    != "question bank"
                ):

                    continue

                if self._matches_strong_topic(
                    document,
                    canonical_topic
                ):

                    matches.append({

                        "index":
                            index,

                        "document":
                            document,

                        "score":
                            100,

                        "distance":
                            0.0
                    })

        # ======================================
        # NORMAL TOPIC
        # ======================================

        else:

            topic_words = set(
                re.findall(
                    r"[A-Za-z0-9]+",
                    self._normalize_text(
                        topic
                    )
                )
            )

            stop_words = {

                "a",
                "an",
                "the",

                "what",
                "which",
                "question",
                "questions",

                "related",
                "about",
                "on",

                "give",
                "me",

                "show",
                "list"
            }

            topic_words = {
                word
                for word in topic_words
                if (
                    word not in stop_words
                    and len(word) >= 2
                )
            }

            for index in candidate_indices:

                document = self.documents[
                    index
                ]

                content_type = (
                    self._get_content_type(
                        document
                    ).lower()
                )

                if (
                    content_type
                    != "question bank"
                ):

                    continue

                text = self._normalize_text(
                    document.get(
                        "text",
                        ""
                    )
                )

                score = 0

                # Exact topic phrase
                normalized_topic = (
                    self._normalize_text(
                        topic
                    )
                )

                if (
                    normalized_topic
                    and normalized_topic in text
                ):

                    score += 100

                # Word matching
                for word in topic_words:

                    if re.search(
                        rf"\b{re.escape(word)}\b",
                        text
                    ):

                        score += 10

                if score > 0:

                    matches.append({

                        "index":
                            index,

                        "document":
                            document,

                        "score":
                            score,

                        "distance":
                            0.0
                    })

        # ======================================
        # SEMANTIC FALLBACK
        # ======================================

        if not matches:

            semantic_results = self.retrieve(
                topic,
                k=max(
                    5,
                    limit
                ),
                subject=subject,
                chapter=chapter
            )

            for result in semantic_results:

                if (
                    result.get(
                        "content_type",
                        ""
                    ).lower()
                    != "question bank"
                ):

                    continue

                index = result.get(
                    "index"
                )

                if index is None:
                    continue

                matches.append({

                    "index":
                        index,

                    "document":
                        self.documents[
                            index
                        ],

                    "score":
                        1,

                    "distance":
                        result.get(
                            "distance",
                            0.0
                        )
                })

        # ======================================
        # SORT
        # ======================================

        matches.sort(

            key=lambda item: (
                item["score"],
                -item["distance"]
            ),

            reverse=True
        )

        # ======================================
        # BUILD RESULTS
        # ======================================

        results = []

        for item in matches[
            :max(1, limit)
        ]:

            result = self._make_result(

                document=
                    item["document"],

                distance=
                    item["distance"],

                index=
                    item["index"],

                lexical_score=
                    item["score"]
            )

            result["score"] = float(
                item["score"]
            )

            results.append(
                result
            )

        return results

    # ==========================================
    # BUILD CONTEXT
    # ==========================================

    def build_context(
        self,
        results
    ):

        if not results:
            return ""

        context_parts = []

        for i, result in enumerate(
            results,
            start=1
        ):

            metadata = result.get(
                "metadata",
                {}
            )

            filename = (
                result.get(
                    "filename"
                )
                or metadata.get(
                    "filename",
                    "Unknown file"
                )
            )

            subject = (
                result.get(
                    "subject"
                )
                or metadata.get(
                    "subject",
                    "General"
                )
            )

            chapter = (
                result.get(
                    "chapter"
                )
                or metadata.get(
                    "chapter",
                    "General"
                )
            )

            content_type = (
                result.get(
                    "content_type"
                )
                or metadata.get(
                    "content_type",
                    "Notes"
                )
            )

            text = result.get(
                "text",
                ""
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

    # ==========================================
    # DOCUMENT COUNT
    # ==========================================

    def get_document_count(self):

        filenames = set()

        for document in self.documents:

            filenames.add(
                self._get_filename(
                    document
                )
            )

        return len(filenames)

    # ==========================================
    # VECTOR COUNT
    # ==========================================

    def get_vector_count(self):

        if self.index is None:
            return 0

        return int(
            self.index.ntotal
        )

    # ==========================================
    # SUBJECTS
    # ==========================================

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
            key=lambda item:
                item.lower()
        )

    # ==========================================
    # CHAPTERS
    # ==========================================

    def get_chapters(
        self,
        subject=None
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
            key=lambda item:
                item.lower()
        )