import os
import re
import sqlite3
import time

import ollama


class LearningTools:
    def __init__(self, model="llama3.2:3b"):
        self.model = model
        self.max_context_chars = 7000
        self.max_generation_tokens = 500
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(backend_dir)
        default_db = os.path.join(project_root, "data", "learning_progress.db")
        self.db_path = os.getenv("DB_PATH", default_db)
        self._init_db()
        self.history = self.get_history()

    def _init_db(self):
        """Initialize SQLite database for persistent learning progress."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS learning_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        subject TEXT NOT NULL,
                        chapter TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        activity TEXT NOT NULL,
                        status TEXT DEFAULT 'Completed'
                    )
                    """
                )
                conn.commit()
        except Exception as e:
            print(f"Warning: Failed to initialize SQLite learning database: {e}")

    # ==========================================================
    # CONTEXT HELPERS
    # ==========================================================

    def _trim_context(self, context):
        if not context:
            return ""

        context = context.strip()

        if len(context) <= self.max_context_chars:
            return context

        trimmed = context[:self.max_context_chars]

        cut = trimmed.rfind("\n")

        if cut > 0:
            trimmed = trimmed[:cut]

        return (
            trimmed
            + "\n\n"
            + "[Remaining retrieved material omitted.]"
        )

    def _extract_topic_terms(self, topic):
        topic_lower = (
            (topic or "")
            .lower()
            .strip()
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
            "8 queens": [
                "8 queens",
                "8-queens",
                "eight queens",
                "eight-queens",
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
            "tuple": [
                "tuple",
                "tuples",
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

        for values in aliases.values():
            if any(
                value in topic_lower
                or topic_lower in value
                for value in values
            ):
                return list(
                    dict.fromkeys(values)
                )

        return re.findall(
            r"[A-Za-z0-9]+",
            topic_lower,
        )

    def _clean_source_block(self, block):
        if not block:
            return ""

        output = []

        for line in block.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            # RAG metadata
            if re.fullmatch(
                r"\[Source\s+\d+\]",
                stripped,
                re.IGNORECASE,
            ):
                continue

            if stripped == "---":
                continue

            if re.match(
                r"^(File|Subject|Chapter|Content Type)\s*:",
                stripped,
                re.IGNORECASE,
            ):
                continue

            if re.fullmatch(
                r"Content\s*:",
                stripped,
                re.IGNORECASE,
            ):
                continue

            output.append(
                line.rstrip()
            )

        return "\n".join(output).strip()

    def _select_relevant_context(
        self,
        topic,
        context,
    ):
        if not context:
            return ""

        terms = self._extract_topic_terms(
            topic
        )

        if not terms:
            return self._trim_context(
                context
            )

        topic_lower = (
            (topic or "")
            .lower()
            .strip()
        )

        blocks = re.split(
            r"\n\s*---+\s*\n",
            context,
        )

        scored_blocks = []

        for raw_block in blocks:
            block = self._clean_source_block(
                raw_block
            )

            if not block:
                continue

            block_lower = block.lower()

            score = 0

            if (
                topic_lower
                and topic_lower in block_lower
            ):
                score += 20

            for term in terms:
                if term.lower() in block_lower:
                    score += 4

            if "python code" in block_lower:
                score += 5

            if "example" in block_lower:
                score += 4

            if "syntax" in block_lower:
                score += 3

            if score > 0:
                scored_blocks.append(
                    (
                        score,
                        block,
                    )
                )

        if not scored_blocks:
            return self._trim_context(
                context
            )

        scored_blocks.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected = []
        current_size = 0

        for _, block in scored_blocks:
            if (
                current_size
                + len(block)
                > self.max_context_chars
            ):
                continue

            selected.append(block)

            current_size += (
                len(block)
                + 2
            )

        if selected:
            return "\n\n---\n\n".join(
                selected
            )

        return self._trim_context(
            context
        )

    # ==========================================================
    # OLLAMA
    # ==========================================================

    def _generate(self, prompt):
        start_time = time.perf_counter()

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                options={
                    "temperature": 0.15,
                    "num_predict": self.max_generation_tokens,
                    "num_ctx": 2048,
                },
                keep_alive="10m",
                stream=False,
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"Ollama generation time: "
                f"{elapsed:.2f} seconds"
            )

            return response[
                "message"
            ]["content"]

        except Exception as e:
            print(
                f"Ollama error: {e}"
            )

            return (
                "Sorry, I could not "
                "generate the answer "
                "right now."
            )

    # ==========================================================
    # SECTION HELPERS
    # ==========================================================

    def _normalize_heading(self, text):
        text = re.sub(
            r"^#+\s*",
            "",
            (text or "").strip(),
        )

        return (
            text
            .replace("**", "")
            .strip()
            .rstrip(":")
            .lower()
        )

    def _has_section(
        self,
        answer,
        section_name,
    ):
        target = self._normalize_heading(
            section_name
        )

        for line in (
            answer or ""
        ).splitlines():

            if (
                self._normalize_heading(
                    line
                )
                == target
            ):
                return True

        return False

    def _get_section_content(
        self,
        answer,
        section_names,
    ):
        if isinstance(
            section_names,
            str,
        ):
            section_names = [
                section_names
            ]

        targets = {
            self._normalize_heading(
                x
            )
            for x in section_names
        }

        all_sections = {
            self._normalize_heading(
                x
            )
            for x in [
                "Introduction",
                "Main Explanation",
                "Important Points",
                "Formula / Syntax",
                "Formula",
                "Example",
                "Example / Application",
                "Theory",
                "Practice",
                "Assessment",
                "Conclusion",
            ]
        }

        lines = (
            answer or ""
        ).splitlines()

        start = None

        for i, line in enumerate(
            lines
        ):
            if (
                self._normalize_heading(
                    line
                )
                in targets
            ):
                start = i + 1
                break

        if start is None:
            return None

        output = []

        for line in lines[start:]:

            if (
                self._normalize_heading(
                    line
                )
                in all_sections
            ):
                break

            output.append(line)

        return "\n".join(
            output
        ).strip()

    def _find_section_line(
        self,
        answer,
        section_names,
    ):
        if isinstance(
            section_names,
            str,
        ):
            section_names = [
                section_names
            ]

        targets = {
            self._normalize_heading(
                x
            )
            for x in section_names
        }

        for i, line in enumerate(
            (answer or "").splitlines()
        ):
            if (
                self._normalize_heading(
                    line
                )
                in targets
            ):
                return i

        return None

    def _insert_before_conclusion(
        self,
        answer,
        section,
        content,
    ):
        lines = (
            answer or ""
        ).splitlines()

        conclusion_index = (
            self._find_section_line(
                answer,
                "Conclusion",
            )
        )

        block = [
            "",
            section,
            "",
            content.strip(),
            "",
        ]

        if conclusion_index is None:
            return (
                answer.rstrip()
                + "\n\n"
                + section
                + "\n\n"
                + content.strip()
            ).strip()

        return "\n".join(
            lines[:conclusion_index]
            + block
            + lines[conclusion_index:]
        ).strip()

    def _replace_section(
        self,
        answer,
        section_names,
        correct_heading,
        content,
    ):
        start = self._find_section_line(
            answer,
            section_names,
        )

        if start is None:
            return self._insert_before_conclusion(
                answer,
                correct_heading,
                content,
            )

        known_sections = {
            self._normalize_heading(
                x
            )
            for x in [
                "Introduction",
                "Main Explanation",
                "Important Points",
                "Formula / Syntax",
                "Formula",
                "Example",
                "Example / Application",
                "Theory",
                "Practice",
                "Assessment",
                "Conclusion",
            ]
        }

        lines = (
            answer or ""
        ).splitlines()

        end = len(lines)

        for i in range(
            start + 1,
            len(lines),
        ):
            if (
                self._normalize_heading(
                    lines[i]
                )
                in known_sections
            ):
                end = i
                break

        new_lines = (
            lines[:start]
            + [
                correct_heading,
                "",
            ]
            + content.splitlines()
            + [""]
            + lines[end:]
        )

        return "\n".join(
            new_lines
        ).strip()

    # ==========================================================
    # CODE VALIDATION
    # ==========================================================

    def _example_content_is_valid(
        self,
        content,
    ):
        if (
            not content
            or len(
                content.strip()
            ) <= 10
        ):
            return False

        low = content.lower()

        invalid = [
            "no example was found",
            "no example",
            "not available",
            "[source",
            "file:",
            "subject:",
            "chapter:",
            "content type:",
            "content:",
        ]

        return not any(
            item in low
            for item in invalid
        )

    def _is_real_code_line(
        self,
        line,
    ):
        if not line:
            return False

        stripped = line.strip()

        if not stripped:
            return False

        lower = stripped.lower()

        if lower.rstrip(":") in {
            "example",
            "example 1",
            "example 2",
            "python code",
            "syntax",
            "output",
            "output:",
            "explanation",
            "correct way",
            "memory idea",
            "problem",
            "solution",
            "result",
            "conclusion",
        }:
            return False

        patterns = [
            r"^(import|from)\s+\w+",

            r"^def\s+\w+\s*\(",

            (
                r"^(if|elif|else|for|while|"
                r"try|except|finally|with)\b"
            ),

            r"^(return|pass|break|continue)\b",

            r"^print\s*\(",

            r"^[A-Za-z_]\w*\s*=\s*input\s*\(",

            (
                r"^[A-Za-z_]\w*"
                r"\s*=\s*[A-Za-z_]\w*"
                r"(?:\.[A-Za-z_]\w*)*"
                r"\s*\("
            ),

            (
                r"^[A-Za-z_]\w*"
                r"\s*=\s*"
                r"[\[\{\(\'\"]"
            ),

            r"^[A-Za-z_]\w*\s*=\s*\d",

            (
                r"^[A-Za-z_]\w*"
                r"\[[^\]]+\]\s*="
            ),

            (
                r"^[A-Za-z_]\w*"
                r"\.[A-Za-z_]\w*"
                r"\s*\("
            ),
            # C / C++ patterns
            r"^#include\s*<",
            r"^(int|void|float|double|char)\s+(main|[A-Za-z_]\w*)\s*\(",
            r"^(printf|scanf|puts|gets)\s*\(",
            r"^(std::)?(cout|cin)\s*<<?",
            r"^\s*[{}]\s*$",
            # Java patterns
            r"^(public|private|protected)\s+(static\s+)?(class|void|int|String)\b",
            r"^System\.(out|err)\.print",
            # SQL patterns
            r"^(SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+TABLE)\b",
            r"^(FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|INNER\s+JOIN|LEFT\s+JOIN)\b",
        ]

        return any(
            re.search(
                pattern,
                stripped,
                re.IGNORECASE,
            )
            for pattern in patterns
        )

    def _format_python_code(
        self,
        code_lines,
    ):
        lines = [
            line.rstrip()
            for line in code_lines
        ]

        nonempty = [
            line
            for line in lines
            if line.strip()
        ]

        if not nonempty:
            return []

        # Preserve indentation if it already exists.
        if any(
            len(line)
            - len(line.lstrip())
            > 0
            for line in nonempty
        ):
            return lines

        output = []
        indent = 0

        for raw_line in lines:
            stripped = raw_line.strip()

            if not stripped:
                continue

            if re.match(
                r"^(elif|else|except|finally)\b",
                stripped,
            ):
                indent = max(
                    0,
                    indent - 1,
                )

            output.append(
                "    " * indent
                + stripped
            )

            if stripped.endswith(":"):
                indent += 1

            elif re.match(
                r"^(return|pass|break|continue)\b",
                stripped,
            ):
                indent = max(
                    0,
                    indent - 1,
                )

        return output

    # ==========================================================
    # SPECIAL BFS EXTRACTION
    # ==========================================================

    def _extract_bfs_example(
        self,
        context,
    ):
        """
        Extract the BFS adjacency-list example.

        RAG retrieval can return overlapping chunks. Instead
        of joining chunks and accidentally duplicating lines,
        we verify the required source lines and reconstruct
        the exact BFS example.
        """

        if not context:
            return None

        # ------------------------------------------------------
        # Remove RAG metadata first.
        # ------------------------------------------------------

        cleaned_lines = []

        for raw_line in context.splitlines():

            stripped = raw_line.strip()

            if not stripped:
                continue

            # [Source 14]
            if re.fullmatch(
                r"\[Source\s+\d+\]",
                stripped,
                re.IGNORECASE,
            ):
                continue

            # Chunk separator
            if stripped == "---":
                continue

            # Metadata
            if re.match(
                r"^(File|Subject|Chapter|Content Type)\s*:",
                stripped,
                re.IGNORECASE,
            ):
                continue

            if re.fullmatch(
                r"Content\s*:",
                stripped,
                re.IGNORECASE,
            ):
                continue

            # Other RAG labels
            if re.fullmatch(
                r"(Output|Explanation|Correct Way|Memory Idea)",
                stripped,
                re.IGNORECASE,
            ):
                continue

            cleaned_lines.append(
                stripped
            )

        normalized = "\n".join(
            cleaned_lines
        )

        normalized_lower = normalized.lower()

        # ------------------------------------------------------
        # Verify that the actual source material contains
        # the complete BFS program.
        # ------------------------------------------------------

        required_fragments = [
            "from collections import deque",
            "graph = {",
            "'A': ['B', 'C']",
            "'B': ['A', 'D', 'E']",
            "'C': ['A', 'F']",
            "'D': ['B']",
            "'E': ['B']",
            "'F': ['C']",
            "def bfs(graph, start):",
            "visited = set([start])",
            "queue = deque([start])",
            "order = []",
            "while queue:",
            "node = queue.popleft()",
            "order.append(node)",
            "for neighbour in graph[node]:",
            "if neighbour not in visited:",
            "visited.add(neighbour)",
            "queue.append(neighbour)",
            "return order",
            "print(bfs(graph, 'A'))",
        ]

        for fragment in required_fragments:

            if (
                fragment.lower()
                not in normalized_lower
            ):
                return None

        # ------------------------------------------------------
        # Reconstruct exact source program.
        # ------------------------------------------------------

        code = """from collections import deque

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B'],
    'F': ['C']
}

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    return order

print(bfs(graph, 'A'))"""

        return (
            "Example 1\n"
            "PYTHON CODE\n"
            + code
        )

    # ==========================================================
    # GENERIC CODE EXTRACTION
    # ==========================================================

    def _extract_code_segments(
        self,
        source_blocks,
    ):
        if isinstance(
            source_blocks,
            str,
        ):
            source_blocks = [
                source_blocks
            ]

        segments = []

        current = []

        explicit_code = False
        marked_example = False

        stop_pattern = re.compile(
            (
                r"^(Output|Explanation|"
                r"Correct Way|Memory Idea|"
                r"Problem|Result|Solution|"
                r"Conclusion|Syntax)\s*:?\s*$"
            ),
            re.IGNORECASE,
        )

        def save_current():
            nonlocal current
            nonlocal explicit_code
            nonlocal marked_example

            if current:

                segments.append(
                    {
                        "code": current[:],
                        "explicit_code": explicit_code,
                        "marked_example": marked_example,
                    }
                )

            current = []
            explicit_code = False
            marked_example = False

        for block in source_blocks:

            if not block:
                continue

            for raw_line in block.splitlines():

                stripped = raw_line.strip()

                if not stripped:

                    if current:
                        current.append("")

                    continue

                if re.fullmatch(
                    r"Example(?:\s+\d+)?",
                    stripped,
                    re.IGNORECASE,
                ):

                    if current:
                        save_current()

                    marked_example = True

                    continue

                if re.fullmatch(
                    r"PYTHON\s+CODE",
                    stripped,
                    re.IGNORECASE,
                ):

                    explicit_code = True

                    continue

                if stop_pattern.fullmatch(
                    stripped
                ):

                    save_current()

                    continue

                if self._is_real_code_line(
                    stripped
                ):

                    current.append(
                        raw_line.rstrip()
                    )

                elif current:

                    save_current()

        save_current()

        return segments

    # ==========================================================
    # EXTRACT EXAMPLE
    # ==========================================================

    def _extract_example(
        self,
        context,
        topic=None,
    ):
        if not context:
            return None

        topic_lower = (
            topic or ""
        ).lower()

        # ------------------------------------------------------
        # BFS gets deterministic source extraction.
        # ------------------------------------------------------

        if (
            "bfs" in topic_lower
            or "breadth first" in topic_lower
            or "breadth-first" in topic_lower
        ):

            bfs_example = (
                self._extract_bfs_example(
                    context
                )
            )

            if bfs_example:
                return bfs_example

        # ------------------------------------------------------
        # Generic extraction for other topics.
        # ------------------------------------------------------

        blocks = re.split(
            r"\n\s*---+\s*\n",
            context,
        )

        blocks = [
            self._clean_source_block(
                block
            )
            for block in blocks
        ]

        blocks = [
            block
            for block in blocks
            if block
        ]

        if not blocks:
            return None

        segments = (
            self._extract_code_segments(
                blocks
            )
        )

        if not segments:
            return None

        topic_terms = (
            self._extract_topic_terms(
                topic
            )
        )

        full_text = (
            "\n".join(
                blocks
            ).lower()
        )

        candidates = []

        for segment in segments:

            code = (
                self._format_python_code(
                    segment["code"]
                )
            )

            code = [
                line
                for line in code
                if line.strip()
            ]

            if not code:
                continue

            code_text = "\n".join(
                code
            )

            score = 50

            if segment[
                "marked_example"
            ]:
                score += 25

            if segment[
                "explicit_code"
            ]:
                score += 20

            score += min(
                len(code),
                20,
            ) * 2

            if re.search(
                r"\bdef\s+\w+\s*\(",
                code_text,
            ):
                score += 15

            if any(
                term.lower()
                in full_text
                for term in topic_terms
            ):
                score += 10

            candidates.append(
                (
                    score,
                    code,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best_code = candidates[0][1]
        code_str = "\n".join(best_code)

        # Detect programming language tag
        if re.search(r"#include\s*<|printf\s*\(|int\s+main\b", code_str, re.IGNORECASE):
            lang_tag = "C CODE"
        elif re.search(r"SELECT\s+.+\s+FROM\b|CREATE\s+TABLE\b|INSERT\s+INTO\b", code_str, re.IGNORECASE):
            lang_tag = "SQL CODE"
        elif re.search(r"public\s+class\b|System\.out\.println", code_str, re.IGNORECASE):
            lang_tag = "JAVA CODE"
        else:
            lang_tag = "PYTHON CODE"

        return (
            "Example 1\n"
            f"{lang_tag}\n"
            + code_str
        )

    # ==========================================================
    # THEORY CLEANING
    # ==========================================================

    def _clean_theory(
        self,
        theory,
        context,
    ):
        if not theory:
            return theory

        output = []

        removable_headings = {
            "theory",
            "example",
            "practice",
            "assessment",
            "conclusion",
        }

        for line in theory.splitlines():

            normalized = (
                self._normalize_heading(
                    line
                )
            )

            if normalized in removable_headings:
                continue

            output.append(line)

        theory = "\n".join(
            output
        ).strip()

        context_lower = context.lower()

        if not re.search(
            (
                r"\b(?:useful|often used|"
                r"commonly used|advantage|"
                r"benefit|efficient)\b"
            ),
            context_lower,
        ):

            theory = re.sub(
                (
                    r"[^.\n]*\b(?:useful|"
                    r"usefully|often used|"
                    r"commonly used|"
                    r"advantage|benefit|"
                    r"efficient|efficiently)\b"
                    r"[^.\n]*\."
                ),
                "",
                theory,
                flags=re.IGNORECASE,
            )

        theory = re.sub(
            r"[^.\n]*CodeWithNishchal[^.\n]*\.",
            "",
            theory,
            flags=re.IGNORECASE,
        )

        theory = re.sub(
            r"\n{3,}",
            "\n\n",
            theory,
        )

        return theory.strip()

    # ==========================================================
    # ALGORITHM & ARCHITECTURE DIAGRAMS (CS SPECIALIZATION)
    # ==========================================================

    def _get_algorithm_diagram(self, topic, context):
        """
        Grounded lightweight algorithm flow and graph structure diagrams.
        Outputs Mermaid flowchart / graph representations and state traces.
        """
        topic_lower = (topic or "").lower().strip()

        # BFS
        if any(x in topic_lower for x in ["bfs", "breadth first", "breadth-first"]):
            return (
                "### 📊 Algorithm Flow & Graph Visualization\n\n"
                "```mermaid\n"
                "graph TD\n"
                "    A[\"Node A (Start)\"] --> B[\"Node B\"]\n"
                "    A --> C[\"Node C\"]\n"
                "    B --> D[\"Node D\"]\n"
                "    B --> E[\"Node E\"]\n"
                "    C --> F[\"Node F\"]\n"
                "    E --> F\n"
                "```\n\n"
                "**FIFO Queue State Progression:**\n"
                "- **Step 1:** Queue: `['A']` ➔ Dequeue **A**, Visit **A**, Enqueue neighbors: `B`, `C`\n"
                "- **Step 2:** Queue: `['B', 'C']` ➔ Dequeue **B**, Visit **B**, Enqueue neighbors: `D`, `E`\n"
                "- **Step 3:** Queue: `['C', 'D', 'E']` ➔ Dequeue **C**, Visit **C**, Enqueue neighbor: `F`\n"
                "- **Step 4:** Queue: `['D', 'E', 'F']` ➔ Dequeue **D**, Visit **D** (leaf)\n"
                "- **Step 5:** Queue: `['E', 'F']` ➔ Dequeue **E**, Visit **E** (neighbor F already visited)\n"
                "- **Step 6:** Queue: `['F']` ➔ Dequeue **F**, Visit **F** (leaf)\n"
                "- **Final Traversal Order:** `A ➔ B ➔ C ➔ D ➔ E ➔ F`"
            )

        # DFS
        if any(x in topic_lower for x in ["dfs", "depth first", "depth-first"]):
            return (
                "### 📊 Algorithm Flow & Graph Visualization\n\n"
                "```mermaid\n"
                "graph TD\n"
                "    A[\"Node A (Start)\"] --> B[\"Node B\"]\n"
                "    B --> D[\"Node D\"]\n"
                "    B --> E[\"Node E\"]\n"
                "    A --> C[\"Node C\"]\n"
                "    C --> F[\"Node F\"]\n"
                "```\n\n"
                "**LIFO Stack / Recursive Traversal Path:**\n"
                "- **Step 1:** Push **A** onto stack ➔ Visit **A**\n"
                "- **Step 2:** Branch deepest to **B** ➔ Push **B** ➔ Visit **B**\n"
                "- **Step 3:** Branch deepest to **D** ➔ Push **D** ➔ Visit **D** (leaf, backtrack to **B**)\n"
                "- **Step 4:** Branch to **E** ➔ Push **E** ➔ Visit **E** (leaf, backtrack to **A**)\n"
                "- **Step 5:** Branch to unvisited neighbor **C** ➔ Push **C** ➔ Visit **C**\n"
                "- **Step 6:** Branch deepest to **F** ➔ Push **F** ➔ Visit **F**\n"
                "- **Traversal Order:** `A ➔ B ➔ D ➔ E ➔ C ➔ F`"
            )

        # Binary Search
        if "binary search" in topic_lower:
            return (
                "### 📊 Algorithm Flow & Search Pipeline\n\n"
                "```mermaid\n"
                "flowchart LR\n"
                "    Start([\"Sorted Array [L..R]\"]) --> Mid[\"Mid = (L + R) // 2\"]\n"
                "    Mid --> Check{\"target == arr[mid]?\"}\n"
                "    Check -- \"Yes\" --> Found([\"Return Index mid\"])\n"
                "    Check -- \"< arr[mid]\" --> Left[\"Search Left: R = mid - 1\"]\n"
                "    Check -- \"> arr[mid]\" --> Right[\"Search Right: L = mid + 1\"]\n"
                "    Left --> Mid\n"
                "    Right --> Mid\n"
                "```\n\n"
                "**Complexity:** Time: `O(log N)` | Space: `O(1)`"
            )

        # DBMS Normalization
        if any(x in topic_lower for x in ["normalization", "normal form"]):
            return (
                "### 📊 Database Normalization Hierarchy\n\n"
                "```mermaid\n"
                "flowchart TD\n"
                "    UNF[\"Unnormalized Form (Repeating Groups)\"] -->|Atomic attributes| NF1[\"1NF (First Normal Form)\"]\n"
                "    NF1 -->|Remove Partial Dependencies| NF2[\"2NF (Second Normal Form)\"]\n"
                "    NF2 -->|Remove Transitive Dependencies| NF3[\"3NF (Third Normal Form)\"]\n"
                "    NF3 -->|Every determinant is a superkey| BCNF[\"BCNF (Boyce-Codd Normal Form)\"]\n"
                "```"
            )

        # OSI Model
        if "osi" in topic_lower:
            return (
                "### 📊 OSI Reference Architecture Flow\n\n"
                "```mermaid\n"
                "flowchart TD\n"
                "    L7[\"7. Application Layer (HTTP, DNS, SMTP)\"] --> L6[\"6. Presentation Layer (SSL/TLS, Compression)\"]\n"
                "    L6 --> L5[\"5. Session Layer (Sockets, RPC)\"]\n"
                "    L5 --> L4[\"4. Transport Layer (TCP, UDP, Port Addressing)\"]\n"
                "    L4 --> L3[\"3. Network Layer (IP, Routing, Routers)\"]\n"
                "    L3 --> L2[\"2. Data Link Layer (Frames, MAC, Ethernet, Switches)\"]\n"
                "    L2 --> L1[\"1. Physical Layer (Bits, Cables, Voltage)\"]\n"
                "```"
            )

        # OS Process Lifecycle
        if any(x in topic_lower for x in ["process state", "process lifecycle", "process management"]):
            return (
                "### 📊 OS Process State Transitions\n\n"
                "```mermaid\n"
                "flowchart LR\n"
                "    New([\"New\"]) -->|Admitted| Ready[\"Ready\"]\n"
                "    Ready -->|Scheduler Dispatch| Running[\"Running\"]\n"
                "    Running -->|Interrupt / Timeout| Ready\n"
                "    Running -->|I/O or Event Wait| Waiting[\"Waiting / Blocked\"]\n"
                "    Waiting -->|I/O Completion| Ready\n"
                "    Running -->|Exit| Terminated([\"Terminated\"])\n"
                "```"
            )

        return None

    # ==========================================================
    # EXPLAIN TOPIC
    # ==========================================================

    def explain_topic(
        self,
        topic,
        context,
    ):
        context = self._trim_context(
            self._select_relevant_context(
                topic,
                context,
            )
        )

        prompt = f"""
You are an academic assistant.

Explain the topic below using ONLY the relevant
uploaded study material.

TOPIC:
{topic}

RELEVANT STUDY MATERIAL:
{context}

STRICT RULES:

- Use only information supported by the study material.
- Do not use outside knowledge.
- Do not invent facts.
- Do not invent formulas.
- Do not invent examples.
- Do not invent uses.
- Do not invent code.
- Ignore unrelated retrieved material.
- Keep the explanation simple.

Use exactly these sections:

Introduction

Main Explanation

Important Points

Formula / Syntax

Example

Conclusion

For a missing formula, write:

Not available in the uploaded study material.

For a missing example, write:

No example was found in the uploaded study material.
"""

        answer = self._generate(
            prompt
        )

        example = self._extract_example(
            context,
            topic,
        )

        if example:

            answer = self._replace_section(
                answer,
                "Example",
                "Example",
                example,
            )

        diagram = self._get_algorithm_diagram(
            topic,
            context,
        )

        if diagram:

            answer = self._insert_before_conclusion(
                answer,
                "Algorithm Flow & Diagram",
                diagram,
            )

        self.record_activity(
            topic,
            "Topic explanation",
        )

        return answer

    # ==========================================================
    # SOLVE QUESTION
    # ==========================================================

    def solve_question(
        self,
        question,
        context,
        example_context=None,
    ):
        context = self._trim_context(
            self._select_relevant_context(
                question,
                context,
            )
        )

        prompt = f"""
You are an academic question-solving assistant.

Answer this question using ONLY the relevant
uploaded study material.

QUESTION:
{question}

RELEVANT STUDY MATERIAL:
{context}

STRICT RULES:

- Use only the provided material.
- Ignore unrelated content.
- Do not use outside knowledge.
- Do not invent facts.
- Do not invent formulas.
- Do not invent examples.
- Do not invent steps.
- Do not invent code.
- Do not assume information that is missing.

Use exactly these sections:

Introduction

Main Explanation

Important Points

Formula / Syntax

Example / Application

Conclusion

For a missing formula, write:

Not available in the uploaded study material.

For a missing example, write:

No example was found in the uploaded study material.
"""

        answer = self._generate(
            prompt
        )

        # Use dedicated full source material for examples
        # when app.py supplies it.
        source = (
            example_context
            if example_context
            else context
        )

        example = self._extract_example(
            source,
            question,
        )

        # ALWAYS replace the model-generated example
        # when a real source example exists.
        if example:

            answer = self._replace_section(
                answer,
                [
                    "Example / Application",
                    "Example",
                ],
                "Example / Application",
                example,
            )

        else:

            answer = self._replace_section(
                answer,
                [
                    "Example / Application",
                    "Example",
                ],
                "Example / Application",
                (
                    "No example was found in the "
                    "uploaded study material."
                ),
            )

        self.record_activity(
            question,
            "Question solving",
        )

        return answer

    # ==========================================================
    # SYNTHESIS
    # ==========================================================

    def synthesize_content(
        self,
        topic,
        context,
    ):
        context = self._trim_context(
            self._select_relevant_context(
                topic,
                context,
            )
        )

        prompt = f"""
You are an academic content synthesis assistant.

Create a concise study guide for:

{topic}

STUDY MATERIAL:
{context}

Use ONLY information supported by the material.

STRICT RULES:

- Do not use outside knowledge.
- Do not invent facts.
- Do not invent examples.
- Do not invent formulas.
- Do not invent steps.
- Do not add unsupported applications.

Use these sections:

1. Overview
2. Important Concepts
3. Detailed Explanation
4. Formulas or Steps
5. Examples from the Material
6. Applications from the Material
7. Key Points for Examination
8. Summary
"""

        answer = self._generate(
            prompt
        )

        self.record_activity(
            topic,
            "Content synthesis",
        )

        return answer

    # ==========================================================
    # LEARNING PROGRESSION
    # ==========================================================

    def learning_progression(
        self,
        topic,
        context,
        practice_questions=None,
        example_context=None,
    ):
        context = self._trim_context(
            self._select_relevant_context(
                topic,
                context,
            )
        )

        # ------------------------------------------------------
        # THEORY
        # ------------------------------------------------------

        theory_prompt = f"""
Explain ONE topic strictly from the supplied
uploaded study material.

TOPIC:
{topic}

MATERIAL:
{context}

Use ONLY supported information.

Do not add:

- outside knowledge
- unsupported uses
- unsupported advantages
- unsupported examples
- unsupported syntax
- unsupported formulas
- unsupported code

Keep it beginner-friendly.

Write only the theory.
"""

        theory = self._generate(
            theory_prompt
        ).strip()

        theory = self._clean_theory(
            theory,
            context,
        )

        if not theory:
            theory = (
                "No sufficient theory was "
                "found in the uploaded "
                "study material."
            )

        # ------------------------------------------------------
        # EXAMPLE
        # ------------------------------------------------------

        source = (
            example_context
            if example_context
            else context
        )

        example = self._extract_example(
            source,
            topic,
        )

        if not example:
            example = (
                "No example was found in "
                "the uploaded study material."
            )

        # ------------------------------------------------------
        # PRACTICE QUESTIONS
        # ------------------------------------------------------

        clean_questions = []

        for question in (
            practice_questions or []
        ):

            question = re.sub(
                r"\s+",
                " ",
                str(question).strip(),
            )

            if (
                question
                and question
                not in clean_questions
            ):
                clean_questions.append(
                    question
                )

        if clean_questions:

            practice = "\n".join(
                f"{i}. {question}"
                for i, question in enumerate(
                    clean_questions,
                    start=1,
                )
            )

        else:

            practice = (
                "No related practice "
                "questions were found in "
                "the uploaded question bank."
            )

        # ------------------------------------------------------
        # ASSESSMENT
        # ------------------------------------------------------

        if clean_questions:

            assessment = "\n".join(
                f"{i}. {question}"
                for i, question in enumerate(
                    clean_questions[:3],
                    start=1,
                )
            )

        else:

            assessment = (
                "No separate assessment "
                "question is available in "
                "the uploaded question bank."
            )

        # ------------------------------------------------------
        # FINAL
        # ------------------------------------------------------

        progression_parts = [
            "THEORY",
            theory,
            "EXAMPLE",
            example,
        ]

        diagram = self._get_algorithm_diagram(
            topic,
            context,
        )

        if diagram:
            progression_parts.extend([
                "DIAGRAM & FLOW",
                diagram,
            ])

        progression_parts.extend([
            "PRACTICE",
            practice,
            "ASSESSMENT",
            assessment,
        ])

        answer = "\n\n".join(
            progression_parts
        )

        self.record_activity(
            topic,
            "Learning progression",
        )

        return answer

    # ==========================================================
    # EXAM PREPARATION
    # ==========================================================

    def exam_prep_guide(
        self,
        topic,
        context,
        practice_questions=None,
        example_context=None,
    ):
        context_trimmed = self._trim_context(
            self._select_relevant_context(
                topic,
                context,
            )
        )

        questions = []

        for question in (
            practice_questions or []
        ):

            question = re.sub(
                r"\s+",
                " ",
                str(question).strip(),
            )

            if (
                question
                and question
                not in questions
            ):
                questions.append(
                    question
                )

        if questions:

            question_text = "\n".join(
                f"{i}. {question}"
                for i, question in enumerate(
                    questions,
                    start=1,
                )
            )

        else:

            question_text = (
                "No related questions found "
                "in the uploaded question bank."
            )

        prompt = f"""
You are an academic exam-preparation assistant.

Use ONLY the supplied material.

TOPIC:
{topic}

STUDY MATERIAL:
{context_trimmed}

QUESTION BANK:
{question_text}

STRICT RULES:

- Do not use outside knowledge.
- Do not invent facts.
- Do not invent formulas.
- Do not invent examples.
- Do not invent steps.
- Do not invent questions.
- Use question-bank questions exactly as supplied.

Create:

THEORY
IMPORTANT POINTS
EXAMPLE
PRACTICE
ASSESSMENT
EXAM REVISION
"""

        answer = self._generate(
            prompt
        )

        source = (
            example_context
            if example_context
            else context
        )

        example = self._extract_example(
            source,
            topic,
        )

        if example:
            answer = self._replace_section(
                answer,
                ["EXAMPLE", "Example"],
                "EXAMPLE",
                example,
            )

        self.record_activity(
            topic,
            "Exam preparation guide",
        )

        return answer

    # ==========================================================
    # STUDY PLAN GENERATOR (WEEK 5-6 DOMAIN SPECIALIZATION)
    # ==========================================================

    def generate_study_plan(
        self,
        subject,
        chapter,
        topics=None,
        days=5,
        context="",
        practice_questions=None,
    ):
        topic_list_str = (
            ", ".join(topics)
            if isinstance(topics, list) and topics
            else (subject or "General")
        )

        q_count = len(practice_questions) if practice_questions else 0
        context_excerpt = self._trim_context(context)[:2000] if context else ""

        prompt = f"""
You are an expert university academic advisor.
Create a structured {days}-Day Revision & Exam Study Plan.

SUBJECT: {subject}
CHAPTER / MODULE: {chapter}
TOPICS: {topic_list_str}
AVAILABLE PRACTICE QUESTIONS: {q_count}

SYLLABUS & CONTEXT EXCERPT:
{context_excerpt}

STRICT INSTRUCTIONS:
- Break the study load into {days} distinct, manageable study sessions (Day 1 through Day {days}).
- For each day, include:
  * Core Concept Focus
  * Reading & Theory Objectives
  * Specific Hands-on / Code / Problem Practice
  * Self-check Questions
- Include a concluding 'High-Yield Exam Strategy & Weak-Area Tips' section.
- Keep the guidance practical and syllabus-focused.
"""

        plan = self._generate(prompt)

        self.record_activity(
            f"{subject} - {chapter}",
            f"{days}-Day study plan",
        )

        return plan

    # ==========================================================
    # SELF-ASSESSMENT QUIZ (WEAK AREA IDENTIFICATION)
    # ==========================================================

    def generate_quiz(
        self,
        topic,
        context,
    ):
        context_trimmed = self._trim_context(
            self._select_relevant_context(
                topic,
                context,
            )
        )

        prompt = f"""
You are an academic assessment assistant.
Create a diagnostic self-assessment quiz on: '{topic}'.
Use ONLY the provided study material.

STUDY MATERIAL:
{context_trimmed}

Format the quiz with:
1. Multiple Choice Questions (2 questions with options A, B, C, D)
2. Conceptual / Definition Question (1 question)
3. Application or Code / Algorithm Problem (1 question)
4. Comprehensive Answer Key with Explanations (placed at the end to evaluate understanding and identify weak areas)

Ensure all questions strictly test concepts from the provided study material.
"""

        quiz = self._generate(prompt)

        self.record_activity(
            topic,
            "Self-assessment quiz",
        )

        return quiz

    # ==========================================================
    # HISTORY & TRACKING
    # ==========================================================

    def record_activity(
        self,
        topic,
        activity,
        subject="General",
        chapter="General",
        status="Completed",
    ):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "timestamp": timestamp,
            "subject": subject or "General",
            "chapter": chapter or "General",
            "topic": topic or "General",
            "activity": activity,
            "status": status,
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO learning_history (timestamp, subject, chapter, topic, activity, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["timestamp"],
                        record["subject"],
                        record["chapter"],
                        record["topic"],
                        record["activity"],
                        record["status"],
                    ),
                )
                conn.commit()
                record["id"] = cursor.lastrowid
        except Exception as e:
            print(f"Warning: Failed to persist activity to SQLite: {e}")

        self.history.append(record)
        return record

    def get_history(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, timestamp, subject, chapter, topic, activity, status
                    FROM learning_history
                    ORDER BY id ASC
                    """
                )
                rows = cursor.fetchall()
                self.history = [dict(row) for row in rows]
                return self.history
        except Exception as e:
            print(f"Warning: Failed to read from SQLite history: {e}")
            return self.history