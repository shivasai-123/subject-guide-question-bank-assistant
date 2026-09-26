import re
import time

import ollama


class LearningTools:

    def __init__(self, model="llama3.2:3b"):

        self.model = model
        self.history = []

        # Keep enough context for useful academic answers.
        self.max_context_chars = 6000

        # Increased so longer answers and programs
        # are less likely to stop in the middle.
        self.max_generation_tokens = 700

    # ==========================================
    # HELPER - TRIM CONTEXT
    # ==========================================

    def _trim_context(self, context):

        if not context:
            return ""

        context = context.strip()

        if len(context) <= self.max_context_chars:
            return context

        trimmed = context[:self.max_context_chars]

        last_newline = trimmed.rfind("\n")

        if last_newline > 0:
            trimmed = trimmed[:last_newline]

        return (
            trimmed
            + "\n\n"
            + "[Remaining retrieved material omitted.]"
        )

    # ==========================================
    # HELPER - EXTRACT RELEVANT TERMS
    # ==========================================

    def _extract_topic_terms(self, topic):

        topic_lower = topic.lower().strip()

        # --------------------------------------
        # Common technical aliases
        # --------------------------------------

        aliases = {

            "bfs": [
                "bfs",
                "breadth first search",
                "breadth-first search"
            ],

            "dfs": [
                "dfs",
                "depth first search",
                "depth-first search"
            ],

            "8 queens": [
                "8 queens",
                "8-queens",
                "eight queens"
            ],

            "water jug": [
                "water jug"
            ],

            "hill climbing": [
                "hill climbing"
            ],

            "tower of hanoi": [
                "tower of hanoi"
            ],

            "alpha beta": [
                "alpha beta",
                "alpha-beta pruning",
                "alpha beta pruning"
            ],

            "tuple unpacking": [
                "tuple unpacking"
            ],

            "tuple packing": [
                "tuple packing"
            ]
        }

        phrases = []

        # --------------------------------------
        # Add aliases when canonical topic exists
        # --------------------------------------

        for alias, variations in aliases.items():

            if alias in topic_lower:

                phrases.extend(
                    variations
                )

        # --------------------------------------
        # Add aliases when variation exists
        # directly in the query
        # --------------------------------------

        for variation_list in aliases.values():

            for variation in variation_list:

                if variation in topic_lower:

                    phrases.append(
                        variation
                    )

        # --------------------------------------
        # Word-based matching
        # --------------------------------------

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
            "related"
        }

        words = re.findall(
            r"[A-Za-z0-9]+",
            topic_lower
        )

        useful_words = [
            word
            for word in words
            if word not in stop_words
            and len(word) >= 2
        ]

        return {
            "phrases": list(
                dict.fromkeys(phrases)
            ),
            "words": set(
                useful_words
            )
        }

    # ==========================================
    # HELPER - SELECT RELEVANT CONTEXT
    # ==========================================

    def _select_relevant_context(
        self,
        topic,
        context
    ):
        """
        Reduce unrelated retrieved material before
        sending it to the local LLM.

        Exact topic phrases receive a strong score.
        Generic words such as "first" or "search"
        do not independently cause a match.
        """

        if not context:
            return ""

        term_info = self._extract_topic_terms(
            topic
        )

        topic_phrases = term_info[
            "phrases"
        ]

        topic_words = term_info[
            "words"
        ]

        normalized_topic = (
            topic.lower().strip()
        )

        # --------------------------------------
        # Split retrieved context into blocks
        # --------------------------------------

        blocks = re.split(
            r"\n\s*[-=]{3,}\s*\n|\n{2,}",
            context
        )

        scored_blocks = []

        for block in blocks:

            block = block.strip()

            if not block:
                continue

            block_lower = block.lower()

            score = 0

            # ----------------------------------
            # Strong score for exact query
            # ----------------------------------

            if normalized_topic and (
                normalized_topic
                in block_lower
            ):

                score += 12

            # ----------------------------------
            # Strong score for technical aliases
            # ----------------------------------

            for phrase in topic_phrases:

                if phrase in block_lower:

                    score += 10 + min(
                        len(phrase.split()),
                        5
                    )

            # ----------------------------------
            # Word overlap
            # ----------------------------------

            matched_words = 0

            for word in topic_words:

                if re.search(
                    rf"\b{re.escape(word)}\b",
                    block_lower
                ):

                    matched_words += 1

            score += matched_words * 2

            # ----------------------------------
            # Keep matching blocks
            # ----------------------------------

            if score > 0:

                scored_blocks.append(
                    (
                        score,
                        block
                    )
                )

        # --------------------------------------
        # Sort by relevance
        # --------------------------------------

        if scored_blocks:

            scored_blocks.sort(
                key=lambda item: item[0],
                reverse=True
            )

            selected = []
            current_size = 0

            for _, block in scored_blocks:

                if (
                    current_size
                    + len(block)
                    > self.max_context_chars
                ):
                    break

                selected.append(
                    block
                )

                current_size += (
                    len(block) + 2
                )

            if selected:

                return "\n\n".join(
                    selected
                )

        # --------------------------------------
        # Fallback
        # --------------------------------------

        return self._trim_context(
            context
        )

    # ==========================================
    # HELPER - CHECK SOLUTION SUPPORT
    # ==========================================

    def _has_solution_support(
        self,
        question,
        context
    ):
        """
        Check whether the uploaded material actually
        contains implementation/code support for a
        programming question.

        This prevents the local LLM from inventing
        a program when the uploaded document contains
        only the question.
        """

        if not context:
            return False

        question_lower = question.lower()

        # --------------------------------------
        # Detect questions that require code
        # --------------------------------------

        programming_phrases = [
            "write a program",
            "write a python program",
            "write a c program",
            "write a c++ program",
            "implement",
            "implementation",
            "program to",
            "code for",
            "write code"
        ]

        code_required = any(
            phrase in question_lower
            for phrase in programming_phrases
        )

        # --------------------------------------
        # Normal theory questions do not need
        # this special code-support check.
        # --------------------------------------

        if not code_required:
            return True

        context_lower = context.lower()

        # --------------------------------------
        # Extract the topic from the question.
        # --------------------------------------

        term_info = self._extract_topic_terms(
            question
        )

        topic_phrases = term_info[
            "phrases"
        ]

        topic_words = term_info[
            "words"
        ]

        # --------------------------------------
        # Look for actual code blocks.
        # --------------------------------------

        code_blocks = re.findall(
            r"```(?:python|py|c|cpp|c\+\+)?\s*(.*?)```",
            context_lower,
            flags=re.DOTALL
        )

        # --------------------------------------
        # Code-like markers.
        # --------------------------------------

        code_markers = [
            "def ",
            "class ",
            "import ",
            "from ",
            "#include",
            "void ",
            "int main",
            "return ",
            "while ",
            "for ",
            "if "
        ]

        # --------------------------------------
        # Strongest check:
        # actual code block containing both
        # topic information and code structure.
        # --------------------------------------

        for code_block in code_blocks:

            has_code = any(
                marker in code_block
                for marker in code_markers
            )

            if not has_code:
                continue

            topic_match = False

            for phrase in topic_phrases:

                if phrase in code_block:
                    topic_match = True
                    break

            if not topic_match:

                for word in topic_words:

                    if re.search(
                        rf"\b{re.escape(word)}\b",
                        code_block
                    ):

                        topic_match = True
                        break

            if topic_match:
                return True

        # --------------------------------------
        # Some notes may not use ``` blocks.
        # Look for a topic-specific code region.
        # --------------------------------------

        for phrase in topic_phrases:

            position = context_lower.find(
                phrase
            )

            if position == -1:
                continue

            start = max(
                0,
                position - 500
            )

            end = min(
                len(context_lower),
                position + 2500
            )

            nearby_text = context_lower[
                start:end
            ]

            has_code = any(
                marker in nearby_text
                for marker in code_markers
            )

            if has_code:
                return True

        # --------------------------------------
        # No implementation support found.
        # --------------------------------------

        return False

    # ==========================================
    # HELPER - CALL OLLAMA
    # ==========================================

    def _generate(self, prompt):

        start_time = time.perf_counter()

        try:

            response = ollama.chat(

                model=self.model,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                options={

                    "temperature": 0.2,

                    "num_predict":
                        self.max_generation_tokens,

                    "num_ctx": 4096
                },

                keep_alive="10m",

                stream=False
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
            ][
                "content"
            ]

        except Exception as e:

            print(
                f"Ollama error: {e}"
            )

            return (
                "Sorry, I could not generate "
                "the answer right now."
            )

    # ==========================================
    # HELPER - NORMALIZE HEADING
    # ==========================================

    def _normalize_heading(self, text):

        text = text.strip()

        text = re.sub(
            r"^#+\s*",
            "",
            text
        )

        text = text.replace(
            "**",
            ""
        ).strip()

        text = text.rstrip(":")

        return text.strip().lower()

    # ==========================================
    # HELPER - CHECK SECTION
    # ==========================================

    def _has_section(
        self,
        answer,
        section_name
    ):

        target = self._normalize_heading(
            section_name
        )

        for line in answer.splitlines():

            if (
                self._normalize_heading(
                    line
                )
                == target
            ):

                return True

        return False

    # ==========================================
    # HELPER - GET SECTION CONTENT
    # ==========================================

    def _get_section_content(
        self,
        answer,
        section_names
    ):

        if isinstance(
            section_names,
            str
        ):

            section_names = [
                section_names
            ]

        target_names = {
            self._normalize_heading(
                name
            )
            for name in section_names
        }

        all_sections = {
            self._normalize_heading(name)
            for name in [

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
                "Conclusion"
            ]
        }

        lines = answer.splitlines()

        start_index = None

        for i, line in enumerate(lines):

            normalized = (
                self._normalize_heading(
                    line
                )
            )

            if normalized in target_names:

                start_index = i + 1
                break

        if start_index is None:
            return None

        content = []

        for i in range(
            start_index,
            len(lines)
        ):

            normalized = (
                self._normalize_heading(
                    lines[i]
                )
            )

            if normalized in all_sections:
                break

            content.append(
                lines[i]
            )

        return "\n".join(
            content
        ).strip()

    # ==========================================
    # HELPER - FIND SECTION
    # ==========================================

    def _find_section_line(
        self,
        answer,
        section_names
    ):

        if isinstance(
            section_names,
            str
        ):

            section_names = [
                section_names
            ]

        targets = {
            self._normalize_heading(
                name
            )
            for name in section_names
        }

        lines = answer.splitlines()

        for i, line in enumerate(lines):

            if (
                self._normalize_heading(
                    line
                )
                in targets
            ):

                return i

        return None

    # ==========================================
    # HELPER - INSERT SECTION
    # ==========================================

    def _insert_before_conclusion(
        self,
        answer,
        section,
        content
    ):

        lines = answer.splitlines()

        conclusion_line = (
            self._find_section_line(
                answer,
                "Conclusion"
            )
        )

        section_lines = [

            "",

            section,

            "",

            content,

            ""
        ]

        if conclusion_line is not None:

            return "\n".join(

                lines[:conclusion_line]

                + section_lines

                + lines[
                    conclusion_line:
                ]

            ).strip()

        return (
            answer.rstrip()
            + "\n\n"
            + section
            + "\n\n"
            + content.strip()
        )

    # ==========================================
    # HELPER - REPLACE SECTION
    # ==========================================

    def _replace_section(
        self,
        answer,
        section_names,
        correct_heading,
        content
    ):

        start_line = (
            self._find_section_line(
                answer,
                section_names
            )
        )

        if start_line is None:

            return self._insert_before_conclusion(
                answer,
                correct_heading,
                content
            )

        known_sections = [

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
            "Conclusion"
        ]

        normalized_known = {
            self._normalize_heading(
                name
            )
            for name in known_sections
        }

        lines = answer.splitlines()

        end_line = len(lines)

        for i in range(
            start_line + 1,
            len(lines)
        ):

            if (
                self._normalize_heading(
                    lines[i]
                )
                in normalized_known
            ):

                end_line = i
                break

        new_lines = []

        new_lines.extend(
            lines[:start_line]
        )

        new_lines.append(
            correct_heading
        )

        new_lines.append("")

        new_lines.extend(
            content.splitlines()
        )

        new_lines.append("")

        new_lines.extend(
            lines[end_line:]
        )

        return "\n".join(
            new_lines
        ).strip()

    # ==========================================
    # HELPER - VALIDATE EXAMPLE
    # ==========================================

    def _example_content_is_valid(
        self,
        content
    ):

        if not content:
            return False

        lowered = content.lower()

        invalid_phrases = [

            "no example was found",

            "not available",

            "no relevant example",

            "no example"
        ]

        for phrase in invalid_phrases:

            if phrase in lowered:
                return False

        return len(
            content.strip()
        ) > 10

    # ==========================================
    # HELPER - EXTRACT EXAMPLE
    # ==========================================

    def _extract_example(
        self,
        context,
        topic=None
    ):

        if not context:
            return None

        topic_terms = []

        if topic:

            topic_terms = [

                word.lower()

                for word in re.findall(
                    r"[A-Za-z0-9]+",
                    topic
                )

                if len(word) >= 3
            ]

        lines = context.splitlines()

        candidates = []

        for i, line in enumerate(lines):

            clean_line = line.strip()

            if not clean_line:
                continue

            is_example = bool(
                re.search(
                    r"\bexample\b",
                    clean_line,
                    re.IGNORECASE
                )
            )

            is_problem = bool(
                re.search(
                    r"\bproblem\s*:",
                    clean_line,
                    re.IGNORECASE
                )
            )

            if not (
                is_example
                or is_problem
            ):
                continue

            example_lines = [
                clean_line
            ]

            for j in range(
                i + 1,
                min(
                    i + 8,
                    len(lines)
                )
            ):

                next_line = (
                    lines[j].strip()
                )

                if not next_line:
                    continue

                if next_line.startswith(
                    "----------------"
                ):
                    break

                if (
                    j > i + 1
                    and re.search(
                        r"^(?:[A-Za-z ]+\s+)?Example\b",
                        next_line,
                        re.IGNORECASE
                    )
                ):
                    break

                example_lines.append(
                    next_line
                )

            block = "\n".join(
                example_lines
            ).strip()

            block_lower = block.lower()

            score = 0

            for term in topic_terms:

                if term in block_lower:
                    score += 10

            if "solution" in block_lower:
                score += 3

            if "code" in block_lower:
                score += 2

            candidates.append(
                (
                    score,
                    block
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        # Don't return a weak unrelated example.
        if candidates[0][0] <= 0:
            return None

        return candidates[0][1]

    # ==========================================
    # 1. EXPLAIN TOPIC
    # ==========================================

    def explain_topic(
        self,
        topic,
        context
    ):

        context = (
            self._select_relevant_context(
                topic,
                context
            )
        )

        context = self._trim_context(
            context
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
- Ignore retrieved material that does not directly relate to the topic.
- Do not use outside knowledge.
- Do not invent facts.
- Do not invent formulas.
- Do not invent examples.
- Keep the explanation simple.
- Do not include unrelated topics.

Use exactly these sections:

Introduction

Main Explanation

Important Points

Formula / Syntax

Example

Conclusion

For Formula / Syntax:
use a formula or syntax only when it appears
in the uploaded study material.

For Example:
use an example related to the requested topic
from the uploaded material.

When none exists, write exactly:

No example was found in the uploaded study material.
"""

        answer = self._generate(
            prompt
        )

        # ----------------------------------
        # Add a source-grounded example if
        # the model failed to provide one.
        # ----------------------------------

        example = self._extract_example(
            context,
            topic
        )

        example_content = (
            self._get_section_content(
                answer,
                "Example"
            )
        )

        if (
            not self._example_content_is_valid(
                example_content
            )
            and example
        ):

            answer = self._replace_section(
                answer,
                "Example",
                "Example",
                example
            )

        self.record_activity(
            topic,
            "Topic explanation"
        )

        return answer

    # ==========================================
    # 2. SOLVE QUESTION
    # ==========================================

    def solve_question(
        self,
        question,
        context
    ):

        # --------------------------------------
        # Select only relevant retrieved material
        # --------------------------------------

        context = (
            self._select_relevant_context(
                question,
                context
            )
        )

        context = self._trim_context(
            context
        )

        # --------------------------------------
        # IMPORTANT:
        # Do not allow the model to invent a
        # programming solution when the uploaded
        # material does not contain one.
        # --------------------------------------

        if not self._has_solution_support(
            question,
            context
        ):

            answer = (
                "Introduction\n\n"
                "The uploaded study material contains "
                "the question, but it does not contain "
                "a complete solution or implementation.\n\n"

                "Main Explanation\n\n"
                "I could not find a complete answer or "
                "program for this question in the "
                "uploaded study material.\n\n"

                "Important Points\n\n"
                "The available material contains the "
                "question itself, but not the required "
                "implementation.\n\n"

                "Formula / Syntax\n\n"
                "Not available in the uploaded study material.\n\n"

                "Example / Application\n\n"
                "No example was found in the uploaded "
                "study material.\n\n"

                "Conclusion\n\n"
                "Please upload the relevant lecture notes, "
                "lab manual, textbook section, or other "
                "study material containing the solution."
            )

            self.record_activity(
                question,
                "Question solving"
            )

            return answer

        # --------------------------------------
        # Generate answer from supported material
        # --------------------------------------

        prompt = f"""
You are an academic question-solving assistant.

Answer the question using ONLY the relevant
uploaded study material.

QUESTION:
{question}

RELEVANT STUDY MATERIAL:
{context}

STRICT RULES:

- Use only the provided material.
- Ignore unrelated retrieved content.
- Do not use outside knowledge.
- Do not invent facts.
- Do not invent formulas.
- Do not invent examples.
- Do not invent steps.
- Do not invent code.
- Give the complete answer when the required information
  is available in the provided material.
- Do not stop in the middle of a code example.
- Keep the answer clear and suitable for a student.

Use exactly these sections:

Introduction

Main Explanation

Important Points

Formula / Syntax

Example / Application

Conclusion

For programming questions:

- Provide the complete program only when the program
  or required implementation is supported by the
  uploaded study material.
- Keep code formatting intact.
- Explain the important parts of the program.
- Include the complete code block before moving to
  the remaining sections.

When a formula is not supported by the material:

Not available in the uploaded study material.

When an example is not supported:

No example was found in the uploaded study material.

When the answer cannot be found:

I could not find the answer in the uploaded document.
"""

        answer = self._generate(
            prompt
        )

        # ----------------------------------
        # Add a reliable source example if
        # the model did not provide one.
        # ----------------------------------

        example = self._extract_example(
            context,
            question
        )

        example_content = (
            self._get_section_content(
                answer,
                "Example / Application"
            )
        )

        if (
            not self._example_content_is_valid(
                example_content
            )
            and example
        ):

            answer = self._replace_section(
                answer,
                [
                    "Example / Application",
                    "Example"
                ],
                "Example / Application",
                example
            )

        self.record_activity(
            question,
            "Question solving"
        )

        return answer

    # ==========================================
    # 3. CONTENT SYNTHESIS
    # ==========================================

    def synthesize_content(
        self,
        topic,
        context
    ):

        context = (
            self._select_relevant_context(
                topic,
                context
            )
        )

        context = self._trim_context(
            context
        )

        prompt = f"""
You are an academic content synthesis assistant.

Create a concise study guide for:

{topic}

STUDY MATERIAL:
{context}

Use ONLY information supported by the material.

Do not use outside knowledge.

Do not invent facts, examples, formulas, or steps.

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
            "Content synthesis"
        )

        return answer

    # ==========================================
    # 4. LEARNING PROGRESSION
    # ==========================================

    def learning_progression(
        self,
        topic,
        context,
        practice_questions=None
    ):
        """
        Create a source-grounded learning progression.

        practice_questions should contain questions retrieved
        from the uploaded question bank.
        """

        context = (
            self._select_relevant_context(
                topic,
                context
            )
        )

        context = self._trim_context(
            context
        )

        if practice_questions:

            practice_text = "\n".join(
                f"{i}. {question}"

                for i, question
                in enumerate(
                    practice_questions,
                    start=1
                )
            )

        else:

            practice_text = (
                "No related question-bank questions "
                "were found."
            )

        prompt = f"""
You are an academic learning assistant.

Create a learning progression for:

{topic}

RELEVANT STUDY MATERIAL:
{context}

RELATED QUESTIONS FROM THE UPLOADED QUESTION BANK:
{practice_text}

STRICT RULES:

- Use only the study material for theory and examples.
- Use only the supplied question-bank questions for practice.
- Do not invent practice questions.
- Do not invent assessment questions.
- Do not use outside knowledge.
- Do not invent formulas, facts, examples, or steps.
- Keep each section concise.

Use exactly this structure:

THEORY

Explain the topic using the relevant study material.

EXAMPLE

Give an example related to the topic from the study material.

If none exists, write:

No example was found in the uploaded study material.

PRACTICE

Show the related question-bank questions exactly as supplied.

If none exist, write:

No related practice questions were found in the uploaded question bank.

ASSESSMENT

Select questions from the supplied question bank that
can be used for self-assessment.

Do NOT create new questions.

If no separate assessment question is available, write:

No separate assessment question is available in the uploaded question bank.
"""

        answer = self._generate(
            prompt
        )

        self.record_activity(
            topic,
            "Learning progression"
        )

        return answer

    # ==========================================
    # 5. EXAM PREP GUIDE
    # ==========================================

    def exam_prep_guide(
        self,
        topic,
        context,
        practice_questions=None
    ):
        """
        Create an exam-preparation guide using only
        the supplied study material and question bank.
        """

        context = (
            self._select_relevant_context(
                topic,
                context
            )
        )

        context = self._trim_context(
            context
        )

        if practice_questions:

            question_text = "\n".join(
                f"{i}. {question}"

                for i, question
                in enumerate(
                    practice_questions,
                    start=1
                )
            )

        else:

            question_text = (
                "No related questions found "
                "in the uploaded question bank."
            )

        prompt = f"""
You are an academic exam-preparation assistant.

Topic:

{topic}

STUDY MATERIAL:
{context}

QUESTION BANK:
{question_text}

Use ONLY the supplied material.

Create:

1. THEORY

A clear explanation of the topic.

2. IMPORTANT POINTS

Only important points supported by the material.

3. EXAMPLE

A relevant example from the study material.

4. PRACTICE

Questions from the supplied question bank.

5. ASSESSMENT

Choose available question-bank questions for self-test.

Do not create new questions.

6. EXAM REVISION

A concise revision summary based only on the material.

Do not use outside knowledge.

Do not invent content.
"""

        answer = self._generate(
            prompt
        )

        self.record_activity(
            topic,
            "Exam preparation"
        )

        return answer

    # ==========================================
    # 6. RECORD ACTIVITY
    # ==========================================

    def record_activity(
        self,
        topic,
        activity
    ):

        record = {
            "topic": topic,
            "activity": activity
        }

        self.history.append(
            record
        )

        return record

    # ==========================================
    # 7. GET HISTORY
    # ==========================================

    def get_history(self):

        return self.history