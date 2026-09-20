import re
import ollama


class LearningTools:

    def __init__(self, model="llama3.2:3b"):

        self.model = model
        self.history = []


    # ==========================================
    # HELPER - CALL OLLAMA
    # ==========================================

    def _generate(self, prompt):

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]


    # ==========================================
    # HELPER - NORMALIZE HEADING
    # ==========================================

    def _normalize_heading(self, text):

        text = text.strip()

        # Remove markdown heading symbols
        text = re.sub(
            r"^#+\s*",
            "",
            text
        )

        # Remove bold markers
        text = text.replace(
            "**",
            ""
        ).strip()

        # Remove trailing colon
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

            if self._normalize_heading(
                line
            ) == target:

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
            self._normalize_heading(name)
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
                "Conclusion"
            ]
        }

        lines = answer.splitlines()

        start_index = None

        for i, line in enumerate(lines):

            normalized = self._normalize_heading(
                line
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

            normalized = self._normalize_heading(
                lines[i]
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
    # HELPER - FORMULA VALIDATION
    # ==========================================

    def _formula_content_is_valid(
        self,
        content
    ):

        if not content:
            return False

        lowered = content.lower()

        invalid_phrases = [
            "not available",
            "no formula",
            "not provided",
            "none available"
        ]

        for phrase in invalid_phrases:

            if phrase in lowered:
                return False

        # Mathematical content should usually
        # contain "=" or common notation.
        if "=" in content:
            return True

        if re.search(
            r"\b[P|C]\s*\(",
            content
        ):
            return True

        if "n!" in content:
            return True

        return False


    # ==========================================
    # HELPER - EXAMPLE VALIDATION
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
    # HELPER - EXTRACT FORMULA
    # ==========================================

    def _extract_formula(
        self,
        context,
        topic=None
    ):

        lines = context.splitlines()

        topic_lower = (
            topic.lower()
            if topic
            else ""
        )


        # --------------------------------------
        # Prefer topic-specific formulas
        # --------------------------------------

        if "permutation" in topic_lower:

            permutation_patterns = [

                r"P\s*\(\s*n\s*,\s*r\s*\)\s*=\s*[^\n]+",

                r"P\s*\(\s*n\s*,\s*r\s*\)\s*=\s*.*"
            ]

            for pattern in permutation_patterns:

                match = re.search(
                    pattern,
                    context,
                    re.IGNORECASE
                )

                if match:

                    return match.group(
                        0
                    ).strip()


        if "combination" in topic_lower:

            combination_patterns = [

                r"C\s*\(\s*n\s*,\s*r\s*\)\s*=\s*[^\n]+",

                r"C\s*\(\s*n\s*,\s*r\s*\)\s*=\s*.*"
            ]

            for pattern in combination_patterns:

                match = re.search(
                    pattern,
                    context,
                    re.IGNORECASE
                )

                if match:

                    return match.group(
                        0
                    ).strip()


        # --------------------------------------
        # Look for explicit formula lines
        # --------------------------------------

        for line in lines:

            clean_line = line.strip()

            if not clean_line:
                continue

            if re.search(
                r"\bformula\b",
                clean_line,
                re.IGNORECASE
            ):

                if "=" in clean_line:

                    return clean_line


        # --------------------------------------
        # General formula patterns
        # --------------------------------------

        formula_patterns = [

            r"[A-Za-zΔρσλD][A-Za-z0-9₀-₉]*\s*=\s*[^\n]+",

            r"[A-Za-z]\s*\(\s*[^\)]*\)\s*=\s*[^\n]+",

            r"\b[A-Za-z]+\s*=\s*[^\n]+"
        ]


        for pattern in formula_patterns:

            match = re.search(
                pattern,
                context
            )

            if match:

                formula = match.group(
                    0
                ).strip()

                # Ignore extremely long sentences
                if len(formula) < 200:

                    return formula


        return None


    # ==========================================
    # HELPER - EXTRACT RELEVANT EXAMPLE
    # ==========================================

    def _extract_example(
        self,
        context,
        topic=None
    ):

        lines = context.splitlines()

        topic_terms = []

        if topic:

            stop_words = {
                "what",
                "is",
                "are",
                "the",
                "an",
                "a",
                "of",
                "for",
                "explain",
                "define",
                "about",
                "how",
                "why"
            }

            topic_terms = [
                word
                for word in re.findall(
                    r"[A-Za-z]+",
                    topic.lower()
                )
                if len(word) >= 4
                and word not in stop_words
            ]


        candidates = []


        # --------------------------------------
        # Find possible examples
        # --------------------------------------

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


            example_lines = []

            example_lines.append(
                clean_line
            )


            # ----------------------------------
            # Collect following lines
            # ----------------------------------

            for j in range(
                i + 1,
                min(
                    i + 10,
                    len(lines)
                )
            ):

                next_line = lines[j].strip()

                if not next_line:
                    continue


                # Chunk separator
                if next_line.startswith(
                    "----------------"
                ):
                    break


                # Stop when another example starts
                if (
                    j > i + 1
                    and re.search(
                        r"^(?:[A-Za-z ]+\s+)?Example\b",
                        next_line,
                        re.IGNORECASE
                    )
                ):
                    break


                # Stop at a major slide heading
                if (
                    j > i + 2
                    and re.search(
                        r"^Slide\s+\d+",
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


            # ----------------------------------
            # Score relevance
            # ----------------------------------

            score = 0

            block_lower = block.lower()

            for term in topic_terms:

                if term in block_lower:
                    score += 10


            if "problem:" in block_lower:
                score += 3


            if "solution:" in block_lower:
                score += 5


            if "=" in block:
                score += 2


            candidates.append(
                (
                    score,
                    block
                )
            )


        if not candidates:
            return None


        # Highest relevance first
        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )


        return candidates[0][1]


    # ==========================================
    # HELPER - FIND SECTION LINE
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
            self._normalize_heading(name)
            for name in section_names
        }


        lines = answer.splitlines()


        for i, line in enumerate(lines):

            if self._normalize_heading(
                line
            ) in targets:

                return i


        return None


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

        lines = answer.splitlines()


        if isinstance(
            section_names,
            str
        ):

            section_names = [
                section_names
            ]


        start_line = self._find_section_line(
            answer,
            section_names
        )


        # --------------------------------------
        # Section does not exist
        # --------------------------------------

        if start_line is None:

            return self._insert_before_conclusion(
                answer,
                correct_heading,
                content
            )


        # --------------------------------------
        # Find next section
        # --------------------------------------

        known_sections = [
            "Introduction",
            "Main Explanation",
            "Important Points",
            "Formula / Syntax",
            "Formula",
            "Example",
            "Example / Application",
            "Conclusion"
        ]


        normalized_known = {
            self._normalize_heading(name)
            for name in known_sections
        }


        end_line = len(lines)


        for i in range(
            start_line + 1,
            len(lines)
        ):

            if self._normalize_heading(
                lines[i]
            ) in normalized_known:

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
    # HELPER - INSERT BEFORE CONCLUSION
    # ==========================================

    def _insert_before_conclusion(
        self,
        answer,
        section,
        content
    ):

        lines = answer.splitlines()


        conclusion_line = self._find_section_line(
            answer,
            "Conclusion"
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
                + lines[conclusion_line:]
            ).strip()


        return (
            answer.rstrip()
            + "\n\n"
            + section
            + "\n\n"
            + content.strip()
        )


    # ==========================================
    # HELPER - COMPLETE ANSWER STRUCTURE
    # ==========================================

    def _complete_answer_structure(
        self,
        answer,
        context,
        topic=None,
        example_heading="Example / Application"
    ):

        # ======================================
        # 1. FORMULA
        # ======================================

        formula = self._extract_formula(
            context,
            topic
        )


        formula_content = self._get_section_content(
            answer,
            [
                "Formula / Syntax",
                "Formula"
            ]
        )


        formula_valid = self._formula_content_is_valid(
            formula_content
        )


        if formula:

            # Always replace an invalid formula
            # section with source-derived formula.
            if not formula_valid:

                answer = self._replace_section(
                    answer,
                    [
                        "Formula / Syntax",
                        "Formula"
                    ],
                    "Formula / Syntax",
                    formula
                )

        else:

            if not formula_valid:

                answer = self._replace_section(
                    answer,
                    [
                        "Formula / Syntax",
                        "Formula"
                    ],
                    "Formula / Syntax",
                    "Not available in the uploaded study material."
                )


        # ======================================
        # 2. EXAMPLE
        # ======================================

        example = self._extract_example(
            context,
            topic
        )


        example_content = self._get_section_content(
            answer,
            [
                example_heading,
                "Example",
                "Example / Application"
            ]
        )


        example_valid = self._example_content_is_valid(
            example_content
        )


        if example:

            if not example_valid:

                answer = self._replace_section(
                    answer,
                    [
                        example_heading,
                        "Example",
                        "Example / Application"
                    ],
                    example_heading,
                    example
                )

        else:

            if not example_valid:

                answer = self._replace_section(
                    answer,
                    [
                        example_heading,
                        "Example",
                        "Example / Application"
                    ],
                    example_heading,
                    "No example was found in the uploaded study material."
                )


        return answer


    # ==========================================
    # 1. EXPLAIN TOPIC
    # ==========================================

    def explain_topic(
        self,
        topic,
        context
    ):

        prompt = f"""
You are an academic assistant.

Explain the requested topic using ONLY the
provided study material.

TOPIC:
{topic}

STUDY MATERIAL:
{context}

==================================================
STRICT RULES
==================================================

1. Use ONLY information contained in the Study Material.

2. Do NOT use outside knowledge.

3. Do NOT invent facts.

4. Do NOT invent examples.

5. Do NOT create hypothetical examples.

6. Do NOT invent formulas.

7. Do NOT invent steps.

8. Do NOT add unsupported information.

9. Inspect ALL retrieved sources carefully.

==================================================
EXAMPLE RULE
==================================================

A valid example includes:

- Example sections
- Worked examples
- Problems with solutions
- Numerical problems
- Problem + Solution
- Step-by-step demonstrations
- Code examples
- Formula applications
- Practical applications

If a relevant example exists, include it.

Do NOT invent examples.

==================================================
FORMULA RULE
==================================================

If a formula appears anywhere in the Study Material:

- Include the actual formula.
- Preserve it as closely as possible.
- Do not modify it.

==================================================
ANSWER STRUCTURE
==================================================

Use EXACTLY these sections:

Introduction

Main Explanation

Important Points

Formula / Syntax

Example

Conclusion

Do not omit any section.

In Formula / Syntax:
include a formula or syntax only when it
actually exists in the Study Material.

In Example:
use an actual example from the Study Material.

If no relevant example exists:

No example was found in the uploaded study material.

Keep the answer simple and suitable for
a college student.
"""


        answer = self._generate(
            prompt
        )


        answer = self._complete_answer_structure(
            answer,
            context,
            topic,
            "Example"
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

        prompt = f"""
You are an academic question-solving assistant.

Answer the question using ONLY the provided
Study Material.

QUESTION:
{question}

STUDY MATERIAL:
{context}

==================================================
STRICT RULES
==================================================

1. Use ONLY information present in the Study Material.

2. Never use outside knowledge.

3. Never invent facts.

4. Never invent examples.

5. Never create hypothetical situations.

6. Never invent formulas.

7. Never invent steps.

8. Never add unsupported information.

9. Inspect ALL retrieved sources before answering.

==================================================
SOURCE ANALYSIS
==================================================

Look for:

- Definitions
- Formulas
- Syntax
- Worked examples
- Problems
- Solutions
- Numerical calculations
- Code examples
- Algorithms
- Procedures
- Step-by-step solutions
- Applications
- Important points
- Question-bank content

==================================================
EXAMPLE RULE
==================================================

If a relevant example exists anywhere in the
Study Material:

YOU MUST INCLUDE IT.

Use the actual example.

Do not replace it with your own example.

==================================================
FORMULA RULE
==================================================

If a relevant formula exists anywhere in the
Study Material:

- Include the actual formula.
- Preserve it.
- Do not create a formula from outside knowledge.

==================================================
ANSWER STRUCTURE
==================================================

You MUST use exactly:

Introduction

Main Explanation

Important Points

Formula / Syntax

Example / Application

Conclusion

Do not omit sections.

If a formula exists, include it.

If no formula exists:

Not available in the uploaded study material.

If an example exists, include it.

If no example exists:

No example was found in the uploaded study material.

==================================================
PROGRAMMING QUESTIONS
==================================================

For programming questions:

- Use code only if code appears in the
  Study Material.
- Do not invent code.
- Do not complete missing code using
  outside knowledge.

==================================================
QUESTION BANK QUESTIONS
==================================================

If the question comes from a Question Bank:

- Preserve the question wording when useful.
- Answer only from the retrieved material.
- Do not invent an answer absent from the sources.

==================================================
ANSWER NOT FOUND
==================================================

If the question cannot be answered from the
Study Material, write exactly:

"I could not find the answer in the uploaded document."

Keep the answer clear and suitable for
a college student.
"""


        answer = self._generate(
            prompt
        )


        answer = self._complete_answer_structure(
            answer,
            context,
            question,
            "Example / Application"
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

        prompt = f"""
You are an academic content synthesis assistant.

Create a study guide for:

{topic}

STUDY MATERIAL:
{context}

==================================================
STRICT RULES
==================================================

- Use ONLY the Study Material.
- Do not use outside knowledge.
- Do not invent facts.
- Do not invent examples.
- Do not invent formulas.
- Do not invent steps.
- Do not invent practice content.
- Do not add unsupported information.

==================================================
EXTRACT CAREFULLY
==================================================

Look for:

- Definitions
- Important concepts
- Formulas
- Syntax
- Procedures
- Worked examples
- Numerical examples
- Problems
- Solutions
- Code examples
- Applications
- Exam points

A worked problem with a solution is an example.

==================================================
ANSWER STRUCTURE
==================================================

1. Overview

2. Important Concepts

3. Detailed Explanation

4. Formulas or Steps

5. Examples from the Material

6. Applications from the Material

7. Key Points for Examination

8. Summary

Use actual examples from the Study Material.

If no example exists:

"Not available in the uploaded study material."

Use only applications explicitly supported
by the Study Material.

Do not silently fill gaps with general knowledge.
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
        context
    ):

        prompt = f"""
Create a learning progression for:

{topic}

STUDY MATERIAL:
{context}

==================================================
USE ONLY THE PROVIDED STUDY MATERIAL
==================================================

STRICT RULES:

- Do not use outside knowledge.
- Do not invent examples.
- Do not invent practice questions.
- Do not invent assessment questions.
- Do not invent formulas.
- Do not invent steps.
- Do not add unsupported information.

==================================================
IDENTIFY ACTUAL CONTENT
==================================================

Carefully identify material for:

Theory
Examples
Practice
Assessment

An item counts as an example if the material
contains:

- A worked problem
- A solved numerical question
- A code demonstration
- A practical application
- A real-world application
- A step-by-step demonstration
- A problem followed by a solution

==================================================
STRUCTURE
==================================================

Theory
↓
Examples
↓
Practice
↓
Assessment

For each stage:

- Use only supported material.
- Never invent missing content.

If a stage has no supporting content, write:

"Not available in the uploaded study material."
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
    # 5. RECORD ACTIVITY
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
    # 6. GET HISTORY
    # ==========================================

    def get_history(self):

        return self.history