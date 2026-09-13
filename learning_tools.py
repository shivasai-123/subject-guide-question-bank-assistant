import ollama


class LearningTools:

    def __init__(self, model="llama3.2:3b"):
        self.model = model
        self.history = []


    # ==========================================
    # 1. EXPLAIN TOPIC
    # ==========================================

    def explain_topic(self, topic, context):

        prompt = f"""
You are an academic assistant.

Your job is to explain the topic using ONLY
the provided study material.

Topic:
{topic}

Study Material:
{context}


IMPORTANT RULES:

1. Use ONLY information found in the Study Material.
2. Do NOT use outside knowledge.
3. Do NOT invent examples.
4. Do NOT create hypothetical examples.
5. Do NOT add facts that are not present in the material.
6. If an example is not present in the material,
   write exactly:
   "No example was found in the uploaded study material."
7. If the topic cannot be answered from the material,
   write exactly:
   "I could not find the answer in the uploaded document."


Structure the answer as:

Introduction

Main Explanation

Important Points

Example
- Use an example ONLY if one exists in the material.
- Otherwise use the exact sentence above.

Conclusion

Keep the answer simple and suitable for a college student.
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        self.record_activity(
            topic,
            "Topic explanation"
        )

        return answer


    # ==========================================
    # 2. SOLVE QUESTION
    # ==========================================

    def solve_question(self, question, context):

        prompt = f"""
You are an academic question-solving assistant.

Answer the question using ONLY the provided
study material.

Question:
{question}

Study Material:
{context}


STRICT RULES:

1. Use ONLY information present in the Study Material.
2. Never use outside knowledge.
3. Never invent examples.
4. Never create hypothetical situations.
5. Never add facts that are not supported by the material.
6. If the question asks for an example and the material
   contains an example, use that example.
7. If the material does not contain an example, write:
   "No example was found in the uploaded study material."
8. If the answer cannot be found in the material, write:
   "I could not find the answer in the uploaded document."


For an exam-style question, structure the answer as:

Introduction

Main Explanation

Important Points

Example
- Only if supported by the material.

Conclusion

Make the answer clear, concise, and student-friendly.
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        self.record_activity(
            question,
            "Question solving"
        )

        return answer


    # ==========================================
    # 3. CONTENT SYNTHESIS
    # ==========================================

    def synthesize_content(self, topic, context):

        prompt = f"""
You are an academic content synthesis assistant.

Create a study guide for:

{topic}

Use ONLY the provided study material.

Study Material:
{context}


STRICT RULES:

- Do not use outside knowledge.
- Do not invent examples.
- Do not invent formulas.
- Do not invent steps.
- Do not add unsupported information.

Organize the answer as:

1. Overview
2. Important Concepts
3. Detailed Explanation
4. Formulas or Steps if available
5. Examples from the Material
6. Key Points for Examination
7. Summary

If a requested item is not available in the material,
clearly say that it was not found.
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        self.record_activity(
            topic,
            "Content synthesis"
        )

        return answer


    # ==========================================
    # 4. LEARNING PROGRESSION
    # ==========================================

    def learning_progression(self, topic, context):

        prompt = f"""
Create a learning progression for:

{topic}

Use ONLY the provided study material.

Study Material:
{context}


STRICT RULES:

- Do not use outside knowledge.
- Do not invent examples.
- Do not invent practice questions.
- Do not invent assessment content.

Organize the material as:

Theory
↓
Examples
↓
Practice
↓
Assessment

For each stage, use only information supported
by the uploaded study material.

If something is not available, clearly say:
"Not available in the uploaded study material."
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        self.record_activity(
            topic,
            "Learning progression"
        )

        return answer


    # ==========================================
    # 5. RECORD ACTIVITY
    # ==========================================

    def record_activity(self, topic, activity):

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