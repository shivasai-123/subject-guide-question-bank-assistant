import ollama


class LearningTools:

    def __init__(self, model="llama3.2:3b"):
        self.model = model
        self.history = []


    # ==========================================
    # 1. TOPIC EXPLANATION
    # ==========================================

    def explain_topic(self, topic, context):

        prompt = f"""
You are an academic assistant.

Explain the following topic using ONLY the
provided study material.

Topic:
{topic}

Study Material:
{context}

Give the answer in a simple and student-friendly
format.

Include:
1. Definition
2. Explanation
3. Important points
4. Examples if available in the material
5. Short conclusion

Do not add information that is not present
in the study material.
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
    # 2. QUESTION SOLVING
    # ==========================================

    def solve_question(self, question, context):

        prompt = f"""
You are an academic question-solving assistant.

Answer the following question using ONLY
the provided study material.

Question:
{question}

Study Material:
{context}

Give a clear answer that a college student
can understand.

If the question is suitable for an exam,
structure the answer with:
- Introduction
- Main explanation
- Steps or important points
- Example if available
- Conclusion

Do not use information outside the provided
study material.

If the answer cannot be found in the material,
say exactly:

"I could not find the answer in the uploaded document."
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

Create a complete study guide for:

{topic}

Use ONLY the provided study material.

Study Material:
{context}

Combine the information from the available
sources without adding outside information.

Organize the answer as:

1. Overview
2. Important concepts
3. Detailed explanation
4. Formulas or steps if available
5. Examples
6. Key points for examination
7. Summary

Keep the explanation simple and well organized.
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
Create a learning progression for the topic:

{topic}

Use ONLY the provided study material.

Organize the learning process as:

Theory
↓
Examples
↓
Practice
↓
Assessment

For each stage, explain what the student
should learn from the provided material.

Study Material:
{context}
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

        self.history.append(record)

        return record


    # ==========================================
    # 6. GET HISTORY
    # ==========================================

    def get_history(self):

        return self.history