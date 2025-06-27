import streamlit as st
import pandas as pd
import requests
from langchain.chat_models import init_chat_model

# Initialize Groq model
model = init_chat_model("llama3-8b-8192", model_provider="groq")

# Google Form submission URL
GOOGLE_FORM_URL = ""
FORM_FIELD_IDS = {
    "question": "entry.1111111111",
    "student_answer": "entry.2222222222",
    "evaluation": "entry.3333333333"
}

# Prompt template
prompt_template = """
You are an expert assignment grader.

Evaluate the StudentAnswer compared to the IdealAnswer using the rubric below.

## Grading Style: {grading_style}

### Rubric
- Content Accuracy (0–4): Are the key ideas and facts correct?
- Completeness (0–3): Are all parts of the expected answer addressed?
- Language and Clarity (0–3): Is the answer clear, well-phrased, and relevant?

Assign marks for each part and provide a concise explanation for each score.

### Input:
Question: {question}

IdealAnswer: {ideal_answer}

StudentAnswer: {student_answer}

### Output format (exactly this):
## Marks:
- Content Accuracy: {{x}} / 4  
- Completeness: {{y}} / 3  
- Language/Clarity: {{z}} / 3  
- Total: {{x+y+z}} / 10

## Explanation:
{{brief explanation for each section}}
"""

# Streamlit setup
st.set_page_config(page_title="Assignment Grader (Groq)", layout="centered")
st.title("Assignment Grader")

# Input form
with st.form("grading_form"):
    question = st.text_area("Enter the question")
    ideal_answer = st.text_area("Enter the ideal answer")
    student_answer = st.text_area("Enter the student's answer")
    grading_style = st.selectbox("Select grading style", ["Balanced", "Strict", "Lenient"])
    submitted = st.form_submit_button("Grade Answer")

# Session history store
if "history" not in st.session_state:
    st.session_state.history = []

# Run the model and display structured output
if submitted:
    if question and ideal_answer and student_answer:
        with st.spinner("Grading in progress..."):
            filled_prompt = prompt_template.format(
                question=question.strip(),
                ideal_answer=ideal_answer.strip(),
                student_answer=student_answer.strip(),
                grading_style=grading_style
            )
            response = model.invoke(filled_prompt)
            evaluation = response.content

            # Save to session history
            record = {
                "question": question,
                "student_answer": student_answer,
                "evaluation": evaluation
            }
            st.session_state.history.append(record)

            # Send to Google Form if configured
            if GOOGLE_FORM_URL:
                form_data = {
                    FORM_FIELD_IDS["question"]: question,
                    FORM_FIELD_IDS["student_answer"]: student_answer,
                    FORM_FIELD_IDS["evaluation"]: evaluation
                }
                try:
                    requests.post(GOOGLE_FORM_URL, data=form_data)
                except Exception as e:
                    st.warning(f"Could not send to Google Form: {e}")

            st.success("Grading completed.")

            # Format output
            if "## Marks:" in evaluation and "## Explanation:" in evaluation:
                marks_section = evaluation.split("## Explanation:")[0].replace("## Marks:", "").strip()
                explanation_section = evaluation.split("## Explanation:")[1].strip()

                st.subheader("Marks Breakdown")
                st.code(marks_section, language="markdown")

                st.subheader("Explanation")
                st.markdown(explanation_section)
            else:
                st.warning("Unexpected output format. Showing full response:")
                st.markdown(evaluation)
    else:
        st.warning("Please complete all fields.")

# Evaluation history and download
if st.session_state.history:
    with st.expander("Previous Evaluations"):
        for idx, entry in enumerate(st.session_state.history[::-1], 1):
            st.markdown(f"### Attempt #{len(st.session_state.history) - idx + 1}")
            st.markdown(f"**Question**: {entry['question']}")
            st.markdown(f"**Student Answer**: {entry['student_answer']}")
            st.markdown(entry['evaluation'])
            st.markdown("---")

    # CSV download
    df = pd.DataFrame(st.session_state.history)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Grading History", csv, "grading_history.csv", "text/csv")
