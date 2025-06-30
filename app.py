import streamlit as st
import pandas as pd
import datetime
from streamlit_auth0 import login_button
from langchain.chat_models import init_chat_model

# Auth0 Credentials
client_id = st.secrets["AUTH0_CLIENT_ID"]
domain = st.secrets["AUTH0_DOMAIN"]

# Login with Google (Auth0)
from streamlit_auth0 import login_button

# Login
user_info = login_button(client_id=client_id, domain=domain)

if user_info:
    with st.sidebar:
        st.markdown("**Signed in as:**")
        st.markdown(f"{user_info['name']}")
        st.markdown(f"{user_info['email']}")
        if st.button("Sign Out"):
            st.session_state.clear()
            st.rerun()

    st.title("Assignment Grader")
    st.success(f"Welcome, {user_info['name']}!")
else:
    st.warning("Please log in with Google to continue.")
    st.stop()

model = init_chat_model("llama3-8b-8192", model_provider="groq")

# Prompt Template
prompt_template = """
You are an expert assignment grader.

Evaluate the StudentAnswer compared to the IdealAnswer using the rubric below.

## Grading Style: {grading_style}

### Rubric (Total: 10 Marks)

1. **Content Accuracy (0–3)**  
   - Are the key facts and concepts correct?

2. **Completeness (0–2)**  
   - Are all required parts of the answer included?

3. **Language & Clarity (0–2)**  
   - Is the response grammatically correct, clear, and easy to understand?

4. **Depth of Understanding (0–2)**  
   - Does the answer show critical thinking or insight?

5. **Structure & Coherence (0–1)**  
   - Is the answer logically organized and well-structured?

---

### Input:

**Question:**  
{question}

**IdealAnswer:**  
{ideal_answer}

**StudentAnswer:**  
{student_answer}

---

### Output Format (exactly as shown below):

## Marks:
- Content Accuracy: x / 3  
- Completeness: y / 2  
- Language & Clarity: z / 2  
- Depth of Understanding: p / 2  
- Structure & Coherence: q / 1  
- **Total: x+y+z+p+q / 10**

## Justification:
- **Content Accuracy**: <explanation>
- **Completeness**: <explanation>
- **Language & Clarity**: <explanation>
- **Depth of Understanding**: <explanation>
- **Structure & Coherence**: <explanation>
"""

# Grading Form 
with st.form("grading_form"):
    question = st.text_area("Enter the question")
    ideal_answer = st.text_area("Enter the ideal answer")
    student_answer = st.text_area("Enter the student's answer")
    grading_style = st.selectbox("Select grading style", ["Balanced", "Strict", "Lenient"])
    submitted = st.form_submit_button("Grade Answer")

# Session History
if "history" not in st.session_state:
    st.session_state.history = []

# Grading Logic
if submitted:
    if question and ideal_answer and student_answer:
        with st.spinner("Grading in progress..."):
            prompt = prompt_template.format(
                question=question.strip(),
                ideal_answer=ideal_answer.strip(),
                student_answer=student_answer.strip(),
                grading_style=grading_style
            )
            response = model.invoke(prompt)
            evaluation = response.content
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save to session history
            st.session_state.history.append({
                "user": user_info["email"],
                "timestamp": timestamp,
                "question": question.strip(),
                "student_answer": student_answer.strip(),
                "evaluation": evaluation.strip()
            })

            st.success("Grading completed.")

            if "## Marks:" in evaluation and "## Justification:" in evaluation:
                marks_section = evaluation.split("## Justification:")[0].replace("## Marks:", "").strip()
                explanation_section = evaluation.split("## Justification:")[1].strip()

                st.subheader("Marks Breakdown")
                st.code(marks_section, language="markdown")

                st.subheader("Justification")
                st.markdown(explanation_section)
            else:
                st.warning("Unexpected format. Full response below:")
                st.markdown(evaluation)
    else:
        st.warning("Please fill in all fields.")

# History + CSV Export
if st.session_state.history:
    with st.expander("Previous Evaluations"):
        for i, entry in enumerate(st.session_state.history[::-1], 1):
            st.markdown(f"### Attempt #{len(st.session_state.history) - i + 1}")
            st.markdown(f"**User**: {entry['user']}")
            st.markdown(f"**Time**: {entry['timestamp']}")
            st.markdown(f"**Question**: {entry['question']}")
            st.markdown(f"**Student Answer**: {entry['student_answer']}")
            st.markdown(entry["evaluation"])
            st.markdown("---")

    df = pd.DataFrame(st.session_state.history)
    st.download_button("Download Grading History", df.to_csv(index=False).encode(), "grading_history.csv", "text/csv")
