import streamlit as st
import pandas as pd
import datetime
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from streamlit_auth0 import login_button
from langchain.chat_models import init_chat_model
import os # Import os for environment variables

# Auth0 credentials
client_id = st.secrets["AUTH0_CLIENT_ID"]
domain = st.secrets["AUTH0_DOMAIN"]

# Login via Auth0
user_info = login_button(client_id=client_id, domain=domain)

# Display user info and sign out button if logged in
if user_info:
    with st.sidebar:
        st.markdown("**Signed in as:**")
        st.markdown(f"**Name:** {user_info['name']}")
        st.markdown(f"**Email:** {user_info['email']}")
        if st.button("Sign Out"):
            st.session_state.clear()
            st.rerun()

    st.title("AI Assignment Grader")
    st.success(f"Welcome, {user_info['name']}! Please provide assignment details below.")
else:
    st.warning("Please log in with Google to continue using the Assignment Grader.")
    st.stop()

# Google Sheets Authentication
creds_dict = st.secrets["gcp_service_account"]
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(creds_dict), scope)

try:
    client = gspread.authorize(creds)
    sheet = client.open("autograde_logs").sheet1
except Exception as e:
    st.error(f"Error connecting to Google Sheets: {e}. Please ensure your 'gcp_service_account' secrets are correctly configured and the service account has access to the 'autograde_logs' spreadsheet.")
    st.stop()

expected_headers = ["User", "DateTime", "Question", "StudentAnswer", "Evaluation", "Feedback", "DetailedFeedback", "GeneratedInstruction"]
current_headers = sheet.row_values(1)
if not current_headers:
    sheet.insert_row(expected_headers, 1)
else:
    for header in expected_headers:
        if header not in current_headers:
            st.warning(f"The column '{header}' is missing in your 'autograde_logs' Google Sheet. Please add it manually for full logging functionality.")

# Set Groq API Key from Streamlit secrets as an environment variable
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Groq API key not found in Streamlit secrets. Please add it to .streamlit/secrets.toml (e.g., GROQ_API_KEY = 'your_key_here')")
    st.stop()

# Initialize the grading LLM
grade_model = init_chat_model("llama3-8b-8192", model_provider="groq")

# Initialize the prompt refinement LLM
refine_model = init_chat_model("llama3-8b-8192", model_provider="groq")


# Prompt template for the assignment grader
base_prompt_template = """
You are an expert assignment grader.

Evaluate the StudentAnswer compared to the IdealAnswer using the rubric below.

## Grading Style: {grading_style}

Grading behavior based on style:
- **Strict**: Award marks only when criteria are fully met. Be conservative and critical.
- **Balanced**: Fairly reward effort and partial correctness while maintaining academic standards.
- **Lenient**: Be generous. Prioritize student effort and intent. Give benefit of doubt for minor issues.

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

# Prompt template for the self-improvement (refining instructions) LLM
refine_prompt_template = """
You are an expert AI assistant that helps refine grading instructions.
A user has provided feedback on an automated assignment grading. Your task is to generate *new, concise, and generalizable instructions* that can be prepended to the main grading prompt to improve its accuracy based on the user's feedback.

Here's the context:

---
Original Question:
{question}

Ideal Answer:
{ideal_answer}

Student Answer:
{student_answer}

AI's Original Evaluation:
{ai_evaluation}

User's Detailed Feedback:
{detailed_feedback}
---

Based on the user's feedback, formulate *new, specific, and actionable instructions* (1-3 sentences) that should be added to the *very beginning* of the grading prompt. These instructions should address the observed discrepancy and guide the AI for future similar grading tasks.
If the feedback indicates no specific instruction is needed for future improvement, output "NO_IMPROVEMENT_NEEDED".

Example of desired output:
"For questions involving historical dates, ensure strict accuracy. Deduct points heavily if dates are incorrect or missing."

Another example:
"When evaluating 'Completeness', ensure all sub-parts explicitly mentioned in the question are addressed, even if briefly."
"""

# Session state initialization for persistent data within the session
if "history" not in st.session_state:
    st.session_state.history = []
if "last_eval" not in st.session_state:
    st.session_state.last_eval = None
if "just_graded" not in st.session_state:
    st.session_state.just_graded = False
if "current_adaptive_instruction" not in st.session_state:
    st.session_state.current_adaptive_instruction = ""

# Assignment Grading Form
with st.form("grading_form"):
    st.subheader("Assignment Details")
    question = st.text_area("Enter the question:", key="question_input", height=100)
    ideal_answer = st.text_area("Enter the ideal answer:", key="ideal_answer_input", height=150)
    student_answer = st.text_area("Enter the student's answer:", key="student_answer_input", height=150)
    grading_style = st.selectbox("Select grading style:", ["Balanced", "Strict", "Lenient"], key="grading_style_select")
    
    submit_button = st.form_submit_button("Grade Answer")

# Grading Logic
if submit_button:
    if question and ideal_answer and student_answer:
        with st.spinner("Grading in progress... Please wait."):
            full_grading_prompt = ""
            if st.session_state.current_adaptive_instruction:
                full_grading_prompt += f"**IMPORTANT ADAPTIVE INSTRUCTION:** {st.session_state.current_adaptive_instruction}\n\n"
            full_grading_prompt += base_prompt_template.format(
                question=question.strip(),
                ideal_answer=ideal_answer.strip(),
                student_answer=student_answer.strip(),
                grading_style=grading_style
            )

            try:
                response = grade_model.invoke(full_grading_prompt)
                evaluation = response.content
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.session_state.history.append({
                    "user": user_info["email"],
                    "timestamp": timestamp,
                    "question": question.strip(),
                    "ideal_answer": ideal_answer.strip(),
                    "student_answer": student_answer.strip(),
                    "evaluation": evaluation.strip(),
                    "feedback": "",
                    "detailed_feedback": "",
                    "generated_instruction": ""
                })

                st.session_state.last_eval = {
                    "email": user_info["email"],
                    "timestamp": timestamp,
                    "question": question.strip(),
                    "ideal_answer": ideal_answer.strip(),
                    "student_answer": student_answer.strip(),
                    "evaluation": evaluation.strip()
                }

                st.session_state.just_graded = True
                st.success("Grading completed successfully!")
            except Exception as e:
                st.error(f"Error during AI grading: {e}. Please try again.")
                st.session_state.just_graded = False
    else:
        st.warning("Please ensure all fields (Question, Ideal Answer, Student Answer) are filled before grading.")

# Display Grading Result and Feedback Form
if st.session_state.get("just_graded", False) and st.session_state.last_eval:
    evaluation = st.session_state.last_eval["evaluation"]

    st.subheader("Evaluation Result")
    if "## Marks:" in evaluation and "## Justification:" in evaluation:
        marks_section = evaluation.split("## Justification:")[0].replace("## Marks:", "").strip()
        explanation_section = evaluation.split("## Justification:")[1].strip()

        st.markdown("---")
        st.subheader("Marks Breakdown")
        st.code(marks_section, language="markdown")

        st.subheader("Justification")
        st.markdown(explanation_section)
        st.markdown("---")
    else:
        st.warning("The AI's evaluation format was unexpected. Displaying full response:")
        st.markdown(evaluation)
        st.markdown("---")

    # Feedback Form for Self-Improvement
    with st.form("feedback_form"):
        st.subheader("Provide Feedback on this Evaluation")
        satisfaction = st.radio(
            "Are you satisfied with this grading?",
            ["Yes", "No"],
            key="satisfaction_radio",
            help="Your feedback helps us improve the grading accuracy."
        )
        detailed_feedback_text = ""
        if satisfaction == "No":
            detailed_feedback_text = st.text_area(
                "If 'No', please explain what was wrong or how it should have been graded differently (e.g., 'Content Accuracy was too high, it missed X fact'). This detailed feedback will be used to improve the AI's grading logic for future assignments.",
                key="detailed_feedback_text_area",
                height=100
            )
        
        submit_feedback_button = st.form_submit_button("Submit Feedback")

    # Logic to process feedback and trigger prompt improvement
    if submit_feedback_button:
        # --- FIX START ---
        # If user is not satisfied but didn't provide detailed feedback, warn them
        if satisfaction == "No" and not detailed_feedback_text.strip():
            st.warning("Please provide detailed feedback if you are not satisfied, so the AI can learn and improve.")
            st.session_state.just_graded = True # Keep the feedback form visible
            st.stop() # Stop execution here to prevent logging and prompt refinement without feedback
        # --- FIX END ---

        st.session_state.history[-1]["feedback"] = satisfaction
        st.session_state.history[-1]["detailed_feedback"] = detailed_feedback_text.strip()

        generated_instruction = ""
        if satisfaction == "No" and detailed_feedback_text.strip():
            with st.spinner("Analyzing feedback and generating prompt improvement..."):
                refine_prompt = refine_prompt_template.format(
                    question=st.session_state.last_eval["question"],
                    ideal_answer=st.session_state.last_eval["ideal_answer"],
                    student_answer=st.session_state.last_eval["student_answer"],
                    ai_evaluation=st.session_state.last_eval["evaluation"],
                    detailed_feedback=detailed_feedback_text.strip()
                )
                try:
                    refine_response = refine_model.invoke(refine_prompt)
                    generated_instruction = refine_response.content.strip()

                    if generated_instruction and generated_instruction != "NO_IMPROVEMENT_NEEDED":
                        st.session_state.current_adaptive_instruction = generated_instruction
                        st.info(f"**New Adaptive Instruction Generated:** '{generated_instruction}'\n\nThis instruction will be applied to all subsequent gradings in this session to improve accuracy.")
                    else:
                        st.info("No specific instruction generated for improvement based on your feedback, or feedback indicated no improvement needed.")
                        st.session_state.current_adaptive_instruction = ""
                except Exception as e:
                    st.error(f"Error during prompt refinement: {e}. The grading prompt could not be improved at this time.")
                    st.session_state.current_adaptive_instruction = ""
        else:
            st.info("Feedback submitted. No prompt improvement needed for positive feedback.")
            st.session_state.current_adaptive_instruction = ""

        st.session_state.history[-1]["generated_instruction"] = generated_instruction

        cleaned_eval = st.session_state.last_eval["evaluation"].replace("\n", " ⏎ ")

        row_to_append = [
            st.session_state.last_eval["email"],
            st.session_state.last_eval["timestamp"],
            st.session_state.last_eval["question"],
            st.session_state.last_eval["student_answer"],
            cleaned_eval,
            satisfaction,
            st.session_state.history[-1]["detailed_feedback"],
            generated_instruction
        ]

        try:
            sheet.append_row(row_to_append)
            st.success("Feedback recorded and grading log updated in Google Sheets.")
        except Exception as e:
            st.error(f"Error appending data to Google Sheet: {e}. Please check permissions and sheet name.")

        st.session_state.just_graded = False

# Display Current Session Evaluations
if st.session_state.history:
    st.markdown("---")
    with st.expander("View Your Current Session Evaluations"):
        for i, entry in enumerate(st.session_state.history[::-1], 1):
            st.markdown(f"### Evaluation #{len(st.session_state.history) - i + 1}")
            st.markdown(f"**User:** {entry['user']}")
            st.markdown(f"**Time:** {entry['timestamp']}")
            st.markdown(f"**Question:** {entry['question']}")
            st.markdown(f"**Student Answer:** {entry['student_answer']}")
            st.markdown("**AI Evaluation:**")
            st.markdown(entry["evaluation"])
            if entry["feedback"]:
                st.markdown(f"**User Feedback:** {entry['feedback']}")
            if entry["detailed_feedback"]:
                st.markdown(f"**Detailed Feedback:** {entry['detailed_feedback']}")
            if entry["generated_instruction"]:
                st.markdown(f"**Generated Instruction:** {entry['generated_instruction']}")
            st.markdown("---")

    df = pd.DataFrame(st.session_state.history)
    st.download_button(
        "Download Current Session Data (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "grading_session.csv",
        "text/csv",
        help="Download all grading records from your current session."
    )

# Display Global Grading History from Google Sheet
st.markdown("---")
if st.checkbox("Show All Grading History (from Google Sheet)"):
    try:
        with st.spinner("Loading all grading history from Google Sheet..."):
            records = sheet.get_all_records()
            df_all = pd.DataFrame(records)
            st.dataframe(df_all)

        st.download_button(
            "Download All Grading History (CSV)",
            df_all.to_csv(index=False).encode("utf-8"),
            "grading_all_users.csv",
            "text/csv",
            help="Download all grading records from all users stored in the Google Sheet."
        )
    except Exception as e:
        st.error(f"Error loading all grading history from Google Sheet: {e}. Please ensure the sheet is accessible and correctly formatted.")
