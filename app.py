import streamlit as st
import pandas as pd
import datetime
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from streamlit_auth0 import login_button
from langchain.chat_models import init_chat_model

# Auth0 credentials
# Ensure these are set in your Streamlit secrets (e.g., .streamlit/secrets.toml)
# [AUTH0]
# CLIENT_ID = "your_auth0_client_id"
# DOMAIN = "your_auth0_domain"
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
            st.rerun() # Rerun to clear the session and show login button

    st.title("AI Assignment Grader")
    st.success(f"Welcome, {user_info['name']}! Please provide assignment details below.")
else:
    st.warning("Please log in with Google to continue using the Assignment Grader.")
    st.stop() # Stop execution until the user logs in

# Google Sheets Authentication
# Ensure 'gcp_service_account' is set in your Streamlit secrets
# [gcp_service_account]
# type = "service_account"
# project_id = "your-project-id"
# private_key_id = "your-private-key-id"
# private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
# client_email = "your-service-account-email@your-project-id.iam.gserviceaccount.com"
# client_id = "your-client-id"
# auth_uri = "https://accounts.google.com/o/oauth2/auth"
# token_uri = "https://oauth2.googleapis.com/token"
# auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
# client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
creds_dict = st.secrets["gcp_service_account"]
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(creds_dict), scope)

try:
    client = gspread.authorize(creds)
    # Open the main grading log sheet
    sheet = client.open("autograde_logs").sheet1
except Exception as e:
    st.error(f"Error connecting to Google Sheets: {e}. Please ensure your 'gcp_service_account' secrets are correctly configured and the service account has access to the 'autograde_logs' spreadsheet.")
    st.stop()

# Ensure headers exist in the Google Sheet for proper logging
expected_headers = ["User", "DateTime", "Question", "StudentAnswer", "Evaluation", "Feedback", "DetailedFeedback", "GeneratedInstruction"]
current_headers = sheet.row_values(1)
if not current_headers:
    # If sheet is empty, insert all expected headers
    sheet.insert_row(expected_headers, 1)
else:
    # Check for missing headers and warn the user
    for header in expected_headers:
        if header not in current_headers:
            st.warning(f"The column '{header}' is missing in your 'autograde_logs' Google Sheet. Please add it manually for full logging functionality.")

# Initialize the grading LLM
# This model will perform the actual assignment grading.
grade_model = init_chat_model("llama3-8b-8192", model_provider="groq")

# Initialize the prompt refinement LLM
# This model will generate new instructions based on user feedback to improve the grading prompt.
refine_model = init_chat_model("llama3-8b-8192", model_provider="groq")


# Base prompt template for the assignment grader
# This template defines the core instructions and rubric for grading.
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
# This prompt guides the AI to generate new, actionable instructions for the main grading prompt.
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
    st.session_state.history = [] # Stores a log of all evaluations in the current session
if "last_eval" not in st.session_state:
    st.session_state.last_eval = None # Stores the details of the most recent evaluation
if "just_graded" not in st.session_state:
    st.session_state.just_graded = False # Flag to control display of feedback form
if "current_adaptive_instruction" not in st.session_state:
    # This variable stores the latest generated adaptive instruction.
    # It will be prepended to the grading prompt for subsequent gradings in the session.
    # For persistence across sessions, this would need to be loaded from a persistent store (e.g., another Google Sheet cell).
    st.session_state.current_adaptive_instruction = ""

# Assignment Grading Form
# This form allows the user to input the question, ideal answer, and student's answer.
with st.form("grading_form"):
    st.subheader("Assignment Details")
    question = st.text_area("Enter the question:", key="question_input", height=100)
    ideal_answer = st.text_area("Enter the ideal answer:", key="ideal_answer_input", height=150)
    student_answer = st.text_area("Enter the student's answer:", key="student_answer_input", height=150)
    grading_style = st.selectbox("Select grading style:", ["Balanced", "Strict", "Lenient"], key="grading_style_select")
    
    # Submit button for the grading form
    submitted = st.form_submit_button("Grade Answer")

# Grading Logic: Executed when the "Grade Answer" button is pressed
if submitted:
    if question and ideal_answer and student_answer:
        with st.spinner("Grading in progress... Please wait."):
            # Construct the full prompt for the grading LLM.
            # Any adaptive instructions generated from previous feedback will be prepended.
            full_grading_prompt = ""
            if st.session_state.current_adaptive_instruction:
                full_grading_prompt += f"**IMPORTANT ADAPTIVE INSTRUCTION:** {st.session_state.current_adaptive_instruction}\n\n"
            full_grading_prompt += base_prompt_template.format(
                question=question.strip(),
                ideal_answer=ideal_answer.strip(),
                student_answer=student_answer.strip(),
                grading_style=grading_style
            )

            # Invoke the grading LLM to get the evaluation
            try:
                response = grade_model.invoke(full_grading_prompt)
                evaluation = response.content
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Store the current evaluation details in session history
                st.session_state.history.append({
                    "user": user_info["email"],
                    "timestamp": timestamp,
                    "question": question.strip(),
                    "ideal_answer": ideal_answer.strip(), # Stored for context in feedback loop
                    "student_answer": student_answer.strip(),
                    "evaluation": evaluation.strip(),
                    "feedback": "",  # To be filled by user feedback
                    "detailed_feedback": "", # To be filled by user feedback
                    "generated_instruction": "" # To be filled if self-improvement occurs
                })

                # Store the last evaluation details for display and feedback processing
                st.session_state.last_eval = {
                    "email": user_info["email"],
                    "timestamp": timestamp,
                    "question": question.strip(),
                    "ideal_answer": ideal_answer.strip(),
                    "student_answer": student_answer.strip(),
                    "evaluation": evaluation.strip()
                }

                st.session_state.just_graded = True # Set flag to display feedback form
                st.success("Grading completed successfully!")
            except Exception as e:
                st.error(f"Error during AI grading: {e}. Please try again.")
                st.session_state.just_graded = False # Reset flag on error
    else:
        st.warning("Please ensure all fields (Question, Ideal Answer, Student Answer) are filled before grading.")

# Display Grading Result and Feedback Form
# This section is displayed only after an assignment has been graded.
if st.session_state.get("just_graded", False) and st.session_state.last_eval:
    evaluation = st.session_state.last_eval["evaluation"]

    st.subheader("AI's Evaluation Result")
    # Parse the evaluation content into Marks and Justification sections
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
    # This form collects user satisfaction and detailed feedback to refine the grading prompt.
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
        
        # Submit button for the feedback form
        submit_feedback_button = st.form_submit_button("Submit Feedback")

    # Logic to process feedback and trigger prompt improvement
    if submit_feedback_button:
        # Update the last entry in the session history with feedback details
        st.session_state.history[-1]["feedback"] = satisfaction
        st.session_state.history[-1]["detailed_feedback"] = detailed_feedback_text

        generated_instruction = ""
        # If user is not satisfied and provided detailed feedback, trigger prompt refinement
        if satisfaction == "No" and detailed_feedback_text:
            with st.spinner("Analyzing feedback and generating prompt improvement..."):
                # Prepare prompt for the refinement LLM
                refine_prompt = refine_prompt_template.format(
                    question=st.session_state.last_eval["question"],
                    ideal_answer=st.session_state.last_eval["ideal_answer"],
                    student_answer=st.session_state.last_eval["student_answer"],
                    ai_evaluation=st.session_state.last_eval["evaluation"],
                    detailed_feedback=detailed_feedback_text
                )
                try:
                    refine_response = refine_model.invoke(refine_prompt)
                    generated_instruction = refine_response.content.strip()

                    # Update the current adaptive instruction if a new one is generated
                    if generated_instruction and generated_instruction != "NO_IMPROVEMENT_NEEDED":
                        st.session_state.current_adaptive_instruction = generated_instruction
                        st.info(f"**New Adaptive Instruction Generated:** '{generated_instruction}'\n\nThis instruction will be applied to all subsequent gradings in this session to improve accuracy.")
                    else:
                        st.info("No specific instruction generated for improvement based on your feedback, or feedback indicated no improvement needed.")
                        st.session_state.current_adaptive_instruction = "" # Clear if no improvement needed
                except Exception as e:
                    st.error(f"Error during prompt refinement: {e}. The grading prompt could not be improved at this time.")
                    st.session_state.current_adaptive_instruction = "" # Ensure it's cleared on error
        else:
            st.info("Feedback submitted. No prompt improvement needed for positive feedback or empty detailed feedback.")
            st.session_state.current_adaptive_instruction = "" # Clear if positive feedback or no detailed feedback

        # Update the generated instruction in the session history for logging
        st.session_state.history[-1]["generated_instruction"] = generated_instruction

        # Prepare evaluation content for Google Sheets (replace newlines for single cell)
        cleaned_eval = st.session_state.last_eval["evaluation"].replace("\n", " ⏎ ")

        # Prepare the row to append to the Google Sheet
        row_to_append = [
            st.session_state.last_eval["email"],
            st.session_state.last_eval["timestamp"],
            st.session_state.last_eval["question"],
            st.session_state.last_eval["student_answer"],
            cleaned_eval,
            satisfaction,
            detailed_feedback_text,
            generated_instruction
        ]

        # Append the data to the Google Sheet
        try:
            sheet.append_row(row_to_append)
            st.success("Feedback recorded and grading log updated in Google Sheets.")
        except Exception as e:
            st.error(f"Error appending data to Google Sheet: {e}. Please check permissions and sheet name.")

        st.session_state.just_graded = False # Reset flag to hide feedback form for next grading cycle

# Display Current Session Evaluations
# This section shows a history of all gradings performed in the current browser session.
if st.session_state.history:
    st.markdown("---")
    with st.expander("View Your Current Session Evaluations"):
        for i, entry in enumerate(st.session_state.history[::-1], 1): # Iterate in reverse for most recent first
            st.markdown(f"### Evaluation #{len(st.session_state.history) - i + 1}")
            st.markdown(f"**User:** {entry['user']}")
            st.markdown(f"**Time:** {entry['timestamp']}")
            st.markdown(f"**Question:** {entry['question']}")
            st.markdown(f"**Student Answer:** {entry['student_answer']}")
            st.markdown("**AI Evaluation:**")
            st.markdown(entry["evaluation"]) # Display original formatted evaluation
            if entry["feedback"]:
                st.markdown(f"**User Feedback:** {entry['feedback']}")
            if entry["detailed_feedback"]:
                st.markdown(f"**Detailed Feedback:** {entry['detailed_feedback']}")
            if entry["generated_instruction"]:
                st.markdown(f"**Generated Instruction:** {entry['generated_instruction']}")
            st.markdown("---")

    # Download button for current session data
    df = pd.DataFrame(st.session_state.history)
    st.download_button(
        "Download Current Session Data (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "grading_session.csv",
        "text/csv",
        help="Download all grading records from your current session."
    )

# Display Global Grading History from Google Sheet
# This allows users to view all historical grading data stored in the Google Sheet.
st.markdown("---")
if st.checkbox("Show All Grading History (from Google Sheet)"):
    try:
        with st.spinner("Loading all grading history from Google Sheet..."):
            records = sheet.get_all_records() # Fetch all records from the sheet
            df_all = pd.DataFrame(records)
            st.dataframe(df_all) # Display as a Streamlit dataframe

        # Download button for all historical data
        st.download_button(
            "Download All Grading History (CSV)",
            df_all.to_csv(index=False).encode("utf-8"),
            "grading_all_users.csv",
            "text/csv",
            help="Download all grading records from all users stored in the Google Sheet."
        )
    except Exception as e:
        st.error(f"Error loading all grading history from Google Sheet: {e}. Please ensure the sheet is accessible and correctly formatted.")
