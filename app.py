import streamlit as st
import pandas as pd
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# -------- Drive (OAuth app-owned) --------
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials as UserCredentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request

# -------- Auth0 login --------
from streamlit_auth0 import login_button

# -------- LLM grading --------
from langchain.chat_models import init_chat_model
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate

# -------- Vision (Gemini only) --------
import google.generativeai as genai

# -------- Misc --------
import os

# =============================================================================
# Auth0 Login
# =============================================================================
client_id = st.secrets["AUTH0_CLIENT_ID"]
domain = st.secrets["AUTH0_DOMAIN"]
user_info = login_button(client_id=client_id, domain=domain)

if user_info:
    with st.sidebar:
        st.markdown("**Signed in as:**")
        st.markdown(f"**Name:** {user_info['name']}")
        st.markdown(f"**Email:** {user_info['email']}")
        if st.button("Sign Out"):
            st.session_state.clear()
            st.rerun()
    st.title("Assignment Grader")
    st.success(f"Welcome, {user_info['name']}! Please provide assignment details below.")
else:
    st.warning("Please log in with Google to continue using the Assignment Grader.")
    st.stop()

# =============================================================================
# Google Sheets via Service Account (unchanged)
# =============================================================================
creds_dict = st.secrets["gcp_service_account"]
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_sa = ServiceAccountCredentials.from_json_keyfile_dict(dict(creds_dict), scope)

try:
    client = gspread.authorize(creds_sa)
    sheet = client.open("autograde_logs").sheet1
except Exception as e:
    st.error(f"Error connecting to Google Sheets: {e}. Ensure the service account can access 'autograde_logs'.")
    st.stop()

expected_headers = ["User", "DateTime", "Question", "StudentAnswer",
                    "Evaluation", "Feedback", "DetailedFeedback",
                    "GeneratedInstruction", "ImageLinks"]
current_headers = sheet.row_values(1)
if not current_headers:
    sheet.insert_row(expected_headers, 1)
else:
    changed = False
    for h in expected_headers:
        if h not in current_headers:
            current_headers.append(h); changed = True
    if changed:
        sheet.update('1:1', [current_headers])

# =============================================================================
# Google Drive via APP-OWNED OAuth (single common folder)
# =============================================================================
OAUTH_CLIENT_ID = st.secrets.get("GOOGLE_OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = st.secrets.get("GOOGLE_OAUTH_CLIENT_SECRET")
OAUTH_REDIRECT_URI = st.secrets.get("GOOGLE_OAUTH_REDIRECT_URI")
DRIVE_UPLOAD_FOLDER_ID = st.secrets.get("DRIVE_UPLOAD_FOLDER_ID", "")
APP_REFRESH_TOKEN = st.secrets.get("GOOGLE_OAUTH_REFRESH_TOKEN", "")

if not (OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET and OAUTH_REDIRECT_URI):
    st.error("Missing Google OAuth client settings in secrets (GOOGLE_OAUTH_CLIENT_ID/SECRET/REDIRECT_URI).")
    st.stop()

if not DRIVE_UPLOAD_FOLDER_ID:
    st.error("Missing DRIVE_UPLOAD_FOLDER_ID in secrets. Create a folder in *your* My Drive and paste its ID.")
    st.stop()

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

def _build_flow(state: str | None = None) -> Flow:
    client_config = {
        "web": {
            "client_id": OAUTH_CLIENT_ID,
            "client_secret": OAUTH_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [OAUTH_REDIRECT_URI],
        }
    }
    f = Flow.from_client_config(client_config, scopes=DRIVE_SCOPES, state=state)
    f.redirect_uri = OAUTH_REDIRECT_URI
    return f

def get_app_drive_service():
    """
    Use a stored refresh token (app-owned) to upload into a single common folder.
    No end-user prompts. If refresh token is missing, guide admin to connect once.
    """
    if APP_REFRESH_TOKEN:
        creds = UserCredentials(
            token=None,
            refresh_token=APP_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=OAUTH_CLIENT_ID,
            client_secret=OAUTH_CLIENT_SECRET,
            scopes=DRIVE_SCOPES,
        )
        try:
            creds.refresh(Request())
        except Exception as e:
            st.error(f"Failed to refresh stored Google OAuth token: {e}")
            return None
        return build("drive", "v3", credentials=creds)

    # No refresh token yet → admin one-time connect flow
    try:
        params = st.query_params
    except Exception:
        params = st.experimental_get_query_params()

    code = params.get("code", None)
    if isinstance(code, list): code = code[0]
    state = params.get("state", None)
    if isinstance(state, list): state = state[0]

    if code:
        flow = _build_flow(state=state)
        try:
            flow.fetch_token(code=code)
            creds = flow.credentials
            refresh_token = creds.refresh_token
            st.success("One-time Google OAuth completed.")
            st.warning("Copy this REFRESH TOKEN into your secrets as GOOGLE_OAUTH_REFRESH_TOKEN:")
            st.code(refresh_token or "(no refresh token returned — ensure access_type=offline, prompt=consent)")
            try:
                st.query_params.clear()
            except Exception:
                st.experimental_set_query_params()
            return build("drive", "v3", credentials=creds)
        except Exception as e:
            st.error(f"OAuth token exchange failed: {e}")
            return None

    flow = _build_flow()
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    st.info("Admin: connect Google Drive ONCE to mint a refresh token for the app (end users won't see this).")
    st.link_button("Admin: Connect Google Drive (one-time)", auth_url)
    return None

def _infer_mime_from_name(name: str) -> str:
    n = name.lower()
    if n.endswith(".png"): return "image/png"
    if n.endswith(".webp"): return "image/webp"
    if n.endswith(".jpg") or n.endswith(".jpeg"): return "image/jpeg"
    return "application/octet-stream"

def upload_image_to_drive_common(file, folder_id: str, drive_service, make_public: bool = True) -> str:
    """Upload to a fixed folder in app owner's My Drive and return a link."""
    if not drive_service:
        return ""
    try:
        file.seek(0)
        mime = _infer_mime_from_name(file.name)
        media = MediaIoBaseUpload(file, mimetype=mime, resumable=False)
        metadata = {"name": file.name, "parents": [folder_id], "mimeType": mime}
        created = drive_service.files().create(
            body=metadata,
            media_body=media,
            fields="id, webViewLink",
        ).execute()
        file_id = created["id"]
        if make_public:
            try:
                drive_service.permissions().create(
                    fileId=file_id,
                    body={"role": "reader", "type": "anyone"},
                    fields="id",
                ).execute()
            except Exception as pe:
                st.warning(f"Making link public failed (policy may block it): {pe}")
        return created.get("webViewLink", "")
    except HttpError as e:
        st.warning(f"Drive upload failed for {file.name}: {e}")
        return ""
    except Exception as e:
        st.warning(f"Drive upload failed for {file.name}: {e}")
        return ""

# =============================================================================
# Models (Groq LLM + Gemini Vision only)
# =============================================================================
if st.secrets.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("GROQ_API_KEY missing in secrets.")
    st.stop()

grade_model = init_chat_model("llama-3.1-8b-instant", model_provider="groq")
refine_model = init_chat_model("llama-3.1-8b-instant", model_provider="groq")

GEMINI_AVAILABLE = False
if st.secrets.get("GEMINI_API_KEY"):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        GEMINI_AVAILABLE = True
    except Exception as e:
        gemini_model = None
        st.warning(f"Gemini not initialized: {e}")
else:
    gemini_model = None

# =============================================================================
# Prompts (unchanged)
# =============================================================================
base_prompt_template = """(omitted for brevity; unchanged rubric)"""
refine_prompt_template = """(omitted for brevity; unchanged refine prompt)"""

response_schemas = [
    ResponseSchema(name="content_accuracy", description="Score out of 3 as a number"),
    ResponseSchema(name="completeness", description="Score out of 2 as a number"),
    ResponseSchema(name="language_clarity", description="Score out of 2 as a number"),
    ResponseSchema(name="depth_understanding", description="Score out of 2 as a number"),
    ResponseSchema(name="structure_coherence", description="Score out of 1 as a number"),
    ResponseSchema(name="justification", description="Detailed explanation"),
]
json_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
json_format_instructions = json_output_parser.get_format_instructions()
json_prompt = ChatPromptTemplate.from_template(
    """
You are an expert assignment grader.

Evaluate the StudentAnswer compared to the IdealAnswer using the rubric below and the selected grading style.

Rubric:
- Content Accuracy: 0–3
- Completeness: 0–2
- Language & Clarity: 0–2
- Depth of Understanding: 0–2
- Structure & Coherence: 0–1

Grading Style: {grading_style}

Question:
{question}

IdealAnswer:
{ideal_answer}

StudentAnswer:
{student_answer}

Return ONLY a JSON object following this schema:
{format_instructions}
""".strip()
)

# =============================================================================
# Session state
# =============================================================================
if "history" not in st.session_state: st.session_state.history = []
if "last_eval" not in st.session_state: st.session_state.last_eval = None
if "just_graded" not in st.session_state: st.session_state.just_graded = False
if "current_adaptive_instruction" not in st.session_state: st.session_state.current_adaptive_instruction = ""
if "last_image_notes" not in st.session_state: st.session_state.last_image_notes = []
if "last_uploaded_links" not in st.session_state: st.session_state.last_uploaded_links = []

# =============================================================================
# Gemini-only helpers
# =============================================================================
def gemini_analyze_image(file, question: str) -> str:
    if not GEMINI_AVAILABLE: return ""
    try:
        file.seek(0)
        bytes_data = file.read()
        mime = _infer_mime_from_name(file.name)
        prompt = (
            "You are assisting an assignment grader. Given the teacher's question below, "
            "extract ONLY content from this image useful for grading the student's answer. "
            "Prefer bullet points, include equations if visible, and name any diagrams.\n\n"
            f"Question:\n{question}"
        )
        resp = gemini_model.generate_content([prompt, {"mime_type": mime, "data": bytes_data}])
        return (resp.text or "").strip()
    except Exception as e:
        st.warning(f"Vision analysis failed for {file.name}: {e}")
        return ""

def process_images_to_context(question: str, images):
    if not images: return "", []
    per_image_notes, all_chunks = [], []
    for f in images:
        txt = gemini_analyze_image(f, question)
        per_image_notes.append({"file": f.name, "vision_excerpt": (txt[:800] + "…") if txt and len(txt) > 800 else txt})
        if txt:
            all_chunks.append(f"[VISION:{f.name}]\n{txt}")
        f.seek(0)
    return "\n\n---\n".join(all_chunks).strip(), per_image_notes

# =============================================================================
# UI Form
# =============================================================================
with st.form("grading_form"):
    st.subheader("Assignment Details")
    question = st.text_area("Enter the question:", height=100)
    ideal_answer = st.text_area("Enter the ideal answer:", height=150)

    st.subheader("Student Answer")
    student_answer_text = st.text_area("Type the student's answer (optional if uploading image):", height=150)
    student_answer_images = st.file_uploader("Or upload student's answer image(s):", type=["jpg","jpeg","png","webp"], accept_multiple_files=True)

    if student_answer_images:
        with st.expander("Preview uploaded student answer images"):
            cols = st.columns(min(3, len(student_answer_images)))
            for i, f in enumerate(student_answer_images):
                with cols[i % len(cols)]:
                    st.image(f, caption=f.name, use_container_width=True)
                    f.seek(0)

    grading_style = st.selectbox("Select grading style:", ["Balanced", "Strict", "Lenient"])
    submit_button = st.form_submit_button("Grade Answer")

# =============================================================================
# Main grading + Upload to common folder
# =============================================================================
if submit_button:
    if not (question and ideal_answer and (student_answer_text or student_answer_images)):
        st.warning("Please ensure the Question and Ideal Answer are filled, and provide either typed Student Answer or upload image(s).")
        st.stop()

    if student_answer_images and not GEMINI_AVAILABLE:
        st.error("Image processing requires Gemini. Add GEMINI_API_KEY to secrets.")
        st.stop()

    drive_service = get_app_drive_service()
    if student_answer_images and not drive_service:
        st.stop()  # waiting for admin to mint refresh token

    uploaded_links = []
    if student_answer_images and drive_service:
        for f in student_answer_images:
            link = upload_image_to_drive_common(f, DRIVE_UPLOAD_FOLDER_ID, drive_service, make_public=True)
            if link: uploaded_links.append(link)

    if uploaded_links:
        st.subheader("Uploaded Image Links")
        for url in uploaded_links:
            st.markdown(f"- [{url}]({url})")

    image_context_text, per_image_notes = process_images_to_context(question, student_answer_images) if student_answer_images else ("", [])
    student_answer_augmented = (student_answer_text or "").strip()
    if image_context_text:
        student_answer_augmented += "\n\n---\n[IMAGE-DERIVED CONTEXT FOR GRADING]\n" + image_context_text

    student_answer_for_log = image_context_text if image_context_text else (student_answer_text or "").strip()
    st.session_state.last_image_notes = per_image_notes
    st.session_state.last_uploaded_links = uploaded_links

    # Run grading
    try:
        chain = json_prompt | grade_model | json_output_parser
        parsed = chain.invoke({
            "grading_style": grading_style,
            "question": question.strip(),
            "ideal_answer": ideal_answer.strip(),
            "student_answer": student_answer_augmented,
            "format_instructions": json_format_instructions,
        })
        ca = float(parsed.get("content_accuracy", 0))
        co = float(parsed.get("completeness", 0))
        lc = float(parsed.get("language_clarity", 0))
        du = float(parsed.get("depth_understanding", 0))
        sc = float(parsed.get("structure_coherence", 0))
        total = ca + co + lc + du + sc
        just = (parsed.get("justification") or "").strip()

        evaluation = f"""## Marks:
- Content Accuracy: {ca}/3
- Completeness: {co}/2
- Language & Clarity: {lc}/2
- Depth of Understanding: {du}/2
- Structure & Coherence: {sc}/1
- **Total: {total}/10**

## Justification:
{just}
"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.session_state.history.append({
            "user": user_info["email"],
            "timestamp": timestamp,
            "question": question.strip(),
            "ideal_answer": ideal_answer.strip(),
            "student_answer": student_answer_for_log,
            "evaluation": evaluation.strip(),
            "feedback": "",
            "detailed_feedback": "",
            "generated_instruction": "",
            "image_links": "; ".join(uploaded_links),
        })
        st.session_state.last_eval = {
            "email": user_info["email"],
            "timestamp": timestamp,
            "question": question.strip(),
            "ideal_answer": ideal_answer.strip(),
            "student_answer": student_answer_for_log,
            "evaluation": evaluation.strip(),
        }
        st.session_state.just_graded = True
        st.success("Grading completed successfully!")
    except Exception as e:
        st.error(f"Error during AI grading: {e}")
        st.session_state.just_graded = False

# =============================================================================
# Results + Feedback (unchanged, plus link display)
# =============================================================================
if st.session_state.get("just_graded", False) and st.session_state.last_eval:
    evaluation = st.session_state.last_eval["evaluation"]

    st.subheader("Evaluation Result")
    if "## Justification:" in evaluation:
        parts = evaluation.split("## Justification:", 1)
        marks_section_raw = parts[0].strip()
        explanation_section = parts[1].strip()
        marks_section = marks_section_raw[len("## Marks:"):].strip() if marks_section_raw.startswith("## Marks:") else marks_section_raw
        st.markdown("---"); st.subheader("Marks Breakdown"); st.code(marks_section, language="markdown")
        st.subheader("Justification"); st.markdown(explanation_section); st.markdown("---")
    else:
        st.warning("Unexpected evaluation format. Displaying full response:"); st.markdown(evaluation); st.markdown("---")

    if st.session_state.last_image_notes:
        with st.expander("Image-derived context used for grading"):
            for note in st.session_state.last_image_notes:
                st.markdown(f"**{note['file']}**")
                if note.get("vision_excerpt"):
                    st.code(note["vision_excerpt"], language="markdown")

    if st.session_state.last_uploaded_links:
        with st.expander("Uploaded image links for this submission"):
            for url in st.session_state.last_uploaded_links:
                st.markdown(f"- [{url}]({url})")

    with st.form("feedback_form"):
        st.subheader("Provide Feedback on this Evaluation")
        satisfaction = st.radio("Are you satisfied with this grading?", ["Yes", "No"], key="satisfaction_radio")
        detailed_feedback_text = st.text_area("If 'No', explain what was wrong.", key="detailed_feedback_text_area", height=100) if satisfaction == "No" else ""
        submit_feedback_button = st.form_submit_button("Submit Feedback")

    if submit_feedback_button:
        if satisfaction == "No" and not detailed_feedback_text.strip():
            st.warning("Please provide details for 'No' feedback.")
            st.session_state.just_graded = True
            st.stop()

        st.session_state.history[-1]["feedback"] = satisfaction
        st.session_state.history[-1]["detailed_feedback"] = detailed_feedback_text.strip()

        generated_instruction = ""
        if satisfaction == "No" and detailed_feedback_text.strip():
            refine_prompt = refine_prompt_template.format(
                question=st.session_state.last_eval["question"],
                ideal_answer=st.session_state.last_eval["ideal_answer"],
                student_answer=st.session_state.last_eval["student_answer"],
                ai_evaluation=st.session_state.last_eval["evaluation"],
                detailed_feedback=detailed_feedback_text.strip()
            )
            try:
                refine_response = refine_model.invoke(refine_prompt)
                generated_instruction = (refine_response.content or "").strip()
                if generated_instruction and generated_instruction != "NO_IMPROVEMENT_NEEDED":
                    st.session_state.current_adaptive_instruction = generated_instruction
                    st.info(f"**New Adaptive Instruction Generated:** '{generated_instruction}'")
                else:
                    st.info("No specific instruction generated for improvement.")
                    st.session_state.current_adaptive_instruction = ""
            except Exception as e:
                st.error(f"Error during prompt refinement: {e}")
                st.session_state.current_adaptive_instruction = ""
        else:
            st.info("Feedback submitted. Thanks!")

        st.session_state.history[-1]["generated_instruction"] = generated_instruction

        cleaned_eval = st.session_state.last_eval["evaluation"].replace("\n", " ⏎ ")
        image_links_cell = "; ".join(st.session_state.last_uploaded_links) if st.session_state.last_uploaded_links else ""
        row_to_append = [
            st.session_state.last_eval["email"],
            st.session_state.last_eval["timestamp"],
            st.session_state.last_eval["question"],
            st.session_state.last_eval["student_answer"],
            cleaned_eval,
            satisfaction,
            st.session_state.history[-1]["detailed_feedback"],
            generated_instruction,
            image_links_cell
        ]
        try:
            sheet.append_row(row_to_append)
            st.success("Feedback recorded and grading log updated in Google Sheets.")
        except Exception as e:
            st.error(f"Error appending data to Google Sheet: {e}")

        st.session_state.just_graded = False
        st.session_state.last_uploaded_links = []

# =============================================================================
# History Viewer + Downloads
# =============================================================================
if st.session_state.history:
    st.markdown("---")
    with st.expander("View Your Current Session Evaluations"):
        for i, entry in enumerate(st.session_state.history[::-1], 1):
            st.markdown(f"### Evaluation #{len(st.session_state.history)-i+1}")
            st.markdown(f"**User:** {entry['user']}")
            st.markdown(f"**Time:** {entry['timestamp']}")
            st.markdown(f"**Question:** {entry['question']}")
            st.markdown(f"**Student Answer:** {entry['student_answer']}")
            st.markdown("**AI Evaluation:**")
            st.markdown(entry["evaluation"])
            if entry.get("image_links"):
                st.markdown("**Image Links:**")
                for url in entry["image_links"].split("; "):
                    if url.strip(): st.markdown(f"- [{url}]({url})")
            if entry["feedback"]: st.markdown(f"**User Feedback:** {entry['feedback']}")
            if entry["detailed_feedback"]: st.markdown(f"**Detailed Feedback:** {entry['detailed_feedback']}")
            if entry["generated_instruction"]: st.markdown(f"**Generated Instruction:** {entry['generated_instruction']}")
            st.markdown("---")

    df = pd.DataFrame(st.session_state.history)
    st.download_button("Download Current Session Data (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "grading_session.csv", "text/csv")

st.markdown("---")
if st.checkbox("Show All Grading History (from Google Sheet)"):
    try:
        with st.spinner("Loading all grading history from Google Sheet..."):
            records = sheet.get_all_records()
            df_all = pd.DataFrame(records)
            st.dataframe(df_all)
        st.download_button("Download All Grading History (CSV)",
                           df_all.to_csv(index=False).encode("utf-8"),
                           "grading_all_users.csv", "text/csv")
    except Exception as e:
        st.error(f"Error loading all grading history from Google Sheet: {e}")
