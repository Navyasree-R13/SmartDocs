
import streamlit as st
import boto3
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="SmartDocs | AI Document Manager",
    page_icon="📄",
    layout="wide"
)

BUCKET_NAME = "smartdocs-storage-2026"
MODEL_NAME = "sshleifer/distilbart-cnn-6-6"

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>

.stApp {
    background-color: #F5F7FC;
}

section[data-testid="stSidebar"] {
    background-color: #101A35;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #172554;
}

.subtitle {
    color: #64748B;
    font-size: 16px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    margin-bottom: 10px;
}

.metric-label {
    color: #64748B;
    font-size: 14px;
}

.metric-value {
    color: #172554;
    font-size: 28px;
    font-weight: 750;
}

.section-heading {
    color: #172554;
    font-size: 23px;
    font-weight: 700;
}

div.stButton > button {
    background-color: #5145CD;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 18px;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #3730A3;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SESSION DATA ----------------

if "documents" not in st.session_state:
    st.session_state.documents = []

if "summary" not in st.session_state:
    st.session_state.summary = None

if "summary_filename" not in st.session_state:
    st.session_state.summary_filename = None


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("# 📄 SmartDocs")
    st.caption("AI DOCUMENT MANAGEMENT")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "My Documents", "AI Summarizer", "About"]
    )

    st.divider()

    st.markdown("### ☁️ Cloud Storage")
    st.caption("Powered by Amazon S3")

    st.markdown("### 🤖 AI Engine")
    st.caption("Powered by Hugging Face")


# ---------------- AWS CONNECTION ----------------
def get_s3_client():
    import os
    import boto3

    try:
        aws_secrets = st.secrets["aws"]

        return boto3.client(
            "s3",
            aws_access_key_id=aws_secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=aws_secrets["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=aws_secrets["AWS_SESSION_TOKEN"],
            region_name=aws_secrets.get("AWS_DEFAULT_REGION", "us-east-1")
        )

    except Exception:
        return boto3.client(
            "s3",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )


# ---------------- AI MODEL ----------------

@st.cache_resource
def load_ai_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    return tokenizer, model


# ---------------- PDF EXTRACTION ----------------

def extract_pdf_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ---------------- AI SUMMARIZATION ----------------

def generate_summary(text, tokenizer, model):

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunk_size = 800
    summaries = []

    total_chunks = (len(tokens) + chunk_size - 1) // chunk_size

    progress = st.progress(0)

    for i in range(0, len(tokens), chunk_size):

        chunk = tokenizer.decode(tokens[i:i + chunk_size])

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        output = model.generate(
            inputs["input_ids"],
            max_length=120,
            min_length=30,
            num_beams=2,
            early_stopping=True,
            forced_bos_token_id=0
        )

        summaries.append(
            tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )
        )

        progress.progress(
            len(summaries) / total_chunks
        )

    progress.empty()

    return "\n\n".join(summaries)


# ==================================================
#                  DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Welcome to SmartDocs 👋</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Your intelligent workspace for managing documents with AI.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📁 Documents Processed</div>
            <div class="metric-value">{len(st.session_state.documents)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🤖 AI Summaries</div>
            <div class="metric-value">{st.session_state.get('summary_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">☁️ Storage</div>
            <div class="metric-value">AWS S3</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown(
        '<div class="section-heading">📤 Upload Your Document</div>',
        unsafe_allow_html=True
    )

    st.write("Upload a PDF and let AI understand your document.")

    uploaded_file = st.file_uploader(
        "Choose your PDF document",
        type=["pdf"],
        key="dashboard_upload"
    )

    if uploaded_file:

        st.success(f"Selected: {uploaded_file.name}")

        st.write(f"File size: {uploaded_file.size:,} bytes")

        col1, col2 = st.columns(2)

        with col1:

            if st.button("☁️ Save to AWS S3", use_container_width=True):

                try:

                    s3 = get_s3_client()

                    uploaded_file.seek(0)

                    s3.upload_fileobj(
                        uploaded_file,
                        BUCKET_NAME,
                        uploaded_file.name
                    )

                    st.success("Document successfully saved to AWS S3!")

                    if uploaded_file.name not in st.session_state.documents:
                        st.session_state.documents.append(uploaded_file.name)

                except Exception as e:

                    st.error(f"S3 upload failed: {e}")

        with col2:

            if st.button("🤖 Generate AI Summary", use_container_width=True):

                try:

                    with st.spinner("Extracting PDF text..."):

                        uploaded_file.seek(0)

                        text = extract_pdf_text(uploaded_file)

                    if not text.strip():

                        st.warning("No readable text found in this PDF.")

                    else:

                        with st.spinner("AI is analysing your document..."):

                            tokenizer, model = load_ai_model()

                            summary = generate_summary(
                                text,
                                tokenizer,
                                model
                            )

                        st.session_state.summary = summary
                        st.session_state.summary_filename = uploaded_file.name
                        st.session_state.summary_count = (
                            st.session_state.get("summary_count", 0) + 1
                        )

                        if uploaded_file.name not in st.session_state.documents:
                            st.session_state.documents.append(uploaded_file.name)

                        st.success("AI summary generated successfully!")

                except Exception as e:

                    st.error(f"Summarization failed: {e}")

    if st.session_state.summary:

        st.divider()

        st.markdown(
            '<div class="section-heading">✨ AI Generated Summary</div>',
            unsafe_allow_html=True
        )

        st.caption(f"Document: {st.session_state.summary_filename}")

        st.write(st.session_state.summary)

        st.download_button(
            "⬇️ Download Summary",
            data=st.session_state.summary,
            file_name="SmartDocs_Summary.txt",
            mime="text/plain"
        )


# ==================================================
#                 MY DOCUMENTS PAGE
# ==================================================

elif page == "My Documents":

    st.markdown(
        '<div class="main-title">My Documents 📁</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">View documents processed during this session.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if st.session_state.documents:

        for document in st.session_state.documents:

            st.markdown(f"📄 **{document}**")

            st.divider()

    else:

        st.info("No documents processed yet. Upload a PDF from Dashboard.")


# ==================================================
#                 AI SUMMARIZER PAGE
# ==================================================

elif page == "AI Summarizer":

    st.markdown(
        '<div class="main-title">AI Summarizer 🤖</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Transform lengthy documents into meaningful summaries.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if st.session_state.summary:

        st.subheader(st.session_state.summary_filename)

        st.write(st.session_state.summary)

        st.download_button(
            "⬇️ Download Summary",
            data=st.session_state.summary,
            file_name="SmartDocs_Summary.txt",
            mime="text/plain"
        )

    else:

        st.info("Generate a summary from the Dashboard to see it here.")


# ==================================================
#                    ABOUT PAGE
# ==================================================

elif page == "About":

    st.markdown(
        '<div class="main-title">About SmartDocs 📄</div>',
        unsafe_allow_html=True
    )

    st.write(
        "SmartDocs is an AI-powered cloud document management "
        "system designed to simplify PDF storage and understanding."
    )

    st.subheader("Technology Stack")

    st.markdown("""
    - **Streamlit:** Web application interface
    - **Python:** Application logic
    - **PyPDF:** PDF text extraction
    - **Hugging Face Transformers:** AI summarization
    - **PyTorch:** AI model execution
    - **Amazon S3:** Cloud document storage
    """)

    st.subheader("Key Features")

    st.markdown("""
    - PDF document upload
    - Cloud document storage
    - AI-powered PDF summarization
    - Summary download
    - Document management dashboard
    """)


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "© 2026 SmartDocs | AI-Powered Cloud Document Management System"
)
