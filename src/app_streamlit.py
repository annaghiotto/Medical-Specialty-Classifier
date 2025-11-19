import streamlit as st
from infer import MedicalSpecialtyPredictor


@st.cache_resource
def load_predictor():
    return MedicalSpecialtyPredictor("artifacts")


predictor = load_predictor()

EXAMPLES = {
    "Cardiology / Pulmonology": "A 67-year-old male presents with chest pain radiating to the left arm and shortness of breath for two hours. ECG reveals ST-segment elevation in anterior leads and troponin is positive. Patient was started on aspirin, heparin, and taken for emergency coronary angiography showing LAD occlusion successfully stented.",
    "Neurology": "A 45-year-old woman presents with recurrent headaches and transient visual blurring over the past three months. Neurological exam shows mild papilledema. MRI of the brain demonstrates no mass lesion but mild increase in intracranial pressure consistent with idiopathic intracranial hypertension.",
    "Orthopedic": "A 34-year-old male reports twisting his right knee while playing soccer yesterday. On exam there is joint effusion, medial joint line tenderness, and positive McMurray test. MRI confirms a medial meniscus tear. Plan for arthroscopic partial meniscectomy discussed.",
}

# Page config
st.set_page_config(
    page_title="Medical Specialty Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* General styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        /*background: #f5f7fa 100%;*/
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white !important;
    }

    .main-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
        color: white !important;
    }

    /* Card container */
    .card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        transition: transform 0.2s;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
    }

    /* Prediction results */
    .prediction-item {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        font-weight: 500;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .prediction-number {
        background: rgba(255, 255, 255, 0.25);
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
        color: white;
    }

    .prediction-label {
        color: white;
    }

    /* History item */
    .history-item {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .history-item strong {
        color: #667eea;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Text area */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s;
    }

    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Slider */
    .stSlider {
        padding: 1rem 0;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 12px;
        font-weight: 600;
        color: #667eea;
    }

    /* Section headers */
    .section-header {
        background: white;
        border: 3px solid transparent;
        border-radius: 15px;

        background-image:
            linear-gradient(white, white),
            linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-origin: border-box;
        background-clip: padding-box, border-box;

        color: #667eea;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding: 0.8rem 1rem;

        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #9e9e9e;
        font-size: 1.1rem;
    }

    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }

    /* Label fixes */
    label {
        color: #333 !important;
    }

    .stMarkdown {
        color: #333;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# Main header
st.markdown(
    """
<div class="main-header">
    <h1>🩺 Medical Specialty Classifier</h1>
    <p>NLP System to predict the Medical Specialty of a clinical report using
            BioClinicalBERT as encoder and SVM as classifier.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Two-column layout
col_input, col_output = st.columns([1.5, 1], gap="large")

with col_input:
    st.markdown(
        '<div class="section-header">📝 Clinical Transcription</div>',
        unsafe_allow_html=True,
    )

    text = st.text_area(
        "Describe symptoms or clinical case:",
        height=250,
        placeholder="Enter a clinical transcription or use one of the examples provided below.\n"
        "E.g., 45-year-old patient with chest pain radiating to left arm, dyspnea and profuse sweating...",
        key="input_text",
        label_visibility="collapsed",
    )

    topk = st.slider(
        "Set the number of answers to show:",
        min_value=1,
        max_value=4,
        value=1,
        key="topk",
    )

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

    with btn_col3:
        predict_clicked = st.button("🚀 Analyze", use_container_width=True)

    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)

    # examples
    st.markdown("**Try an example:**")
    example_cols = st.columns(3)

    for example_name, example_text in EXAMPLES.items():
        st.markdown(f"- {example_text}\n\n  Expected output: {example_name}")

    if predict_clicked:
        if not text.strip():
            st.warning("⚠️ Please enter some text before analyzing.")
        else:
            with st.spinner("🔬 Analysis in progress..."):
                labels = predictor.predict(text, topk=topk)

            if isinstance(labels, str):
                labels = [labels]
            elif isinstance(labels, list) and labels and isinstance(labels[0], list):
                labels = labels[0]

            labels = [str(x) for x in labels]

            prediction_data = {
                "text": text,
                "topk": topk,
                "labels": labels,
            }
            st.session_state.last_prediction = prediction_data
            st.session_state.history.insert(0, prediction_data)

    st.markdown("</div>", unsafe_allow_html=True)

with col_output:
    st.markdown('<div class="section-header">🎯 Results</div>', unsafe_allow_html=True)

    last_pred = st.session_state.last_prediction

    if not last_pred or not last_pred.get("labels"):
        st.markdown(
            """
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <div>No prediction available yet</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">Enter text and click "Analyze"</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        labels = last_pred["labels"]
        for i, lab in enumerate(labels, start=1):
            st.markdown(
                f"""
            <div class="prediction-item">
                <div class="prediction-number">{i}</div>
                <div class="prediction-label">{lab}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # History section
    st.markdown('<div class="section-header">📊 History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            """
        <div class="empty-state" style="padding: 2rem 1rem;">
            <div class="empty-state-icon">📋</div>
            <div>No previous analysis</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        for idx, pred in enumerate(st.session_state.history, start=1):
            text_preview = pred["text"][:100] + (
                "..." if len(pred["text"]) > 100 else ""
            )

            st.markdown(
                f"""
                <div class="history-item">
                    <strong>Analysis #{idx}</strong><br>
                    <span style="color: #666; font-size: 0.9rem;">{text_preview}</span><br>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for i, lab in enumerate(pred["labels"], start=1):
                st.markdown(
                    f"<div style='margin-left: 1rem; color: #667eea;'>• {lab}</div>",
                    unsafe_allow_html=True,
                )

            if idx < len(st.session_state.history):
                st.markdown(
                    "<hr style='margin: 1rem 0; border: none; border-top: 1px solid #e0e0e0;'>",
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)
