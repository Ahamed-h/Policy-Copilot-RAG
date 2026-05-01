import streamlit as st
from rag import inference

st.set_page_config(
    page_title="Policy Copilot RAG",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}
.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}
.sub-title {
    font-size: 1.05rem;
    margin-bottom: 1.2rem;
    max-width: 900px;
}
.metric-card {
    border: 1px solid rgba(128, 128, 128, 0.2);
    padding: 1rem 1rem;
    border-radius: 14px;
    margin-bottom: 0.5rem;
}
.metric-title {
    font-size: 0.85rem;
    opacity: 0.7;
    margin-bottom: 0.2rem;
}
.metric-value {
    font-size: 1.05rem;
    font-weight: 600;
}
.small-note {
    font-size: 0.9rem;
    opacity: 0.7;
}
.badge {
    display: inline-block;
    padding: 0.3rem 0.7rem;
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 999px;
    font-size: 0.8rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

WELCOME_MESSAGE = (
    "Hello — I’m **Policy Copilot**. I can help answer questions grounded in your company "
    "policies, onboarding guides, and internal documentation."
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]

if "chat_count" not in st.session_state:
    st.session_state.chat_count = 0

def reset_conversation():
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.chat_count = 0

with st.sidebar:
    st.title("Policy Copilot")
    st.write("Internal knowledge assistant for policy and compliance queries.")
    st.divider()
    st.subheader("Suggested prompts")
    st.write("- What is the leave approval process?")
    st.write("- Are there probation-period leave restrictions?")
    st.write("- What are the onboarding steps for new employees?")
    st.write("- What documents are required for reimbursement?")
    st.divider()
    st.subheader("Assistant behavior")
    st.write("Answers are grounded in uploaded policy PDFs and should stay concise, relevant, and document-based.")
    st.divider()
    st.button("Reset Chat", use_container_width=True, on_click=reset_conversation)

st.markdown('<div class="badge">AI-powered internal policy assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Policy Copilot RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A retrieval-augmented assistant for answering employee policy, onboarding, and internal process questions with grounded document context.</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-title">Primary use case</div><div class="metric-value">Policy Q&A</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-title">Knowledge source</div><div class="metric-value">PDF documents</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Questions asked</div><div class="metric-value">{st.session_state.chat_count}</div></div>', unsafe_allow_html=True)

st.markdown("### Conversation")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about leave policy, onboarding, reimbursements, or internal guidelines...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_count += 1

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reviewing policy documents..."):
            stream = inference(prompt)
            full_response = st.write_stream(stream)

    final_response = f"""### Policy Response

    {full_response}

    ---

    **Response style:** grounded in uploaded policy documents, concise, and decision-oriented.
    """

    st.session_state.messages.append({"role": "assistant", "content": final_response})

st.markdown(
    '<div class="small-note">Tip: use specific questions for better retrieval, for example "What is the notice period during probation?"</div>',
    unsafe_allow_html=True
)