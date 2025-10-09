import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Azy — Home",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles
st.markdown(
    """
    <style>
    :root {
        --azy-primary: #0ea5e9;
        --azy-primary-dark: #0284c7;
        --azy-bg: #0b1220;
        --azy-card: #111827;
        --azy-text: #e5e7eb;
        --azy-muted: #9ca3af;
    }
    .azy-hero { background: radial-gradient(1200px 400px at 10% -10%, rgba(14,165,233,0.25), transparent),
                           radial-gradient(1200px 400px at 90% -10%, rgba(99,102,241,0.2), transparent);
                padding: 28px 28px 16px 28px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.06); }
    .azy-title { display:flex; align-items:center; gap:12px; font-weight:800; font-size: 28px; color: var(--azy-text); }
    .azy-sub { color: var(--azy-muted); margin-top: 6px; font-size: 15px; }
    .azy-section { background: var(--azy-card); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 16px; }
    .azy-h3 { color: var(--azy-text); font-weight: 800; font-size: 18px; margin: 0 0 10px 0; }
    .azy-p { color: var(--azy-text); font-size: 14px; margin: 6px 0; }
    .azy-li { color: var(--azy-text); font-size: 14px; margin: 4px 0; }
    .azy-avatar { width: 84px; height: 84px; border-radius: 50%; display:flex; align-items:center; justify-content:center; font-size: 42px; background: rgba(14,165,233,0.12); border:1px solid rgba(14,165,233,0.35); }
    .azy-bubble { background:white; color:#1f2937; border-radius: 12px; padding: 12px 14px; display:inline-block; position: relative; font-size: 14px; }
    .azy-bubble:after { content: ""; position:absolute; left: -8px; top: 18px; width:0; height:0; border-top:8px solid transparent; border-bottom:8px solid transparent; border-right:8px solid white; }
    .azy-muted { color: var(--azy-muted); font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero
st.markdown(
    """
    <div class="azy-hero">
        <div class="azy-title">🚨 <span>Meet Azy, Your Disaster Preparedness Chatbot</span></div>
        <div class="azy-sub">Using the power of AI to equip those who need help and those who would like to give help with access to essential information instantly.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("\n")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown(
        """
        <div class="azy-section">
            <div class="azy-h3">Get to Know Azy</div>
            <p class="azy-p">Azy is an exciting new channel that uses the power of artificial intelligence to guide website visitors to assistance, resources and information about disaster preparedness. Chat with her today!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("\n")
    st.markdown(
        """
        <div class="azy-section">
            <div class="azy-h3">Azy can answer your questions about:</div>
            <ul>
                <li class="azy-li">Disaster Preparedness, including correct information on preparedness methods</li>
                <li class="azy-li">Volunteering, including how to apply or volunteer remotely to become a member of the Tutzar community</li>
                <li class="azy-li">And more!</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("\n")
    st.markdown(
        """
        <div class="azy-section">
            <div class="azy-h3">What Else Azy Does?</div>
            <p class="azy-p"><b>Natural Language Processing</b>: a subfield of artificial intelligence that allows Azy to process large amounts of natural language data. This is what allows Azy to understand what users say!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="azy-section" style="display:flex; gap:14px; align-items:center;">
            <div class="azy-avatar">👩‍💻</div>
            <div>
                <div class="azy-bubble">Hello! How can I help you today?</div>
                <div class="azy-muted" style="margin-top:8px;">Meet the inspiration for the Azy ChatBot...</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("\n")
    st.markdown(
        """
        <div class="azy-section">
            <div class="azy-h3">Start chatting</div>
            <p class="azy-p">Open the “app.py” page from the sidebar to ask Azy a question. She will search expert-curated documents and answer with sources.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


