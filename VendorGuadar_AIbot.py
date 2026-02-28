import streamlit as st
import re
import tempfile
import os
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================================
#  PROFESSIONAL UI (Custom CSS)
# ==========================================================
def apply_ui_design():
    st.markdown("""
        <style>
        .main { background: #0e1117; }
        .runway-box {
            /* Neutralized: Changed from Red to a subtle Slate Grey */
            background: rgba(65, 68, 76, 0.1); 
            border: 1px solid #41444C; 
            padding: 20px; border-radius: 15px;
            text-align: center; margin-bottom: 25px;
        }
        /* Neutralized: Death Date is now White/Silver instead of Red */
        .death-date { 
            color: #E0E0E0; 
            font-size: 28px; 
            font-weight: bold; 
            font-family: 'Courier New'; 
        }
        div[data-testid="stMetricValue"] { color: #00FFA3 !important; font-family: 'Courier New', monospace; }
        .stButton>button {
            border-radius: 25px;
            background: linear-gradient(45deg, #00FFA3, #00D1FF);
            color: black; font-weight: bold; border: none; transition: 0.3s;
        }
        .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 25px rgba(0, 255, 163, 0.5); }
        </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 🔐 CONFIG & SESSION
# ==========================================================
GROQ_API_KEY = "GROQ_API_KEY"
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"

# --- n8n INTEGRATION CONFIG ---
# Ensure your n8n Header Auth value is "SUSH_VG_2026_SECURE"
WEBHOOK_SECRET = "SUSH_VG_2026_SECURE" 
N8N_WEBHOOK_URL = "https://sush-1610.app.n8n.cloud/webhook-test/vendorguard-analysis"

if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "analysis" not in st.session_state: st.session_state.analysis = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

apply_ui_design()

# ==========================================================
# 🛠️ HELPER FUNCTIONS
# ==========================================================
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def send_to_n8n(payload):
    """Sends the forensic analysis data to your n8n workflow"""
    headers = {"X-VendorGuard-Token": WEBHOOK_SECRET}
    try:
        # We use a timeout to ensure the app doesn't hang if n8n is offline
        response = requests.post(N8N_WEBHOOK_URL, json=payload, headers=headers, timeout=10)
        return response.status_code
    except Exception as e:
        return str(e)

def build_vector_store(files):
    all_docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            all_docs.extend(loader.load())
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return FAISS.from_documents(splitter.split_documents(all_docs), load_embedder())

def safe_extract(label, text, default="50"):
    match = re.search(rf"{label}\s*[:\-]*\s*(\d+)", text, re.IGNORECASE)
    return match.group(1) if match else default

def safe_section(title, text):
    pattern = rf"{title}:(.*?)(?=\n[A-Z][a-z\s]+:|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else "Data points not identified."

# ==========================================================
# 🏠 SIDEBAR: THE RUNWAY CALCULATOR
# ==========================================================
with st.sidebar:
    st.header("🔬 Live Simulation")
    sim_burn = st.slider("Monthly Burn ($)", 5000, 500000, 80000)
    sim_runway = st.slider("Current Runway (Months)", 1, 36, 6)
    
    death_date = datetime.now() + timedelta(days=sim_runway * 30)
    
    st.markdown(f"""
        <div class="runway-box">
            <p style="margin-bottom:0px; font-size:14px; opacity:0.8;">ESTIMATED DEATH DATE</p>
            <p class="death-date">{death_date.strftime('%b %d, %Y')}</p>
            <p style="font-size:12px; opacity:0.6;">Based on ${sim_burn:,}/mo burn</p>
        </div>
    """, unsafe_allow_html=True)
    
    if sim_runway < 6:
        st.error("🚨 CRITICAL: Immediate funding required.")
    elif sim_runway < 12:
        st.warning("⚠️ WARNING: High risk. Low capital cushion.")
    else:
        st.success("✅ HEALTHY: Sufficient runway for scaling.")

# ==========================================================
# 🏠 MAIN DASHBOARD
# ==========================================================
st.title("🚀 VendorGuard AI")
st.caption("Forensic Due Diligence • Predictive Financial Auditing • VC Intelligence")

files = st.file_uploader("Upload Startup Deck & Financials (PDF)", type=["pdf"], accept_multiple_files=True)

if files and st.button("🚀 EXECUTE FORENSIC ANALYSIS"):
    with st.spinner("⚡ Mining data & calculating risk coefficients..."):
        st.session_state.vector_store = build_vector_store(files)
        
        docs = st.session_state.vector_store.similarity_search("revenue growth, burn rate, marketing CAC, team risks", k=6)
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"""
        Act as a Senior VC Partner. Analyze this startup based on:
        Docs: {context}
        Current Burn: ${sim_burn}/mo | Runway: {sim_runway}mo.

        RETURN FORMAT (STRICT):
        Overall Risk Score: X/100
        Survival Prob: X%
        Financial Risk: X/100
        Market Risk: X/100
        Execution Risk: X/100
        Verdict: [INVEST / PASS / WATCHLIST]
        
        INVESTMENT THESIS: [Write logic]
        CRITICAL RED FLAGS: [List 3 specific risks]
        QUANT ANALYSIS: [Detailed financial reasoning]
        """
        st.session_state.analysis = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.1
        ).choices[0].message.content

        # --- n8n TRIGGER LOGIC ---
        verdict_match = re.search(r"Verdict:\s*(.*)", st.session_state.analysis, re.IGNORECASE)
        current_verdict = verdict_match.group(1).strip() if verdict_match else "WATCHLIST"
        
        # Preparing the payload to send to your n8n workflow
        payload = {
            "startup_name": files[0].name,
            "overall_risk_score": safe_extract('Overall Risk Score', st.session_state.analysis),
            "survival_probability": safe_extract('Survival Prob', st.session_state.analysis),
            "financial_risk": safe_extract('Financial Risk', st.session_state.analysis),
            "market_risk": safe_extract('Market Risk', st.session_state.analysis),
            "execution_risk": safe_extract('Execution Risk', st.session_state.analysis),
            "verdict": current_verdict,
            "burn_rate": sim_burn,
            "runway_months": sim_runway,
            "timestamp": datetime.now().isoformat()
        }
        
        # Attempt to trigger n8n
        status = send_to_n8n(payload)
        if status == 200:
            st.toast("✅ Analysis synced to n8n successfully!")
        else:
            st.error(f"⚠️ n8n Sync Failed. Status Code: {status}. Make sure n8n is 'Listening'!")

# ==========================================================
# 📊 THE VISUAL RESULTS
# ==========================================================
if st.session_state.analysis:
    res = st.session_state.analysis
    
    # Top Level Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Score", f"{safe_extract('Overall Risk Score', res)}/100")
    c2.metric("Survival Prob.", f"{safe_extract('Survival Prob', res)}%")
    
    verdict_match = re.search(r"Verdict:\s*(.*)", res, re.IGNORECASE)
    v_text = verdict_match.group(1) if verdict_match else "WATCHLIST"
    c3.subheader(f"Verdict: {v_text}")

    # Visual Analytics (Radar Chart)
    col_left, col_right = st.columns([2, 3])
    
    with col_left:
        f = int(safe_extract("Financial Risk", res, "50"))
        m = int(safe_extract("Market Risk", res, "50"))
        e = int(safe_extract("Execution Risk", res, "50"))
        
        fig = go.Figure(data=go.Scatterpolar(r=[f,m,e], theta=['Financial','Market','Execution'], fill='toself', line_color='#00FFA3'))
        fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=300, margin=dict(l=40, r=40, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🎯 Investment Thesis")
        st.info(safe_section("INVESTMENT THESIS", res))

    # Detailed Tabs
    t1, t2 = st.tabs(["🔍 Quant Reasoning", "🚩 Red Flags"])
    with t1: st.write(safe_section("QUANT ANALYSIS", res))
    with t2: st.error(safe_section("CRITICAL RED FLAGS", res))

# ==========================================================
# 💬 INTERACTIVE AUDITOR
# ==========================================================
st.markdown("---")
st.subheader("💬 Ask the Auditor")

if st.session_state.vector_store:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role): st.write(msg)

    if user_input := st.chat_input("Ex: 'Tell me more about their market risk...'"):
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"): st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Checking logs..."):
                rel_docs = st.session_state.vector_store.similarity_search(user_input, k=4)
                chat_context = "\n\n".join([d.page_content for d in rel_docs])
                chat_prompt = f"Docs: {chat_context}\nFull Analysis: {st.session_state.analysis}\nQ: {user_input}"
                ans = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": chat_prompt}]).choices[0].message.content
                st.write(ans)
                st.session_state.chat_history.append(("assistant", ans))
else:
    st.info("💡 Ingest documents above to unlock the Interactive Auditor.")