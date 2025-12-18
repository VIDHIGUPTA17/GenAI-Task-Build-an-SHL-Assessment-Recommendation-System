
import streamlit as st
import requests
import re
from typing import Dict, Any

# ===============================
# CONFIG
# ===============================
API_BASE_URL = "http://localhost:8000"   # Change if deployed
RECOMMEND_ENDPOINT = f"{API_BASE_URL}/recommend"

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide"
)

# ===============================
# HELPERS
# ===============================
def is_url(text: str) -> bool:
    return bool(re.match(r"https?://", text.strip()))

def fetch_jd_from_url(url: str) -> str:
    """Fetch job description text from URL"""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text[:5000]  # limit size
    except Exception as e:
        st.error(f"Failed to fetch JD from URL: {e}")
        return ""

def call_recommend_api(query: str, k: int) -> Dict[str, Any]:
    payload = {
        "query": query,
        "k": k
    }
    response = requests.post(RECOMMEND_ENDPOINT, params=payload, timeout=30)
    response.raise_for_status()
    return response.json()

# ===============================
# UI HEADER
# ===============================
st.title("🎯 SHL Assessment Recommendation System")

st.markdown(
    """
Hiring managers and recruiters often struggle to identify the **right assessments**
for open roles.  
This tool uses **AI-powered semantic search** to recommend the most relevant SHL
assessments from a **natural language query, job description, or JD URL**.
"""
)

# ===============================
# INPUT SECTION
# ===============================
st.subheader("🔍 Enter Job Requirement")

input_text = st.text_area(
    label="Job Description / Natural Language Query / JD URL",
    placeholder=(
        "Examples:\n"
        "- Java developer with strong communication skills, max 40 minutes\n"
        "- We are hiring a data analyst with SQL and Python experience\n"
        "- https://company.com/jobs/backend-engineer"
    ),
    height=180
)

col1, col2 = st.columns([1, 1])
with col1:
    k = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

with col2:
    submit = st.button("🚀 Get Recommendations")

# ===============================
# PROCESS
# ===============================
if submit:
    if not input_text.strip():
        st.warning("Please enter a query, job description, or URL.")
    else:
        with st.spinner("Analyzing requirement and fetching assessments..."):
            final_query = input_text.strip()

            # Handle URL input
            if is_url(final_query):
                jd_text = fetch_jd_from_url(final_query)
                if jd_text:
                    final_query = f"Job Description:\n{jd_text}"

            try:
                response = call_recommend_api(final_query, k)
                results = response.get("recommended_assessments", [])

                st.success(f"✅ Found {len(results)} relevant assessments")

                # ===============================
                # RESULTS
                # ===============================
                for idx, assessment in enumerate(results, start=1):
                    with st.expander(f"📌 Recommendation #{idx}", expanded=True):
                        st.markdown(f"**🔗 URL:** {assessment['url']}")
                        st.markdown(f"**📝 Description:** {assessment['description']}")
                        st.markdown(f"**⏱ Duration:** {assessment['duration']} minutes")
                        st.markdown(f"**🧠 Adaptive Support:** {assessment['adaptive_support']}")
                        st.markdown(f"**🌐 Remote Support:** {assessment['remote_support']}")
                        st.markdown(
                            f"**📊 Test Type:** {', '.join(assessment['test_type'])}"
                        )

            except requests.exceptions.RequestException as e:
                st.error(f"API Error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Powered by FastAPI • FAISS • SentenceTransformers • Streamlit")
