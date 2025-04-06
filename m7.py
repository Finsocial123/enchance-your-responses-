import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(
    page_title="Finsocial Digital System - Intelligent Query Processing",
    layout="wide"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: #34495e;">Intelligent Query Processing with Enhanced Model Responses</h2>
        </div>
        """, 
        unsafe_allow_html=True
    )

st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h3>Finsocial Digital System</h3>
    </div>
    """, 
    unsafe_allow_html=True
)

st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Select Model",
    ["qwen", "openthinker", "nemotron-mini:latest"]
)

# Extend the configuration to include the additional model for Qwen reasoning.
MODEL_CONFIG = {
    "qwen": {"base_url": "https://unknown-veronique-finsocialdigitalsystem-cf02d63f.koyeb.app/v1", "api_key": "qwen-api-key"},
    "openthinker": {"base_url": "https://unknown-veronique-finsocialdigitalsystem-cf02d63f.koyeb.app/v1", "api_key": "openthinker-api-key"},
    "nemotron-mini:latest": {"base_url": "https://unknown-veronique-finsocialdigitalsystem-cf02d63f.koyeb.app/v1", "api_key": "nemotron-mini-api-key"},
    "llama3.2:1b": {"base_url": "https://unknown-veronique-finsocialdigitalsystem-cf02d63f.koyeb.app/v1", "api_key": "llama3.2-api-key"},
    "Qwen/QwQ-32B-AWQ": {"base_url": "https://screeching-arleen-finsocialdigitalsystem-1718d858.koyeb.app/v1", "api_key": "Qwen/QwQ-32B-AWQ-api-key"}
}

selected_config = MODEL_CONFIG[selected_model]
api_key = selected_config["api_key"]
base_url = selected_config["base_url"]

# Read template file without caching to avoid stale data.
def load_templates(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading template file: {e}")
        return []

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_agent(model_name):
    config = MODEL_CONFIG[model_name]
    return Agent(model=OpenAILike(id=model_name, api_key=config["api_key"], base_url=config["base_url"]))

def prepare_embeddings(file_path):
    templates = load_templates(file_path)
    if not templates:
        return [], None
    
    template_texts = [t["general"] for t in templates]
    embedding_model = get_embedding_model()
    template_embeddings = embedding_model.encode(template_texts, convert_to_tensor=True)
    return templates, template_embeddings

template_file = "merged_final.json"
templates, template_embeddings = prepare_embeddings(template_file)

def rank_templates(query, templates, template_embeddings):
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, template_embeddings)[0].cpu().numpy()
    ranked_indices = np.argsort(scores)[::-1]
    return [(templates[i], float(scores[i])) for i in ranked_indices]

def generate_cot_summary(query, top_templates, agent):
    # Generate a chain-of-thought summary based on extracted template insights.
    avg_score = np.mean([s for _, s in top_templates]) if top_templates else 0
    enhanced_templates = [
        (t, float(avg_score * 1.1)) if s < avg_score * 0.8 else (t, float(s))
        for t, s in top_templates
    ]
    unique_content = list(set(
        [content for template, _ in enhanced_templates if "content" in template for content in template["content"]]
    ))
    ensemble_text = "\n".join(unique_content)

    if not ensemble_text.strip():
        return "No relevant templates were found to summarize."
    
    summary_response = agent.run(f"Summarize this content concisely: {ensemble_text}").content.strip()
    return summary_response

# Qwen reasoning enhancement using a customized prompt based on your framework.
def enhance_with_qwen(cot_summary, query, agent):
    if not cot_summary.strip():
        return ""

    prompt = (
        "You are AI, developed by Finsocial Digital System. Use the following summarized insights "
        "from similar templates to generate a clear and well-structured final answer to the user's query. "
        "You can perform internal reasoning, but your final answer must be provided only under <|answer|>.\n\n"
        "---\n"
        f"User Query:\n{query}\n\n"
        f"Template Insights (Summarized):\n{cot_summary}\n"
        "---\n\n"
        "First reason internally as needed, but DO NOT include that in the final output.\n"
        "Your final answer must start after the tag: <|answer|>\n"
    )

    # Get model output (reasoning + final answer)
    output = agent.run(prompt).content.strip()

    # Extract only the final answer from the output
    if "<|answer|>" in output:
        final_answer = output.split("<|answer|>", 1)[1].strip()
    else:
        # Fallback: return entire output if tag not found
        final_answer = output

    return final_answer



def process_query(query):
    response_data = {}
    threshold = 0.5

    ranked_templates = rank_templates(query, templates, template_embeddings)
    top_templates = [(template, score) for template, score in ranked_templates if score > threshold]
    top_10 = top_templates[:10] if len(top_templates) >= 10 else top_templates

    agent_direct = get_agent(selected_model)
    agent_enhanced = get_agent(selected_model)
    agent_cot = get_agent("llama3.2:1b")
    agent_qwen = get_agent("Qwen/QwQ-32B-AWQ")

    def get_direct_response():
        with st.spinner("Generating Direct Model Response..."):
            return agent_direct.run(query).content.strip()
        
    def get_enhanced_response():
        if top_10:
            cot_summary = generate_cot_summary(query, top_10, agent_cot)
            qwen_final_answer = enhance_with_qwen(cot_summary, query, agent_qwen)
            return qwen_final_answer
        return "No relevant templates found for enhancement."
    
    
    with ThreadPoolExecutor() as executor:
        future_direct = executor.submit(get_direct_response)
        direct_response = future_direct.result()

    enhanced_response = get_enhanced_response()
    cot_summary = generate_cot_summary(query, top_10, agent_cot)

    response_data["full_direct_response"] = direct_response
    response_data["template_direct_response"] = enhanced_response
    response_data["cot_summary"] = cot_summary

    return response_data

# Main Interface
if templates:
    query = st.text_area("Enter your query:", height=100)
    process_button = st.button("Process Query")

    if process_button and query:
        result = process_query(query)

        with st.expander("📌 Direct Response From Model", expanded=True):
            st.write(result["full_direct_response"])

        with st.expander("📌 Enhanced Response", expanded=True):
            st.write(result["template_direct_response"])

        with st.expander("🔍 Chain-of-Thought Reasoning", expanded=True):
            st.write(result.get("cot_summary", ""))

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: #f0f2f6; padding: 10px; text-align: center; border-top: 2px solid #2c3e50;">
            © 2024 Finsocial Digital System. All rights reserved. Empowering Financial Intelligence.
        </div>
        """, 
        unsafe_allow_html=True
    )
else:
    st.error("No templates loaded. Please check your template file path and format.")
