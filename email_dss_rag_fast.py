import pandas as pd
import streamlit as st
from transformers import pipeline
from openai import OpenAI
import os

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# === CACHED RESOURCES ===
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_db(_embedding_model):  # <- Leading underscore fixes unhashable param issue
    return Chroma(
        persist_directory="chroma_sop",
        embedding_function=_embedding_model,
        collection_name="sop_collection"
    )

@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_llm(api_key):
    return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key, temperature=0.2)

# === Initialize cached models ===
embedding_model = load_embedding_model()
vector_db = load_vector_db(embedding_model)
classifier = load_classifier()
openai_key = "xxx"  # Replace with actual key
client = OpenAI(api_key=openai_key)
llm = load_llm(openai_key)

# === Priority Classification ===
def classify_priority(text):
    labels = [
        "This email is urgent and requires immediate action to prevent penalties or major risks.",
        "This email needs to be addressed within a few days to avoid escalation.",
        "This email is informational and does not require urgent action."
    ]
    label_map = {
        labels[0]: "High",
        labels[1]: "Medium",
        labels[2]: "Low"
    }
    result = classifier(text, labels)
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    raw_priority = label_map[top_label]

    if top_score >= 0.53:
        priority = raw_priority
    elif top_score >= 0.50:
        priority = raw_priority if raw_priority in ["Medium", "Low"] else "Medium"
    else:
        priority = "Medium"
    return priority, round(top_score, 2), label_map[top_label]

def explain_priority(model_label, confidence_score, final_label):
    explanation = f"Model predicted '{model_label}' with {round(confidence_score * 100)}% confidence."
    if final_label != model_label:
        explanation += f" Overridden to '{final_label}' due to business rules."
    else:
        if confidence_score >= 0.70:
            explanation += " High confidence accepted."
        elif confidence_score >= 0.50:
            explanation += " Acceptable confidence accepted."
        else:
            explanation += " Moderate confidence, use with caution."
    return explanation

# === Department Classifier ===
def classify_department(text):
    labels = [
        "This email is about finance, tax, invoicing, or budgeting.",
        "This email is about IT systems, software access, or technical support."
    ]
    label_map = {
        labels[0]: "Finance/Tax",
        labels[1]: "IT"
    }
    result = classifier(text, labels)
    return label_map[result['labels'][0]]

# === GPT Summary ===
def generate_summary(text):
    prompt = f"Summarize the following email in one sentence:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# === RAG Chain ===
def get_suggestive_action(summary, department):
    retriever = vector_db.as_retriever(search_kwargs={"k": 2, "filter": {"department": department}})
    
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template=(
            "You are an assistant trained to follow company SOPs strictly. You will be given an email summary and relevant SOP context.\n\n"
            "Your task is to generate a clear and complete action recommendation based **only on the SOP context**.\n\n"
            "The recommendation must be **1 to 2 concise sentences** that include **all relevant actions, emails, or links** explicitly mentioned in the SOP.\n"
            "- Do not omit important steps.\n"
            "- Do not invent or assume anything not found in the SOP context.\n"
            "- Only include exact information present in the SOP.\n\n"
            "If no matching instructions are found in the SOP, respond with exactly:\n"
            "'No applicable SOP found'\n\n"
            "### Email Summary:\n{question}\n\n"
            "### SOP Context:\n{context}"
        )
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False
    )

    return qa_chain.run(summary)

# === Process Emails ===
def process_emails(df):
    summaries, departments, priorities, prio_confidences, explanations, suggestions = [], [], [], [], [], []

    for _, row in df.iterrows():
        full_text = f"{row['Subject']} {row['Body']}"
        dept = classify_department(full_text)
        prio, prio_conf, prio_label = classify_priority(full_text)
        explanation = explain_priority(prio_label, prio_conf, prio)
        summary = generate_summary(full_text)

        if prio == "High":
            action = get_suggestive_action(summary, dept)
        else:
            action = ""

        summaries.append(summary)
        departments.append(dept)
        priorities.append(prio)
        prio_confidences.append(prio_conf)
        explanations.append(explanation)
        suggestions.append(action)

    df["Department"] = departments
    df["Priority"] = priorities
    df["Priority Confidence"] = prio_confidences
    df["Priority Explanation"] = explanations
    df["Summary"] = summaries
    df["Suggestive Action"] = suggestions

    order = {"High": 3, "Medium": 2, "Low": 1}
    df["Priority Rank"] = df["Priority"].map(order)
    df = df.sort_values(by=["Priority Rank", "Priority Confidence"], ascending=[False, False])
    df = df.drop(columns=["Priority Rank"])
    return df

# === Streamlit App ===
def main():
    st.set_page_config(page_title="AI Email Triaging DSS", layout="wide")
    st.title("AI-Based Email Triaging Decision Support System")

    uploaded_file = st.file_uploader("Upload your email CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        expected = {"Timestamp", "Sender", "Subject", "Body"}

        if expected.issubset(df.columns):
            with st.spinner("AI is analyzing your emails..."):
                df_out = process_emails(df)

            st.subheader("Filtered Results")
            dept_filter = st.selectbox("Filter by Department", df_out["Department"].unique())
            filtered = df_out[df_out["Department"] == dept_filter]

            st.bar_chart(filtered["Priority"].value_counts())
            

            for _, row in filtered.iterrows():
                st.markdown(f"""
                    <b>Subject:</b> {row['Subject']}  
                    <b>Sender:</b> {row['Sender']}  
                    <b>Summary:</b> {row['Summary']}  
                    <b>Priority:</b> <span title="{row['Priority Explanation']}">{row['Priority']}</span>  
                    {"<b>Suggested Action:</b> " + row['Suggestive Action'] if row['Priority'] == "High" and row['Suggestive Action'] else ""} 
                    <hr>
                """, unsafe_allow_html=True)
        else:
            st.error("CSV must contain columns: Timestamp, Sender, Subject, Body")

if __name__ == "__main__":
    main()
