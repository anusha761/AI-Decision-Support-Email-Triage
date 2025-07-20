import pandas as pd
import streamlit as st
from transformers import pipeline
from openai import OpenAI

from langgraph.graph import StateGraph, END

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from typing import TypedDict, Optional



# Load vector DB and models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_sop", embedding_function=embedding_model,collection_name="sop_collection")


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Read openai key
with open("openai_key.txt", "r") as f:
    openai_key = f.read().strip()
client = OpenAI(api_key=openai_key)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="chroma_sop", embedding_function=embedding_model, collection_name="sop_collection")


# Setting Agent State
class AgentState(TypedDict):
    text: str
    department: Optional[str]
    summary: Optional[str]
    priority: Optional[str]
    priority_confidence: Optional[float]
    priority_explanation: Optional[str]
    suggestive_action: Optional[str]
    activity: Optional[str]



# Node Functions
def dept_node(state):
    labels = [
        "This email is about finance, tax, invoicing, or budgeting.",
        "This email is about IT systems, software access, or technical support."
    ]
    label_map = {
        labels[0]: "Finance/Tax",
        labels[1]: "IT"
    }
    result = classifier(state["text"], labels)
    #print(result)
    return {**state, "department": label_map[result["labels"][0]]}

def summary_node(state):
    prompt = f"Summarize the following email in one sentence:\n\n{state['text']}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    #print("summary done")
    return {**state, "summary": response.choices[0].message.content.strip()}

def priority_node(state):
    labels = [
        "This email is urgent and requires immediate action to prevent penalties or major risks.",
        "This email needs to be addressed within a few days to avoid escalation.",
        "This email is informational and does not require urgent action."
    ]
    label_map = {labels[0]: "High", labels[1]: "Medium", labels[2]: "Low"}

    result = classifier(state["text"], labels)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    raw_priority = label_map[top_label]

    if top_score >= 0.53:
        final_priority = raw_priority
    elif top_score >= 0.50:
        final_priority = raw_priority if raw_priority in ["Medium", "Low"] else "Medium"
    else:
        final_priority = "Medium"

    explanation = f"Model predicted priority with {round(top_score * 100)}% confidence."
    if final_priority != raw_priority:
        explanation += f" Overridden to '{final_priority}' due to business rules."
    else:
        explanation += " Accepted as is."

    
    activity="End"
    if final_priority=="High":
        activity="Action"
    
    
    #print(final_priority, activity)
    return {
        **state,
        "priority": final_priority,
        "priority_confidence": round(top_score, 2),
        "priority_explanation": explanation,
        "activity": activity
    }



def action_node(state):
    retriever = vector_db.as_retriever(search_kwargs={"k": 2, "filter": {"department": state["department"]}})
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
    ))
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False
    )
    #print("action ran")
    return {**state, "suggestive_action": qa.run(state["summary"])}

# LangGraph Setup
graph = StateGraph(AgentState)

graph.add_node("Dept", dept_node)
graph.add_node("Summary", summary_node)
graph.add_node("Priority", priority_node)
graph.add_node("Action", action_node)

# Edges (in sequence)
graph.set_entry_point("Dept")
graph.add_edge("Dept", "Summary")
graph.add_edge("Summary", "Priority")

# Conditional edge based on priority
graph.add_conditional_edges("Priority", lambda state: state["activity"], {
    "Action": "Action",
    "End": END
})

# Action leads to end
graph.add_edge("Action", END)

runnable=graph.compile()

# Email Processing
def process_emails(df):
    results = []

    for _, row in df.iterrows():
        text = f"{row['Subject']} {row['Body']}"
        #print(row['Subject'])
        output = runnable.invoke({"text": text})

        results.append({
            "Department": output.get("department", ""),
            "Summary": output.get("summary", ""),
            "Priority": output.get("priority", ""),
            "Priority Confidence": output.get("priority_confidence", ""),
            "Priority Explanation": output.get("priority_explanation", ""),
            "Suggestive Action": output.get("suggestive_action", "")
        })

    df2 = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    order = {"High": 3, "Medium": 2, "Low": 1}
    df2["Priority Rank"] = df2["Priority"].map(order)
    df2 = df2.sort_values(by=["Priority Rank", "Priority Confidence"], ascending=[False, False]).drop(columns=["Priority Rank"])
    return df2




# Streamlit App
def main():
    st.set_page_config(page_title="AI Email Triaging DSS", layout="wide")
    st.title("AI Email Triaging Decision Support System with LangGraph")

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
