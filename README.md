# AI-Powered Email Triage Decision Support System

> An advanced **AI-driven Decision Support System (DSS)** designed to revolutionize enterprise email management by automating triage with zero-shot explainability, priority classification, department-specific sorting, and actionable next steps based on company SOPs.



## Project Overview

This flagship project implements a hybrid Decision Support System combining **Model-driven**, **Data-driven**, and **Knowledge-driven** approaches to address the critical challenge of managing high-volume corporate email inflow. Business users are overwhelmed with hundreds of daily emails, where crucial communications are often buried in neutral or ambiguous language, causing delays and operational risks.

By leveraging **zero-shot AI models** (facebook/bart-large-mnli), **business rule integration**, and **context-aware retrieval augmented generation (RAG)** with LangChain and ChromaDB, this system delivers:

- **Explainable Department Classification:** Intelligently categorizes emails into Finance/Tax and IT with descriptive labeling for precise routing.
- **Hybrid Priority Scoring:** Combines AI confidence scores with business logic for High, Medium, and Low priority tagging, maintaining transparency via confidence tooltips and rule-based overrides.
- **Actionable Recommendations:** Uses LangChain RAG and SOP context embeddings to generate concise, reliable next-step actions on high-priority emails.
- **Interactive Enterprise UI:** Streamlit interface enabling department filtering, priority visualization, and rich email summaries — empowering decision makers with data-driven insights at a glance.

This system empowers organizations to save time, reduce risk, and ensure consistent compliance, setting a new standard for AI-assisted communication management.



## Key Features

- **Zero-shot Explainability:** No training data needed, ensuring adaptability and full transparency in decision-making.
- **Multi-Dimensional Classification:** Department and priority classifiers built on robust transformer models.
- **AI with Business Rule Governance**: Model predictions are enhanced with confidence-based thresholds and business logic to ensure oversight, accountability, and explainable overrides — combining the strengths of AI with domain expertise.
- **SOP-Based Action Recommendation**: Integrates Retrieval-Augmented Generation (RAG) to suggest next steps based on departmental SOPs - fully grounded in company policies.
- **Visual Analytics:** Priority distribution graphs with confidence score tooltips for informed triage.
- **Scalable and Modular:** Designed with open-source frameworks to support easy integration and future enhancements.



## Tech Stack

| Component             | Technology / Model                    |
|-----------------------|-------------------------------------|
| AI Classification     | `facebook/bart-large-mnli` (Zero-shot) |
| Embeddings            | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Retrieval Database    | ChromaDB                            |
| Generative LLM        | OpenAI GPT-3.5 Turbo |
| RAG Framework         | LangChain                          |
| Frontend UI           | Streamlit                          |
| Programming Language  | Python                            |



## Problem Statement

- Enterprise teams receive hundreds of emails daily, leading to information overload.
- Critical emails are often hidden in neutral or complex language, delaying responses.
- Manual triaging is time-consuming, error-prone, and risky for business continuity.



## Solution Overview

1. **Email ingestion:** User uploads CSV with emails.
2. **Department Classification:** Zero-shot model predicts Finance/Tax or IT category.
3. **Priority Scoring:** Hybrid AI and business-rule logic assign priority with explainable confidence.
4. **Summarization:** GPT-3.5 Turbo generates concise email summaries.
5. **Action Suggestion:** RAG retrieves SOP context and generates precise next steps for high-priority cases.
6. **Visualization & Filtering:** Streamlit UI provides dynamic filtering, priority charts, and detailed email views with explanations.



## Enterprise Impact

- Accelerates critical email response times.
- Enhances compliance and risk mitigation via SOP-driven guidance.
- Reduces cognitive load and operational overhead on teams.
- Provides a transparent, explainable AI workflow aligned with enterprise governance.



## Getting Started

Run:

```bash
streamlit run main.py
```
Upload your email dataset as a CSV containing these columns: Timestamp, Sender, Subject, Body
Experience seamless, intelligent triage with actionable insights instantly.



## Streamlit UI Detailed Screenshots

To explore the interactive email triaging dashboard in detail and see the system in action, please refer to the PDF below showcasing the Streamlit interface:

📄 [View Streamlit UI Screenshots](./outputs.pdf)

This PDF showcases:

- Uploading email CSV file and system ingestion
- Department-based filtering of emails (Finance/Tax and IT)
- Dynamic bar charts displaying email priority distribution
- Detailed email views with summarized content, priority labels, and confidence explanations
- Suggested actions for high-priority emails with transparent AI-business rule insights



## Highlights

### Screenshot 1
<img width="1510" height="698" alt="image" src="https://github.com/user-attachments/assets/2d99881a-408d-4d95-8868-8ae815b9a39a" />

### Screenshot 2
<img width="1514" height="743" alt="image" src="https://github.com/user-attachments/assets/daf5f166-d3dd-4963-a762-b749da0b382f" />


<img width="1506" height="731" alt="image" src="https://github.com/user-attachments/assets/a32010b4-bd59-421f-a048-fd3d19e67583" />


<img width="1511" height="746" alt="image" src="https://github.com/user-attachments/assets/f12cc947-f62f-4d9a-9735-2f685462a1c6" />


<img width="1512" height="735" alt="image" src="https://github.com/user-attachments/assets/d09fa452-f738-405d-aa36-94588d539e40" />


<img width="1537" height="731" alt="image" src="https://github.com/user-attachments/assets/d717b579-445a-47c7-97fa-0b79a7afa377" />


<img width="1528" height="740" alt="image" src="https://github.com/user-attachments/assets/30ed5f82-14b5-4368-b103-9f1a20a26124" />



# Contact
Anusha Chaudhuri [anusha761]
