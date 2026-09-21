# SilverVault

**A RAG-based academic learning assistant for BTech and MTech CS/IT students.**

SilverVault is a college project that aims to help CS/IT students study their semester subjects using **Retrieval-Augmented Generation (RAG)**.

Instead of going through large reference books for every concept, students can ask questions and receive concise, source-grounded explanations based on the selected academic material.

## Project Objective

The goal of SilverVault is to create a simple AI-powered academic platform that helps students:

* Understand core concepts from their semester subjects
* Learn from reference books and academic material
* Ask questions using natural language
* Take mock examinations
* Analyze their performance
* Identify weak concepts
* Receive targeted revision recommendations

## Project Status

**Currently under development.**

The project is being developed as a college NLP/RAG project.

## Planned Features

### AI Study Assistant

Students can ask questions about their academic subjects and receive answers based on the provided reference material.

###  Retrieval-Augmented Generation

SilverVault will retrieve relevant information from academic documents before generating an answer.

###  Mock Examinations

Students will be able to attempt subject-wise mock tests containing questions of different difficulty levels.

###  Performance Analysis

The system will analyze examination results and identify topics where the student needs improvement.

###  Personalized Revision

Based on weak topics, SilverVault will recommend relevant concepts and revision material.

##  Planned RAG Pipeline

```text
Academic Documents
        ↓
   Text Extraction
        ↓
      Chunking
        ↓
     Embeddings
        ↓
   Vector Database
        ↓
      Retrieval
        ↓
        LLM
        ↓
 Answer + Source
```

## 🛠️ Planned Technology Stack

| Component            | Technology            |
| -------------------- | --------------------- |
| Programming Language | Python                |
| Web Application      | Streamlit             |
| NLP / RAG            | LangChain             |
| Embeddings           | Sentence Transformers |
| Vector Database      | FAISS                 |
| LLM                  | To be decided         |
| Document Processing  | PyPDF                 |

##  Project Structure

```text
silvervault/
│
├── app/
│   └── app.py
│
├── data/
│   └── README.md
│
├── src/
│   ├── rag.py
│   ├── embeddings.py
│   └── utils.py
│
├── tests/
│   └── test_rag.py
│
├── docs/
│   └── project_notes.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

##  Target Users

SilverVault is intended primarily for:

* BTech CS/IT students
* MTech CS/IT students
* Students studying semester-based computer science subjects

## 🔬 Research Direction

The project will also explore the use of RAG for academic question answering, particularly how retrieved academic material can help provide more relevant and source-grounded answers.

Possible areas of investigation include:

* Retrieval quality
* Chunking strategies
* Embedding models
* RAG vs. non-RAG question answering
* Answer relevance and correctness

##  Project Disclaimer

SilverVault is an educational college project and is not intended to replace textbooks, teachers, university resources, or professional academic guidance.

##  Development

This project is being developed as part of a college academic project.

**Project Name:** SilverVault
**Domain:** NLP / RAG / Generative AI / Educational Technology
