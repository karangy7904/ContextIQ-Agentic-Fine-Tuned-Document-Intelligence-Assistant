import os

import streamlit as st

from groq import Groq
from pypdf import PdfReader
from transformers import pipeline

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from langchain_community.vectorstores import (
    FAISS
)
from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from agents import run_document_agent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📄",
    layout="wide"
)

st.title(
    "📄 DocuChat AI"
)

st.caption(
    "Agentic Fine-Tuned PDF "
    "Question Answering Assistant"
)


# ============================================================
# SETTINGS
# ============================================================

def get_setting(
    name,
    default=None
):

    value = os.getenv(name)

    if value:
        return value

    try:

        value = st.secrets.get(
            name
        )

        if value:
            return value

    except Exception:
        pass

    return default


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def get_groq_client():

    api_key = get_setting(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    return Groq(
        api_key=api_key
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )


# ============================================================
# FINE-TUNED QA MODEL
# ============================================================

@st.cache_resource
def load_qa_model():

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # Set FINE_TUNED_QA_MODEL to your own
    # Hugging Face model repository or local
    # fine-tuned model folder.
    #
    # Example:
    #
    # FINE_TUNED_QA_MODEL =
    # "karangy7904/my-finetuned-qa-model"
    #
    # If it is not configured, the app uses
    # a standard QA model so the app can run.
    # --------------------------------------------------------

    model_name = get_setting(
        "FINE_TUNED_QA_MODEL",
        "deepset/roberta-base-squad2"
    )

    return pipeline(
        task="question-answering",
        model=model_name,
        tokenizer=model_name
    )


# ============================================================
# SESSION STATE
# ============================================================

if "vectorstore" not in (
    st.session_state
):

    st.session_state.vectorstore = None


if "documents" not in (
    st.session_state
):

    st.session_state.documents = []


if "sources" not in (
    st.session_state
):

    st.session_state.sources = []


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_documents(
    uploaded_files
):

    documents = []

    for uploaded_file in (
        uploaded_files
    ):

        try:

            uploaded_file.seek(0)

            reader = PdfReader(
                uploaded_file
            )

            for page_number, page in (
                enumerate(
                    reader.pages,
                    start=1
                )
            ):

                text = (
                    page.extract_text()
                    or ""
                )

                text = text.strip()

                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source":
                                uploaded_file.name,
                            "page":
                                page_number
                        }
                    )
                )

        except Exception as exc:

            st.warning(
                f"Could not process "
                f"{uploaded_file.name}: "
                f"{exc}"
            )

    return documents


# ============================================================
# CHUNK DOCUMENTS
# ============================================================

def split_documents(
    documents
):

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
    )

    return splitter.split_documents(
        documents
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader(
    "1. Upload PDF Documents"
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS DOCUMENT BUTTON
# ============================================================

if (
    uploaded_files
    and st.button(
        "Process PDFs",
        type="primary"
    )
):

    # --------------------------------------------------------
    # EXTRACT TEXT
    # --------------------------------------------------------

    with st.spinner(
        "Extracting text from PDFs..."
    ):

        page_documents = (
            extract_pdf_documents(
                uploaded_files
            )
        )

    if not page_documents:

        st.error(
            "No readable text was found "
            "inside the uploaded PDFs."
        )

        st.stop()

    # --------------------------------------------------------
    # CHUNK DOCUMENTS
    # --------------------------------------------------------

    with st.spinner(
        "Splitting documents "
        "into searchable chunks..."
    ):

        chunks = split_documents(
            page_documents
        )

    # --------------------------------------------------------
    # EMBEDDINGS + FAISS
    # --------------------------------------------------------

    with st.spinner(
        "Creating document embeddings "
        "and FAISS index..."
    ):

        embeddings = (
            load_embeddings()
        )

        vectorstore = (
            FAISS.from_documents(
                chunks,
                embeddings
            )
        )

    # --------------------------------------------------------
    # SAVE STATE
    # --------------------------------------------------------

    st.session_state.vectorstore = (
        vectorstore
    )

    st.session_state.documents = (
        chunks
    )

    st.session_state.sources = sorted(
        {
            doc.metadata.get(
                "source",
                "Unknown"
            )
            for doc in chunks
        }
    )

    st.success(
        f"Processed "
        f"{len(st.session_state.sources)} "
        f"PDF document(s) into "
        f"{len(chunks)} searchable chunks."
    )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if (
    st.session_state.vectorstore
    is not None
):

    st.divider()

    st.subheader(
        "2. Document Information"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "PDF Documents",
        len(
            st.session_state.sources
        )
    )

    col2.metric(
        "Searchable Chunks",
        len(
            st.session_state.documents
        )
    )

    with st.expander(
        "📚 Uploaded Documents"
    ):

        for source in (
            st.session_state.sources
        ):

            st.write(
                f"• {source}"
            )

    # ========================================================
    # AI QUESTION AREA
    # ========================================================

    st.divider()

    st.subheader(
        "3. Ask DocuChat Agents"
    )

    st.write(
        "You can ask factual questions, "
        "request summaries, or compare "
        "multiple uploaded PDFs."
    )

    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: What is the main "
            "conclusion of this document?"
        )
    )

    ask_button = st.button(
        "Ask AI Agents",
        type="primary"
    )

    # ========================================================
    # RUN AGENTS
    # ========================================================

    if (
        question
        and ask_button
    ):

        try:

            with st.spinner(
                "Loading AI agents..."
            ):

                client = (
                    get_groq_client()
                )

                qa_pipeline = (
                    load_qa_model()
                )

            with st.spinner(
                "Agents are analyzing "
                "the documents..."
            ):

                result = (
                    run_document_agent(
                        question=question,
                        vectorstore=(
                            st.session_state
                            .vectorstore
                        ),
                        all_documents=(
                            st.session_state
                            .documents
                        ),
                        qa_pipeline=(
                            qa_pipeline
                        ),
                        client=client
                    )
                )

        except Exception as exc:

            st.error(
                f"Agent execution failed: "
                f"{exc}"
            )

            st.stop()

        # ====================================================
        # FINAL ANSWER
        # ====================================================

        st.subheader(
            "🤖 Answer"
        )

        st.write(
            result["answer"]
        )

        # ====================================================
        # VERIFICATION STATUS
        # ====================================================

        verification = (
            result[
                "verification"
            ]
        )

        if verification[
            "supported"
        ]:

            st.success(
                "✅ Answer verified against "
                "retrieved PDF evidence."
            )

        else:

            st.warning(
                "⚠️ The verification agent "
                "could not strongly verify "
                "this answer."
            )

        # ====================================================
        # AGENT EXECUTION TRACE
        # ====================================================

        with st.expander(
            "🤖 Agent Execution Trace"
        ):

            plan = result[
                "plan"
            ]

            route_labels = {
                "qa":
                    (
                        "🔍 Retrieval Agent → "
                        "🧠 Fine-Tuned QA Agent → "
                        "✅ Verification Agent"
                    ),

                "summary":
                    (
                        "🔍 Retrieval Agent → "
                        "📝 Summary Agent → "
                        "✅ Verification Agent"
                    ),

                "compare":
                    (
                        "🔍 Multi-Document "
                        "Retrieval → "
                        "⚖️ Comparison Agent → "
                        "✅ Verification Agent"
                    )
            }

            st.write(
                "**Route selected:**",
                plan["route"]
            )

            st.write(
                "**Agents executed:**",
                route_labels.get(
                    plan["route"],
                    plan["route"]
                )
            )

            st.write(
                "**Verification requested:**",
                (
                    "Yes"
                    if plan[
                        "needs_verification"
                    ]
                    else "No"
                )
            )

            st.write(
                "**Retrieval retry used:**",
                (
                    "Yes"
                    if result[
                        "retried"
                    ]
                    else "No"
                )
            )

        # ====================================================
        # FINE-TUNED MODEL OUTPUT
        # ====================================================

        if (
            result["qa_result"]
            is not None
        ):

            with st.expander(
                "🧠 Fine-Tuned QA "
                "Agent Result"
            ):

                qa_result = (
                    result[
                        "qa_result"
                    ]
                )

                st.write(
                    "**Extracted answer:**",
                    qa_result.get(
                        "answer",
                        "Not available"
                    )
                )

                score = qa_result.get(
                    "score",
                    0
                )

                st.write(
                    "**QA confidence score:**",
                    f"{score:.4f}"
                )

                source_doc = (
                    qa_result.get(
                        "document"
                    )
                )

                if (
                    source_doc
                    is not None
                ):

                    st.write(
                        "**Best source:**",
                        (
                            f"{source_doc.metadata.get(
                                'source',
                                'Unknown'
                            )}, "
                            f"page "
                            f"{source_doc.metadata.get(
                                'page',
                                '?'
                            )}"
                        )
                    )

        # ====================================================
        # VERIFICATION DETAILS
        # ====================================================

        with st.expander(
            "✅ Verification Agent Result"
        ):

            st.write(
                "**Supported:**",
                (
                    "Yes"
                    if verification[
                        "supported"
                    ]
                    else "No"
                )
            )

            st.write(
                "**Confidence:**",
                verification[
                    "confidence"
                ]
            )

            st.write(
                "**Explanation:**",
                verification[
                    "explanation"
                ]
            )

        # ====================================================
        # RETRIEVED SOURCES
        # ====================================================

        with st.expander(
            "🔍 Retrieved PDF Evidence"
        ):

            retrieved_documents = (
                result[
                    "retrieval"
                ]
            )

            if not retrieved_documents:

                st.info(
                    "No relevant document "
                    "chunks were retrieved."
                )

            else:

                for index, doc in (
                    enumerate(
                        retrieved_documents,
                        start=1
                    )
                ):

                    source = (
                        doc.metadata.get(
                            "source",
                            "Unknown"
                        )
                    )

                    page = (
                        doc.metadata.get(
                            "page",
                            "?"
                        )
                    )

                    st.markdown(
                        f"### Evidence {index}"
                    )

                    st.write(
                        f"**Document:** {source}"
                    )

                    st.write(
                        f"**Page:** {page}"
                    )

                    st.write(
                        doc.page_content
                    )

                    st.divider()


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "Upload one or more PDFs and "
        "click **Process PDFs** to begin."
    )

