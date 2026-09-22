import torch
import json
import os
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _document_label(doc):
    source = doc.metadata.get(
        "source",
        "Unknown document"
    )

    page = doc.metadata.get(
        "page",
        "?"
    )

    return f"{source} - page {page}"


def _context_from_documents(
    documents,
    max_chars=18000
):

    context_parts = []
    current_length = 0

    for doc in documents:

        label = _document_label(doc)

        text = doc.page_content.strip()

        block = (
            f"\n[Source: {label}]\n"
            f"{text}\n"
        )

        if (
            current_length + len(block)
            > max_chars
        ):
            break

        context_parts.append(block)

        current_length += len(block)

    return "\n".join(
        context_parts
    )


# ============================================================
# 1. ORCHESTRATOR AGENT
# ============================================================

def plan_question(
    question,
    client,
    number_of_documents=1
):

    system_prompt = f"""
You are the orchestrator for an intelligent PDF
question-answering system.

The system currently contains
{number_of_documents} uploaded document(s).

Choose exactly one route:

qa
Use this when the user asks a factual question,
asks for an explanation, definition, detail,
specific information, or asks what the document says.

summary
Use this when the user asks to summarize,
give an overview, list key points, explain the
whole document, or identify the main ideas.

compare
Use this only when the user wants to compare,
contrast, find differences, similarities, or
relationships between multiple documents.

Also decide whether answer verification is needed.
Normally verification should be true.
"""

    schema = {
        "type": "object",
        "properties": {
            "route": {
                "type": "string",
                "enum": [
                    "qa",
                    "summary",
                    "compare"
                ]
            },
            "needs_verification": {
                "type": "boolean"
            }
        },
        "required": [
            "route",
            "needs_verification"
        ],
        "additionalProperties": False
    }

    try:

        response = (
            client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name":
                            "document_agent_plan",
                        "strict": True,
                        "schema": schema
                    }
                },
                temperature=0
            )
        )

        return json.loads(
            response
            .choices[0]
            .message
            .content
        )

    except Exception:

        # Fallback routing if structured
        # generation fails.

        question_lower = (
            question.lower()
        )

        comparison_words = [
            "compare",
            "difference",
            "differences",
            "similarities",
            "contrast",
            "versus",
            " vs "
        ]

        summary_words = [
            "summarize",
            "summary",
            "overview",
            "main points",
            "key points",
            "main ideas"
        ]

        if (
            number_of_documents > 1
            and any(
                word in question_lower
                for word in comparison_words
            )
        ):

            route = "compare"

        elif any(
            word in question_lower
            for word in summary_words
        ):

            route = "summary"

        else:

            route = "qa"

        return {
            "route": route,
            "needs_verification": True
        }


# ============================================================
# 2. RETRIEVAL AGENT
# ============================================================

def retrieval_agent(
    question,
    vectorstore,
    k=6,
    source=None
):

    filter_dict = None

    if source is not None:

        filter_dict = {
            "source": source
        }

    documents = (
        vectorstore.similarity_search(
            question,
            k=k,
            filter=filter_dict
        )
    )

    return documents


# ============================================================
# 3. FINE-TUNED QA AGENT
# ============================================================

def fine_tuned_qa_agent(
    question,
    documents,
    qa_pipeline
):

    model = qa_pipeline["model"]
    tokenizer = qa_pipeline["tokenizer"]

    context = _context_from_documents(
        documents,
        max_chars=12000
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Answer only from the supplied "
                "document context. If the answer "
                "cannot be found in the context, "
                "say that the document does not "
                "provide enough information."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question:\n{question}"
            )
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return {
        "answer": answer,
        "score": None,
        "document": (
            documents[0]
            if documents
            else None
        )
    }

# ============================================================
# 4. GROUNDED RESPONSE AGENT
# ============================================================

def grounded_response_agent(
    question,
    qa_result,
    documents,
    client
):

    context = (
        _context_from_documents(
            documents
        )
    )

    extracted_answer = (
        qa_result.get(
            "answer",
            ""
        )
    )

    prompt = f"""
You are a document intelligence assistant.

The Fine-Tuned QA Agent produced this candidate
answer:

{extracted_answer}

User question:

{question}

Retrieved evidence:

{context}

Create a clear final response.

Rules:

1. Use only information supported by the
   retrieved document evidence.

2. The fine-tuned QA result is a candidate,
   not automatically correct.

3. Do not invent facts.

4. If the evidence does not support a reliable
   answer, clearly say that the document does
   not provide enough information.

5. Keep the answer concise but informative.

6. Do not invent page numbers.
"""

    response = (
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# 5. SUMMARY AGENT
# ============================================================

def summary_agent(
    question,
    documents,
    client
):

    context = (
        _context_from_documents(
            documents,
            max_chars=24000
        )
    )

    prompt = f"""
You are a document summarization agent.

User request:

{question}

Document content:

{context}

Create a grounded summary using only the
provided document content.

Include:

- main idea
- important points
- important conclusions

Do not add information that is not contained
in the document.
"""

    response = (
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# 6. DOCUMENT COMPARISON AGENT
# ============================================================

def comparison_agent(
    question,
    documents_by_source,
    client
):

    sections = []

    for source, docs in (
        documents_by_source.items()
    ):

        context = (
            _context_from_documents(
                docs,
                max_chars=7000
            )
        )

        sections.append(
            f"""
DOCUMENT: {source}

{context}
"""
        )

    combined_context = (
        "\n\n".join(sections)
    )

    prompt = f"""
You are a document comparison agent.

User request:

{question}

Documents:

{combined_context}

Compare the documents using only the supplied
evidence.

Clearly identify:

- similarities
- differences
- document-specific findings
- relevant conclusions

Do not invent claims that are not present in
the documents.
"""

    response = (
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# 7. VERIFICATION AGENT
# ============================================================

def verification_agent(
    question,
    answer,
    documents,
    client
):

    if not documents:

        return {
            "supported": False,
            "confidence": "low",
            "explanation":
                "No supporting document "
                "evidence was retrieved."
        }

    context = (
        _context_from_documents(
            documents
        )
    )

    prompt = f"""
You are a verification agent.

Determine whether the proposed answer is
supported by the retrieved PDF evidence.

QUESTION:

{question}

PROPOSED ANSWER:

{answer}

EVIDENCE:

{context}

Do not judge whether the answer sounds plausible.

Judge only whether the supplied document
evidence supports it.
"""

    schema = {
        "type": "object",
        "properties": {
            "supported": {
                "type": "boolean"
            },
            "confidence": {
                "type": "string",
                "enum": [
                    "low",
                    "medium",
                    "high"
                ]
            },
            "explanation": {
                "type": "string"
            }
        },
        "required": [
            "supported",
            "confidence",
            "explanation"
        ],
        "additionalProperties": False
    }

    try:

        response = (
            client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name":
                            "verification_result",
                        "strict": True,
                        "schema": schema
                    }
                },
                temperature=0
            )
        )

        return json.loads(
            response
            .choices[0]
            .message
            .content
        )

    except Exception as exc:

        return {
            "supported": False,
            "confidence": "low",
            "explanation":
                (
                    "Verification could not "
                    f"complete: {exc}"
                )
        }


# ============================================================
# 8. MAIN ORCHESTRATOR
# ============================================================

def run_document_agent(
    question,
    vectorstore,
    all_documents,
    qa_pipeline,
    client
):

    sources = sorted(
        {
            doc.metadata.get(
                "source",
                "Unknown document"
            )
            for doc in all_documents
        }
    )

    # --------------------------------------------------------
    # PLAN
    # --------------------------------------------------------

    plan = plan_question(
        question=question,
        client=client,
        number_of_documents=len(
            sources
        )
    )

    route = plan["route"]

    # Cannot compare only one PDF.
    if (
        route == "compare"
        and len(sources) < 2
    ):

        route = "qa"

        plan["route"] = "qa"

    retrieved_documents = []

    qa_result = None

    retried = False

    # ========================================================
    # QA ROUTE
    # ========================================================

    if route == "qa":

        retrieved_documents = (
            retrieval_agent(
                question=question,
                vectorstore=vectorstore,
                k=6
            )
        )

        qa_result = (
            fine_tuned_qa_agent(
                question=question,
                documents=(
                    retrieved_documents
                ),
                qa_pipeline=qa_pipeline
            )
        )

        answer = (
            grounded_response_agent(
                question=question,
                qa_result=qa_result,
                documents=(
                    retrieved_documents
                ),
                client=client
            )
        )

    # ========================================================
    # SUMMARY ROUTE
    # ========================================================

    elif route == "summary":

        summary_query = (
            "main ideas key points "
            "important conclusions "
            + question
        )

        retrieved_documents = (
            retrieval_agent(
                question=summary_query,
                vectorstore=vectorstore,
                k=12
            )
        )

        answer = (
            summary_agent(
                question=question,
                documents=(
                    retrieved_documents
                ),
                client=client
            )
        )

    # ========================================================
    # COMPARISON ROUTE
    # ========================================================

    else:

        documents_by_source = (
            defaultdict(list)
        )

        for source in sources:

            docs = retrieval_agent(
                question=question,
                vectorstore=vectorstore,
                k=4,
                source=source
            )

            documents_by_source[
                source
            ].extend(docs)

            retrieved_documents.extend(
                docs
            )

        answer = (
            comparison_agent(
                question=question,
                documents_by_source=(
                    documents_by_source
                ),
                client=client
            )
        )

    # ========================================================
    # VERIFY ANSWER
    # ========================================================

    verification = (
        verification_agent(
            question=question,
            answer=answer,
            documents=(
                retrieved_documents
            ),
            client=client
        )
    )

    # ========================================================
    # ONE RETRIEVAL RETRY FOR QA
    # ========================================================

    if (
        route == "qa"
        and not verification[
            "supported"
        ]
    ):

        retried = True

        retry_query = (
            question
            + " supporting evidence "
              "detailed explanation"
        )

        retrieved_documents = (
            retrieval_agent(
                question=retry_query,
                vectorstore=vectorstore,
                k=10
            )
        )

        qa_result = (
            fine_tuned_qa_agent(
                question=question,
                documents=(
                    retrieved_documents
                ),
                qa_pipeline=qa_pipeline
            )
        )

        answer = (
            grounded_response_agent(
                question=question,
                qa_result=qa_result,
                documents=(
                    retrieved_documents
                ),
                client=client
            )
        )

        verification = (
            verification_agent(
                question=question,
                answer=answer,
                documents=(
                    retrieved_documents
                ),
                client=client
            )
        )

    return {
        "answer": answer,
        "plan": plan,
        "retrieval":
            retrieved_documents,
        "qa_result": qa_result,
        "verification":
            verification,
        "retried": retried,
        "sources": sources
    }
