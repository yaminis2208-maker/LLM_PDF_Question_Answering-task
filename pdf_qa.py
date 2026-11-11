what from pathlib import Path
from collections import Counter
import re
import math

import pandas as pd
import chromadb
import ollama

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
PDF_PATH = DATASET_DIR / "sample.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"

DATASET_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

CHUNKS_CSV = DATASET_DIR / "chunks.csv"
QUERIES_CSV = DATASET_DIR / "queries.csv"
ANSWERS_CSV = DATASET_DIR / "answers.csv"
SEARCH_CSV = DATASET_DIR / "search_results.csv"

COLLECTION_NAME = "pdf_question_answering"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

TOP_K = 5


# ============================================================
# 2. CHECK PDF FILE
# ============================================================

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}\n"
        "Place sample.pdf inside the dataset folder."
    )

if PDF_PATH.stat().st_size == 0:
    raise ValueError("The PDF file is empty!")

print("\nPDF Question Answering System")
print("=" * 60)


# ============================================================
# 3. EXTRACT TEXT FROM PDF
# ============================================================

print("\nExtracting PDF text...")

reader = PdfReader(str(PDF_PATH))

page_texts = []
full_text = ""

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text() or ""

    if text.strip():
        page_texts.append({
            "page": page_number,
            "text": text
        })

        full_text += f"\n--- Page {page_number} ---\n{text}"

if not full_text.strip():
    raise ValueError(
        "No text was extracted. Your PDF may be scanned."
    )

print("PDF loaded successfully!")
print("Total pages:", len(reader.pages))
print("Extracted characters:", len(full_text))


# ============================================================
# 4. SAVE EXTRACTED TEXT
# ============================================================

with open(
    DATASET_DIR / "extracted_text.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(full_text)

print("Extracted text saved successfully!")


# ============================================================
# 5. SPLIT TEXT INTO CHUNKS
# ============================================================

print("\nSplitting PDF into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

documents = []

for page_data in page_texts:

    page_documents = splitter.create_documents(
        [page_data["text"]],
        metadatas=[{"page": page_data["page"]}]
    )

    documents.extend(page_documents)

if not documents:
    raise ValueError("No text chunks were generated.")

for index, document in enumerate(documents, start=1):
    document.metadata["chunk_id"] = index

chunks = [doc.page_content for doc in documents]

chunks_df = pd.DataFrame([
    {
        "chunk_id": doc.metadata["chunk_id"],
        "page": doc.metadata["page"],
        "text": doc.page_content
    }
    for doc in documents
])

chunks_df.to_csv(
    CHUNKS_CSV,
    index=False,
    encoding="utf-8"
)

print("Total chunks:", len(chunks))
print("Chunks saved to chunks.csv")


# ============================================================
# 6. CREATE OLLAMA EMBEDDINGS
# ============================================================

print("\nLoading Ollama embedding model...")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

print("Ollama embeddings loaded successfully!")


# ============================================================
# 7. STORE EMBEDDINGS IN CHROMADB
# ============================================================

print("\nCreating ChromaDB collection...")

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

# Remove the old collection to avoid duplicate chunks
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

vectorstore.add_documents(documents)

print("Embeddings stored successfully in ChromaDB!")


# ============================================================
# 8. KEYWORD SEARCH FUNCTION (BM25)
# ============================================================

def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


def keyword_search(query, documents, top_k=5):

    query_tokens = tokenize(query)

    tokenized_docs = [
        tokenize(doc.page_content)
        for doc in documents
    ]

    total_docs = len(documents)

    if total_docs == 0:
        return []

    document_frequency = Counter()

    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))

    average_length = (
        sum(len(tokens) for tokens in tokenized_docs)
        / total_docs
    )

    if average_length == 0:
        return []

    k1 = 1.5
    b = 0.75

    scores = []

    for index, tokens in enumerate(tokenized_docs):

        term_frequency = Counter(tokens)
        document_length = len(tokens)

        score = 0.0

        for term in query_tokens:

            frequency = term_frequency[term]

            if frequency == 0:
                continue

            df = document_frequency[term]

            idf = math.log(
                1 + (
                    total_docs - df + 0.5
                ) / (df + 0.5)
            )

            denominator = frequency + k1 * (
                1 - b + b * document_length / average_length
            )

            score += idf * (
                frequency * (k1 + 1)
            ) / denominator

        scores.append((index, score))

    scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return scores[:top_k]


# ============================================================
# 9. SEMANTIC SEARCH FUNCTION
# ============================================================

def semantic_search(query):

    results = vectorstore.similarity_search_with_score(
        query,
        k=min(TOP_K, len(documents))
    )

    search_results = []

    for rank, (doc, distance) in enumerate(
        results,
        start=1
    ):

        # Chroma uses cosine distance in this collection.
        similarity = 1.0 - float(distance)

        search_results.append({
            "rank": rank,
            "chunk_id": doc.metadata["chunk_id"],
            "page": doc.metadata["page"],
            "score": similarity,
            "distance": float(distance),
            "text": doc.page_content
        })

    return search_results


# ============================================================
# 10. SAVE CSV RESULTS
# ============================================================

def append_csv(data, file_path, columns):

    df = pd.DataFrame(data, columns=columns)

    df.to_csv(
        file_path,
        mode="a",
        header=not file_path.exists()
        or file_path.stat().st_size == 0,
        index=False,
        encoding="utf-8"
    )


# ============================================================
# 11. ASK QUESTIONS
# ============================================================

print("\nPDF chatbot is ready!")
print("Type 'exit' to stop.")

while True:

    query = input("\nEnter your question: ").strip()

    if query.lower() in ["exit", "quit", "stop"]:
        print("Thank you for using PDF Question Answering!")
        break

    if not query:
        print("Please enter a valid question.")
        continue

    # --------------------------------------------------------
    # SAVE QUERY
    # --------------------------------------------------------

    append_csv(
        [{"query": query}],
        QUERIES_CSV,
        ["query"]
    )

    # --------------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH - TOP 5 RESULTS")
    print("=" * 60)

    semantic_results = semantic_search(query)

    for result in semantic_results:

        print(f"\nRank: {result['rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Page: {result['page']}")
        print(
            f"Similarity Score: "
            f"{result['score']:.4f}"
        )

        print(result["text"][:300])

    # --------------------------------------------------------
    # KEYWORD SEARCH
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("KEYWORD SEARCH - TOP 5 RESULTS")
    print("=" * 60)

    keyword_results = keyword_search(
        query,
        documents,
        TOP_K
    )

    formatted_keyword_results = []

    for rank, (index, score) in enumerate(
        keyword_results,
        start=1
    ):

        doc = documents[index]

        result = {
            "rank": rank,
            "chunk_id": doc.metadata["chunk_id"],
            "page": doc.metadata["page"],
            "score": score,
            "text": doc.page_content
        }

        formatted_keyword_results.append(result)

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Page: {result['page']}")
        print(f"BM25 Score: {score:.4f}")
        print(result["text"][:300])

    # --------------------------------------------------------
    # SAVE ALL SEARCH RESULTS
    # --------------------------------------------------------

    search_rows = []

    for result in semantic_results:

        search_rows.append({
            "query": query,
            "search_type": "Semantic",
            "rank": result["rank"],
            "chunk_id": result["chunk_id"],
            "page": result["page"],
            "score": result["score"],
            "score_type": "Cosine similarity",
            "distance": result["distance"],
            "chunk_text": result["text"]
        })

    for result in formatted_keyword_results:

        search_rows.append({
            "query": query,
            "search_type": "Keyword",
            "rank": result["rank"],
            "chunk_id": result["chunk_id"],
            "page": result["page"],
            "score": result["score"],
            "score_type": "BM25 relevance score",
            "distance": None,
            "chunk_text": result["text"]
        })

    search_columns = [
        "query",
        "search_type",
        "rank",
        "chunk_id",
        "page",
        "score",
        "score_type",
        "distance",
        "chunk_text"
    ]

    append_csv(
        search_rows,
        SEARCH_CSV,
        search_columns
    )

    print("\nSearch results saved to search_results.csv")

    # --------------------------------------------------------
    # PREPARE CONTEXT FOR OLLAMA
    # --------------------------------------------------------

    retrieved_context = ""

    for result in semantic_results:

        retrieved_context += (
            f"\nPage: {result['page']}\n"
            f"Chunk ID: {result['chunk_id']}\n"
            f"{result['text']}\n"
        )

    # --------------------------------------------------------
    # GENERATE ANSWER USING OLLAMA
    # --------------------------------------------------------

    prompt = f"""
You are an expert PDF question-answering assistant.

Answer the question using ONLY the provided PDF context.

Instructions:
1. Understand the complete question.
2. Give a clear and accurate answer.
3. Explain concepts in simple language.
4. If asked for differences, use bullet points.
5. If the answer is unavailable in the context,
   say that it could not be found in the provided PDF.
6. Do not invent facts.
7. Give a detailed but relevant answer.

PDF CONTEXT:
{retrieved_context}

USER QUESTION:
{query}

ANSWER:
"""

    print("\n" + "=" * 60)
    print("GENERATING ANSWER")
    print("=" * 60)

    try:

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

    except Exception as error:

        print("Ollama error:", error)
        print("Check that Ollama is running and the model is installed.")
        continue

    print("\nANSWER")
    print("=" * 60)
    print(answer)

    # --------------------------------------------------------
    # SAVE ANSWER
    # --------------------------------------------------------

    append_csv(
        [{
            "query": query,
            "answer": answer,
            "retrieved_chunks": len(semantic_results)
        }],
        ANSWERS_CSV,
        ["query", "answer", "retrieved_chunks"]
    )

    print("\nAnswer saved to answers.csv")

# ============================================================
# END OF PROJECT
# ============================================================
