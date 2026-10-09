import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # WARN: must import before LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS

# ── 1. Unique Prompt names on Hub ────────────────────────────────────────
# Replace with a name that is unique in your LangSmith workspace.
PROMPT_V1_NAME = "duongducvuong-rag-prompt-v1"
PROMPT_V2_NAME = "duongducvuong-rag-prompt-v2"

# ── 2. Define the two Prompt Templates ───────────────────────────────────
# SYSTEM_V1 – concise style (2‑4 sentences) and must contain {context}
SYSTEM_V1 = (
    "Bạn là trợ lý AI hữu ích. Chỉ dùng thông tin trong context để trả lời câu hỏi. "
    "Giữ câu trả lời ngắn gọn, trong 2‑4 câu.\n\nContext:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])

# SYSTEM_V2 – expert style (3‑5 sentences) and must contain {context}
SYSTEM_V2 = (
    "Bạn là chuyên gia AI. Đọc kỹ context, xác định các facts liên quan, và "
    "trình bày câu trả lời một cách rõ ràng, có cấu trúc, trong 3‑5 câu.\n\nContext:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])

# ── 3. Push Prompts to Prompt Hub ────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """Upload both prompt templates to LangSmith Prompt Hub.

    Any error (e.g., network issues, duplicate name) is caught and reported.
    """
    # V1
    try:
        url = client.push_prompt(
            PROMPT_V1_NAME,
            PROMPT_V1,
            description="V1 – concise style",
        )
        print(f"[OK] Pushed V1 → {url}")
    except Exception as e:
        print(f"[WARN] V1 error: {e}")

    # V2
    try:
        url = client.push_prompt(
            PROMPT_V2_NAME,
            PROMPT_V2,
            description="V2 – expert style",
        )
        print(f"[OK] Pushed V2 → {url}")
    except Exception as e:
        print(f"[WARN] V2 error: {e}")

# ── 4. Pull Prompts from Prompt Hub ───────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """Retrieve the two prompts from LangSmith Prompt Hub.

    If retrieval fails, fall back to the locally defined prompt templates.
    Returns a dict mapping prompt name → ChatPromptTemplate.
    """
    prompts = {}
    # V1
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"[OK] Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"[INFO] Using local fallback for '{PROMPT_V1_NAME}'")
    # V2
    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"[OK] Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"[INFO] Using local fallback for '{PROMPT_V2_NAME}'")
    return prompts

# ── 5. Deterministic A/B Routing based on MD5 hash ────────────────────────
def get_prompt_version(request_id: str) -> str:
    """Return the prompt name (V1 or V2) based on a deterministic hash.

    Even hash → V1, odd hash → V2.
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME

# ── 6. Traced A/B Query ───────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """Run a RAG chain using the selected prompt version.

    Returns a dictionary containing the original question, the generated answer,
    and the version identifier ("v1" or "v2").
    """
    # Retrieve top‑k documents (k is defined in the retriever)
    docs = retriever.invoke(question)
    # Concatenate page_content of each document
    context = "\n\n".join(doc.page_content for doc in docs)
    # Build and run the chain
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}

# ── 7. Setup Vectorstore (reuse logic from Task 1) ────────────────────────
def setup_vectorstore():
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text)
    return build_vectorstore(chunks, embeddings)

# ── 8. Main execution flow ───────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # Initialise LangSmith client (API key is read from .env via config)
    client = Client(api_key=config.LANGSMITH_API_KEY)

    # Push prompts to the hub (first run will create them, subsequent runs update)
    push_prompts_to_hub(client)

    # Pull prompts back – ensures we are using the Hub version
    prompts = pull_prompts_from_hub(client)

    # Build vectorstore, retriever and LLM
    vectorstore = setup_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    # A/B routing over the sample questions
    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]
        result = ask_ab(retriever, llm, prompt, question, version_tag)
        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}…")

    print(f"\n[INFO] Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("[OK] Step 2 completed! Check Prompt Hub and traces on LangSmith.")

if __name__ == "__main__":
    main()
