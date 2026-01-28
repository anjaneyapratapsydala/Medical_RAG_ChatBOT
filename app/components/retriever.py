from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import BaseOutputParser
import re
from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger

logger = get_logger(__name__)

class CleanTextParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Remove all HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Normalize excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_retriever_qa_chain():
    logger.info("Loading vector store")

    db = load_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": 1})

    llm = load_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
"You are a medical assistant. Use ONLY the provided context to answer.\n"
"Respond in PURE Markdown.\n"
"STRICT RULES:\n"
"- Do NOT use HTML tags of any kind\n"
"- Do NOT use <br>, <p>, <div>, or similar\n"
"- Use blank lines for spacing\n"
"- Use Markdown headings (##), bullet points (-), and paragraphs only\n"
"If the answer is not in the context, say: I don't know."
),
        ("human",
         "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    chain = (
        {
            "context": retriever|format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | CleanTextParser()
       
    )

    logger.info("Successfully created Retrieval QA chain")
    return chain
