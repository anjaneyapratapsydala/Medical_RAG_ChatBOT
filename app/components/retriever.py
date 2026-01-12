from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger

logger = get_logger(__name__)

def create_retriever_qa_chain():
    logger.info("Loading vector store")
    db = load_vector_store()

    retriever = db.as_retriever(search_kwargs={"k": 1})
    llm = load_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a medical assistant. Answer ONLY using the provided context. "
         "If the answer is not in the context, say you don't know."),
        ("human", 
         "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("Successfully created Retrieval QA chain")
    return chain
