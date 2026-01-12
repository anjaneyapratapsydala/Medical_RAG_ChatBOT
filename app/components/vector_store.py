from langchain_community.vectorstores import FAISS
import os

from app.components.embeddings import get_embeddings as get_embeddings_model

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

def load_vector_store():
    try:
        embedding_model = get_embeddings_model()

        if os.path.exists(DB_FAISS_PATH):
            logger.info(f"Loading FAISS vector store from {DB_FAISS_PATH}")
            vector_store = FAISS.load_local(
                DB_FAISS_PATH, 
                embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info("Successfully loaded FAISS vector store.")
            return vector_store
        else:
            logger.warning(f"FAISS vector store path {DB_FAISS_PATH} does not exist.")
    
    except Exception as e:
        error_message = CustomException("Error occurred while loading FAISS vector store.", e)
        logger.error(str(error_message))


###creating new vectorstore function
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to create vector store.")
        
        logger.info("Creating FAISS vector store from text chunks.")

        embedding_model = get_embeddings_model()

        db = FAISS.from_documents(
            text_chunks,
            embedding_model
        )

        logger.info(f"Saving FAISS vector store to {DB_FAISS_PATH}")

        db.save_local(DB_FAISS_PATH)
        logger.info("Successfully saved FAISS vector store.")
        return db
    
    except Exception as e:
        error_message = CustomException("Failed to create or save FAISS vector store.", e)
        logger.error(str(error_message))
