from langchain_huggingface import HuggingFaceEndpoint
from app.config.config import groq_api_key, groq_api_key
from langchain_groq import ChatGroq

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(model_name: str = "llama-3.1-8b-instant", groq_api_key: str = groq_api_key):
    try:
        logger.info("Loading HuggingFace LLM model.")

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.3,
            max_tokens=512,
        )
        
        logger.info("Successfully loaded HuggingFace LLM model.")

        return llm
    
    except Exception as e:
        error_message = CustomException("Error occurred while loading HuggingFace LLM model.", e)
        logger.error(str(error_message))
        return None
