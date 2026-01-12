import os
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"Data path {DATA_PATH} does not exist.")
        
        logger.info(f"Loading PDF files from directory: {DATA_PATH}")
        
        loader = DirectoryLoader(path=DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)

        documents = loader.load()

        if not documents:
            logger.warning(f"No PDF documents found in directory: {DATA_PATH}")
        else:
            logger.info(f"successfully loaded {len(documents)} documents from {DATA_PATH}")

        return documents
    except Exception as e:
        error_message = CustomException("Error occurred while loading PDF files.", e)
        logger.error(str(error_message))
        return []


def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents provided for text chunking.")


        logger.info(f"Splitting {len(documents)} documents into text chunks.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,                
            )    
        
        text_chunks = text_splitter.split_documents(documents)
        logger.info(f"Successfully created {len(text_chunks)} text chunks.")
        return text_chunks
    
    except Exception as e:
         error_message = CustomException("Error occurred while creating text chunks.",e)
         logger.error(str(error_message))
         return []
    
