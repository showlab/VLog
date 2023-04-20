import os 
import pdb
import pickle
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ChatVectorDBChain
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings 

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the discussion is about the video content.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

qa_template = """You are an AI assistant designed for answering questions about a video.
You are given a document and a question, the document records what people see and hear from this video.
Try to connet these information and provide a conversational answer.
Question: {question}
=========
{context}
=========
"""
QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])
    

class LlmReasoner():
    def __init__(self, args):
        self.history = []
        self.gpt_version = args.gpt_version
        self.data_dir = args.data_dir
        self.tmp_dir = args.tmp_dir
        self.qa_chain = None
        self.vectorstore = None
        self.top_k = 3
        self.llm = OpenAI(temperature=0,  model_name=self.gpt_version)
         
    def exist_vectorstore(self, video_id):
        pkl_path = os.path.join(self.tmp_dir, f"{video_id}.pkl")
        log_path = os.path.join(self.data_dir, f"{video_id}.log")
        if os.path.exists(pkl_path) and os.path.exists(log_path):
            with open(pkl_path, 'rb') as file:
                self.vectorstore = pickle.load(file)
                
            self.qa_chain = ChatVectorDBChain.from_llm(
            self.llm,
            self.vectorstore,
            qa_prompt=QA_PROMPT,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            )
            self.qa_chain.top_k_docs_for_context = self.top_k
            return True
        return False
    
    def create_vectorstore(self, video_id):     
        pkl_path = os.path.join(self.tmp_dir, f"{video_id}.pkl")
        
        if not os.path.exists(pkl_path):
            loader = UnstructuredFileLoader(os.path.join(self.data_dir, f"{video_id}.log"))
            raw_documents = loader.load()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter()
            documents = text_splitter.split_documents(raw_documents)

            # Load Data to vectorstore
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(documents, embeddings)
        
            # Save vectorstore
            with open(pkl_path, "wb") as f:
                pickle.dump(vectorstore, f)
        
        
        with open(pkl_path, 'rb') as file:
            self.vectorstore = pickle.load(file)
        
        self.qa_chain = ChatVectorDBChain.from_llm(
            self.llm,
            self.vectorstore,
            qa_prompt=QA_PROMPT,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        )
        self.qa_chain.top_k_docs_for_context = self.top_k

        return 

    def __call__(self, question):
        print(f"Question: {question}")
        response = self.qa_chain({"question": question, "chat_history": self.history})["answer"]
        self.history.append((question, response))
        
        print(f"Assistant: {response}")
        print("\n")
        return response
    
    def clean_history(self):
        self.history = []
