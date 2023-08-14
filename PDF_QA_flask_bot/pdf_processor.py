from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

#def process_query(pdf_path,query)
#load openapi key
load_dotenv()

def pdf_processor(filename,query):
    #Load pdf document
    
    loader=PyMuPDFLoader(filename)
    text=loader.load()
    
    #breakdown the loaded text to smaller chunks in a way the semantically similar pieces are together
    #chunk size is mentioned to maintain the continuity and context
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=10)
    text_chunks=text_splitter.split_documents(text)
    #print(text_chunks)
    
    
    #Represent the textal information in the numerical vector format
    embeddings=OpenAIEmbeddings()
    docsearch=FAISS.from_documents(text_chunks,embeddings)
    
    #query="what was the Giraff's age?"
    #process the query and get the relevant embeddings from docstore using vectorstore(vectorstore used as relevant embedding finders)
    #relevant_docs=docsearch.similarity_search(query)
    
    #Chains allow us to combine multiple components together to create a single, coherent application.
    #for different tasks like qa, summarize,sequential we have different chains
    
    retriever = docsearch.as_retriever()
    llm = ChatOpenAI()
    #template="""If you don't know the answer just say exactly in the quotes "I'm sorry.. I don't know about that idiot".. don't make up the answer..Always say "Thanks for asking!!"""
    #prompt=PromptTemplate.from_template(template)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    llm_response = qa(query)
    return llm_response["result"]
    
    ##initialize the llm model and create the respective chain to generate answer based on the retreived docs
    #llm=OpenAI()
    #
    ##When using a language model, you typically provide a prompt to indicate what you want the model to talk about or respond to. The model then generates a coherent and contextually relevant response based on the given prompt.
    #template="""If you don't know the answer just say exactly in the quotes "I'm sorry.. I don't know about that idiot".. don't make up the answer..Always say "Thanks for asking!!"""
    #prompt=PromptTemplate.from_template(template)
    #chain=load_qa_chain(llm,chain_type='stuff')
    #
    ##Run the chain which returns the generated answer
    #response=chain.run(input_documents=relevant_docs,question=query,prompt=prompt)
    #print(response)









