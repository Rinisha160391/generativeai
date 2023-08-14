import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with your PDF💬..💬...")
    pdf=st.file_uploader("Please upload your file here",type="pdf")
    
    if pdf is not None:
        #Load pdf document   
        if pdf is not None:
          pdf_reader = PdfReader(pdf)
          text = ""
          for page in pdf_reader.pages:
            text += page.extract_text()
        
        #breakdown the loaded text to smaller chunks in a way the semantically similar pieces are together
        #chunk size is mentioned to maintain the continuity and context
        char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000,
                                                 chunk_overlap=200,length_function=len)
        text_chunks = char_text_splitter.split_text(text)
        
        #Represent the textal information in the numerical vector format
        embeddings=OpenAIEmbeddings()
        docsearch=FAISS.from_texts(text_chunks,embeddings)
        
        #Chains allow us to combine multiple components together to create a single, coherent application.
        #for different tasks like qa, summarize,sequential we have different chains
        llm=OpenAI()
        
        #When using a language model, you typically provide a prompt to indicate what you want the model to talk about or respond to. The model then generates a coherent and contextually relevant response based on the given prompt.
        template="""If you don't know the answer just say exactly in the quotes "I'm sorry.. I don't know about that idiot".. don't make up the answer..Always say "Thanks for asking!!"""
        prompt=PromptTemplate.from_template(template)
        chain=load_qa_chain(llm,chain_type='stuff')
        
        query=st.text_input("Type your question:")
        
        if query!=None:
            #process the query and get the relevant embeddings from docstore using vectorstore(vectorstore used as relevant embedding finders)
            relevant_docs=docsearch.similarity_search(query)
            #Run the chain which returns the generated answer
            response=chain.run(input_documents=relevant_docs,question=query,prompt=prompt)
        
            st.write(response)
            #print(text_chunks)

if __name__=='__main__':
    main()
    
    