from flask import Flask

from langchain.text_splitter import RecursiveCharacterTextSplitter , CharacterTextSplitter
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document



import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain.memory import ConversationBufferMemory 
import time

from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings


from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


app=Flask(__name__)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key is None:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables")

def file_uploading(file_paths):
    
    loader = PyPDFLoader(file_paths)
    pages = loader.load() 
    return pages
    

def join_pages(file_paths):
    pages = file_uploading(file_paths)
    pdf_text = "\n".join([page.page_content for page in pages])
    return pdf_text

# def chunk_text():
#     chunk_size = 300
#     chunk_overlap = 30

def chunk_text(file_paths, chunk_size=300,chunk_overlap=30):
    pdf_text=join_pages(file_paths)

    r_splitter=RecursiveCharacterTextSplitter(
     chunk_size=chunk_size,
     separators=['\n','\n\n'," ",""],
     chunk_overlap=chunk_overlap
     )
    text_chunks = r_splitter.split_text(pdf_text)
    print(text_chunks[:3])
    return text_chunks


def create_vectorstore(embed_model,file_paths,doc):
    text_chunks=chunk_text(file_paths=file_paths)
    docs = [Document(page_content=chunk) for chunk in text_chunks]
    vector_store = FAISS.from_documents(docs, embed_model)
    vector_store.save_local(f"{file_paths}vectorstores")
    #vector_store.merge_from(doc)
    #vector_store.save_local("faiss_flaskcombined_index")
    #vs2.merge_from(vector_store)
    #vs2.save_local("faiss_flaskcombined_index")
    #uploads\Established_drugs_endometrial_cancerIMIM.pdfvectorstores.merge_from(uploads\Flagellum_specific_ATPaseMemBioMed.pdfvectorstores)
    combined_vs = FAISS.load_local(doc[0], embed_model, allow_dangerous_deserialization=True)
    for vs_dir in doc[1:]:
        vs = FAISS.load_local(vs_dir, embed_model, allow_dangerous_deserialization=True)
        combined_vs.merge_from(vs)
    combined_vs.save_local("faiss_flaskcombined_index")

def create_conversation_chain(embed_model):
    retrievers = FAISS.load_local("faiss_flaskcombined_index",embed_model,allow_dangerous_deserialization=True ).as_retriever()
    #retrievers=create_vectorstore()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retrievers,
        memory=memory,
    )
    return chain



def creating_responses(chain,query):
    #chain=create_conversation_chain()
    print("Welcome here! How can I assist you?")
    #query = request.get_json('question')
    print(query)
    response = chain.invoke(query)
    #print("\nBot:", response['answer'])
    return response['answer']

embed_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)

@app.route('/overall_process', methods=['POST'])
def overall_process():
    
    #embed_model=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    #embed_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-exp-03-07" , api_key=api_key)
    print(api_key)


    #create_vectorstore(embed_model,file_paths)
    query = request.json.get('question')
    chain=create_conversation_chain(embed_model)
    
    response = creating_responses(chain,query)
    return response  
	

# def upload_files():
#     UPLOAD_FOLDER = 'uploads'
#     app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#         if request.method == 'POST':
#             uploaded_files = request.files.getlist('files')
    
#             for file in uploaded_files:
#                 if file(file.filename):
#                     filename = secure_filename(file.filename)
#                     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return 'Files uploaded successfully'
#         else:
#             print("no file uploaded")


@app.route('/upload', methods=['POST'])
def upload_files():
    UPLOAD_FOLDER = 'uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    uploaded_files = request.files.getlist('files')
    print(uploaded_files)
    file_paths = [] 

    for file in uploaded_files:
        if file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            file.save(file_path)
            #print(f"Saved file to: {file_path}")
            file_paths.append(file_path)  
            #print("Absolute upload folder path:", os.path.abspath(UPLOAD_FOLDER))
    print(file_paths)



    #return 'Files uploaded and loaded successfully'
    doc = []
    for file_path in file_paths:
        doc.append(f"{file_path}vectorstores")
        create_vectorstore(embed_model,file_path,doc)
        

    return "file uploaded and embeddings generated for each file"

if __name__== "__main__":
    
    app.run()


"""

https://docs.google.com/spreadsheets/d/14EcXjkVB34zNRkDCfglYQeKXcObT5zVH/edit?usp=drive_link&ouid=114071573332580210084&rtpof=true&sd=true

{
    "assistantId": "4ccfad12-cfed-47f1-a670-de80759c8e46",
    "phoneNumberId": "2309fe55-5686-4152-9663-8edb71ea54f6",
    "customer": {
        "number": "+92{{$json['phone number']}}"
    }
"""




