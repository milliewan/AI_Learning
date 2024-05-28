import os
from os.path import exists
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Functions
def load(**kwargs):
    """Load data from website or file."""

    for format, value in kwargs.items():
        if format == "website":
            try: 
                loader = WebBaseLoader(value)
                data = loader.load_and_split()
            except Exception as e:
                print(e)
            else:
                return data
        elif format == "pdf":
            try: 
                print(value)
                loader = PyPDFLoader(value)
                data = loader.load_and_split()
            except Exception as e:
                print(e)
            else:
                return data

def embed(data):
    """Embed data into Chroma i.e. create a vector representation of a piece of text."""

    vectorstore = Chroma.from_documents(
                            documents=data,
                            embedding=OpenAIEmbeddings(),
                            )
    return vectorstore

def answer(llm, vectorstore):
    """Define a custom prompt.
    Define a RetrievalQA chain that uses the Chroma vector store as a retriever and passes in the question."""

    context = input("Please give me context: ")
    question = input("Please ask me a question: ")

    template = """Use the following pieces of context to answer the question. 
                If you don't know the answer, don't make it up.
                {context}
                Question: {question}
                """

    chain_prompt = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=vectorstore.as_retriever(),
                                        chain_type_kwargs={"prompt": chain_prompt} 
                                        )
    answer = qa_chain({"query": question})

    return answer["result"]

def chat(llm, url=None, file_path=None):
    """Chatbot."""
    try:
        if url!='':
            pages = load(website=url)
        elif file_path!='':
            pages = load(pdf=file_path)
        else:
            raise Exception
    except Exception as e:
        print("No data specified")
    else:
        vectorstore = embed(pages)
        answer = answer(llm, vectorstore)
        return answer

# Parameters
url = "https://www.bbcgoodfood.com/recipes/lemon-drizzle-cake"
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

print(chat(llm=llm, url=url))
