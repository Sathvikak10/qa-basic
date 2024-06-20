from haystack import Finder
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.document_store.sql import SQLDocumentStore
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.retriever.sparse import TfidfRetriever
from haystack.utils import print_answers
from flask import Flask, request, render_template
#from flask_ngrok import run_with_ngrok
app = Flask(__name__)
app._static_folder = 'static'
#run_with_ngrok(app)

@app.route("/")
def home():
    return render_template("index2.html")
    

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    temp = get_response(userText)
    return temp



def tutorial3_basic_qa_pipeline_without_elasticsearch():    
    document_store = InMemoryDocumentStore()
    doc_dir = "data/articles"    
    dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)   
    document_store.write_documents(dicts)
    retriever = TfidfRetriever(document_store=document_store)
    reader = FARMReader(model_name_or_path='trained_model2')
    
    
    from haystack.pipeline import ExtractiveQAPipeline
    pipe = ExtractiveQAPipeline(reader, retriever)
   
    global get_response
    def get_response(query):
        flag = True
        print(query)      
    
        while(flag == True):        
            query = query.lower()
            if(query == 'quit'):
                flag = False
                break;
            if 'jump' in query:
                return "Here's a fun game where you can see it for yourself : https://cosmos-book.github.io/high-jump/index.html"
            
            prediction = pipe.run(query, top_k_retriever=1, top_k_reader=1)
            test = prediction
            context = test['answers'][0]['context']
            x = (context[test['answers'][0]['offset_start']:test['answers'][0]['offset_end']:])
            split_text = context.split('\n')
            data = [i for i in split_text if x in i]
            answer = 'Answer: ' + prediction['answers'][0]['answer'] + ' // '
            return answer + (data[0])
            #return print_answers(prediction, details="minimal")
    #prediction = pipe.run(query="what is a star?", top_k_retriever=1, top_k_reader=1)

    

    #print_answers(prediction, details="minimal")


if __name__ == "__main__":
    tutorial3_basic_qa_pipeline_without_elasticsearch()
    app.run()

