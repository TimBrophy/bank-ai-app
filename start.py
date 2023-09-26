import pandas as pd
import streamlit as st
import os
import openai
from elasticsearch import Elasticsearch
from apikey import user, password, cloud_id, openai_api_key, openai_api_type, openai_api_base, openai_api_version
from langchain.embeddings import ElasticsearchEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import tiktoken
import json
from datetime import datetime, timedelta


#------------------------------------------
#        connect to elasticsearch
#------------------------------------------

os.environ['openai_api_base']=st.secrets['openai_api_base']
os.environ['openai_api_key']=st.secrets['openai_api_key']
os.environ['openai_api_version']=st.secrets['openai_api_version']
os.environ['elastic_cloud_id']=st.secrets['cloud_id']
os.environ['elastic_user']=st.secrets['user']
os.environ['elastic_password']=st.secrets['password']

BASE_URL = openai_api_base
API_KEY = openai_api_key
DEPLOYMENT_NAME = "timb-fsi-demo"
chat_model = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version=openai_api_version,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
    temperature=0.1
)

es = Elasticsearch(
    cloud_id=os.environ['elastic_cloud_id'],
    basic_auth=(os.environ['elastic_user'], os.environ['elastic_password'])
)

# Instantiate ElasticsearchEmbeddings using credentials
model_id = ".elser_model_1"
embeddings = ElasticsearchEmbeddings.from_es_connection(model_id, es)



def customer_support_search_operation(index, question):
    expansion_query = {
        "bool": {
            "should": [
                {
                    "text_expansion": {
                        "ml.inference.body_content_expanded.predicted_value": {
                            "model_id": model_id,
                            "model_text": question
                        }
                    }
                },
                {
                    "match": {
                        "body_content": question
                    }
                }
            ]
        }
    }
    field_list = ['title', 'body_content', '_score']
    results = es.search(index=index, query=expansion_query, size=20, fields=field_list)    
    response_data = [{"_score": hit["_score"], **hit["_source"]} for hit in results["hits"]["hits"]]
    documents = []
    # Check if there are hits
    if "hits" in results and "total" in results["hits"]:
        total_hits = results["hits"]["total"]

        # Check if there are any hits with a value greater than 0
        if isinstance(total_hits, dict) and "value" in total_hits and total_hits["value"] > 0:
            for hit in response_data:
                if hit['_score'] > 0:
                    doc_data = {field: hit[field] for field in field_list if field in hit}
                    documents.append(doc_data)
    return documents

def transaction_search_operation(index, question, days):
    set_range_date = datetime.now() - timedelta(days=days)
    expansion_query = {
        "bool": {
            "should": [
                {
                    "text_expansion": {
                        "ml.inference.description_expanded.predicted_value": {
                            "model_id": model_id,
                            "model_text": question
                        }
                    }
                },
                {
                    "match": {
                        "description": question
                    }
                }
            ], 
            "filter": [
                {
                    "range": {
                        "transaction_date": {
                            "gte": set_range_date
                        }
                    }
                }
            ]
        }
    }

    field_list = ['transaction_date', 'account_number', 'balance', 'description', 'transaction_type', 'value', 'entity', '_score']
    results = es.search(index=index, query=expansion_query, size=100, fields=field_list)    
    response_data = [{"_score": hit["_score"], **hit["_source"]} for hit in results["hits"]["hits"]]
    documents = []
    # Check if there are hits
    if "hits" in results and "total" in results["hits"]:
        total_hits = results["hits"]["total"]

        # Check if there are any hits with a value greater than 0
        if isinstance(total_hits, dict) and "value" in total_hits and total_hits["value"] > 0:
            for hit in response_data:
                if hit['_score'] > 0:
                    doc_data = {field: hit[field] for field in field_list if field in hit}
                    documents.append(doc_data)
    return documents

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return ' '.join(tokens[:max_tokens])

def set_assistant_type():
    if assistant_type == 'Transaction analyser':
        st.session_state.assistant = 'Transaction analyser'
    return assistant_type

#------------------------------------------------
#        start with the form and control flow
#------------------------------------------------


st.title('Conversational search')
with st.sidebar:
    assistant_type = st.selectbox("Which feature do you want to use?", 
        ('Transaction analyser', 'Customer support', 'Report analyser'), key='assistant_type', on_change=set_assistant_type)

with st.form("search-form"):
    question = st.text_input("Go ahead and ask your question:", placeholder="Please help me understand what I spend my money on...")
    if st.session_state.assistant_type == 'Transaction analyser':
        days = st.slider('How may days transactions should I use?', 1, 180, 90)
    submitted = st.form_submit_button("Submit")

#-----------------------------------------------------------
#        Search for context and interact with the LLM
#-----------------------------------------------------------

    if submitted:
        # st.write(st.session_state.assistant_type)
        st.session_state.question = question
        if st.session_state.assistant_type == "Transaction analyser":         
            # run a transaction search
            index = "search-transactions"
            results = transaction_search_operation(index, question, days)
            string_results = json.dumps(results)
            df_results = pd.DataFrame(results)

            
            # interact with the LLM
            augmented_prompt = f"""Using the contexts below, answer the query.
            Contexts: {string_results}

            Query: {question}"""

            messages = [
                SystemMessage(content="You are a helpful financial analyst using transaction search results to give advice to customers. If you can asnwer a question, attempt to answer it fully."),
                # HumanMessage(content="Hi AI, how are you today?"),
                # AIMessage(content="I am great thank you. How can I help you today?"),
                HumanMessage(content=augmented_prompt)
                ]
        elif st.session_state.assistant_type == 'Customer support':
            index = "search-customer-support"
            results = customer_support_search_operation(index, question)
            string_results = json.dumps(results)
            df_results = pd.DataFrame(results)
            string_results = truncate_text(string_results, 12000)
            # interact with the LLM
            augmented_prompt = f"""Using only the context below, answer the query.
            Context: {string_results}

            Query: {question}"""

            messages = [
                SystemMessage(content="You are a helpful customer support agent that answers questions based only on the context provided. When you respond, please cite your source."),
                # HumanMessage(content="Hi AI, how are you today?"),
                # AIMessage(content="I am very good. How may I help you?"),
                HumanMessage(content=augmented_prompt)
                ]
                        
        st.subheader('Virtual assistant:')
        chat_bot = st.chat_message("ai assistant", avatar="🤖")
        # st.write(num_tokens_from_string(string_results, "cl100k_base"))
        with st.status("Contacting Skynet...") as status:
            chat_bot.info(chat_model(messages).content)
            status.update(label="AI response complete!", state="complete")

        # handle any context data that we want to represent
        if st.session_state.assistant_type == 'Transaction analyser':
            st.subheader('Transactions:')
        
        st.dataframe(df_results)