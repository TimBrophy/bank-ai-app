import math

import pandas as pd
import streamlit as st
import os
import openai
from elasticsearch import Elasticsearch
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
from PIL import Image

# ------------------------------------------
#        connect to elasticsearch
# ------------------------------------------

os.environ['openai_api_base'] = st.secrets['openai_api_base']
os.environ['openai_api_key'] = st.secrets['openai_api_key']
os.environ['openai_api_version'] = st.secrets['openai_api_version']
os.environ['elastic_cloud_id'] = st.secrets['cloud_id']
os.environ['elastic_user'] = st.secrets['user']
os.environ['elastic_password'] = st.secrets['password']

BASE_URL = os.environ['openai_api_base']
API_KEY = os.environ['openai_api_key']
DEPLOYMENT_NAME = "timb-fsi-demo"
chat_model = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version=os.environ['openai_api_version'],
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


def report_analyser_search_operation(index, question, report_name):
    expansion_query = {
        "bool": {
            "should": [
                {
                    "text_expansion": {
                        "ml.inference.text_expanded.predicted_value": {
                            "model_id": model_id,
                            "model_text": question
                        }
                    }
                },
                {
                    "match": {
                        "text": question
                    }
                }
            ],
            "filter": {
                "term": {
                    "report_name": report_name
                }
            }
        }
    }
    field_list = ['page', 'text', '_score']
    results = es.search(index=index, query=expansion_query, size=20, fields=field_list)
    response_data = [{"_score": hit["_score"], **hit["_source"]} for hit in results["hits"]["hits"]]
    documents = []
    # Check if there are hits
    if "hits" in results and "total" in results["hits"]:
        total_hits = results["hits"]["total"]
        # Check if there are any hits with a value greater than 0
        if isinstance(total_hits, dict) and "value" in total_hits and total_hits["value"] > 0:
            for hit in response_data:
                if hit['_score'] > 5:
                    doc_data = {field: hit[field] for field in field_list if field in hit}
                    documents.append(doc_data)
    return documents


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

    field_list = ['transaction_date', 'account_number', 'balance', 'description', 'transaction_type', 'value', 'entity',
                  '_score']
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
    return ' '.join(tokens[:max_tokens])


def set_assistant_type():
    if assistant_type == 'Transaction analyser':
        st.session_state.assistant = 'Transaction analyser'
    return assistant_type


def get_reports(index):
    aggregation_query = {
        "size": 0,
        "query": {
            "match_all": {
            }
        },
        "aggs": {
            "reports": {
                "terms": {
                    "field": "report_name",
                    "size": 1000
                }
            }
        }
    }
    reports = es.search(index=index, body=aggregation_query)
    buckets = reports['aggregations']['reports']['buckets']
    report_list = []
    for bucket in buckets:
        key = bucket['key']
        report_list.append(key)
    return report_list


def get_campaigns(index, text):
    expansion_query = {
        "bool": {
            "should": [
                {
                    "text_expansion": {
                        "ml.inference.campaign_description_expanded.predicted_value": {
                            "model_id": model_id,
                            "model_text": text
                        }
                    },
                    "text_expansion": {
                        "ml.inference.campaign_name_expanded.predicted_value": {
                            "model_id": model_id,
                            "model_text": text
                        }
                    }
                },
                {
                    "match": {
                        "campaign_description": {
                            "query": text,
                            "boost": 1
                        }
                    },
                    "match": {
                        "campaign_name": {
                            "query": text,
                            "boost": 1
                        }
                    }
                }
            ]
        }
    }

    field_list = ['campaign_name', 'campaign_description', '_score']
    campaign_results = es.search(index=index, query=expansion_query, size=1, fields=field_list)
    response_data = [{"_score": hit["_score"], **hit["_source"]} for hit in campaign_results["hits"]["hits"]]
    documents = []
    # Check if there are hits
    if "hits" in campaign_results and "total" in campaign_results["hits"]:
        total_hits = campaign_results["hits"]["total"]

        # Check if there are any hits with a value greater than 0
        if isinstance(total_hits, dict) and "value" in total_hits and total_hits["value"] > 0:
            for hit in response_data:
                if hit['_score'] > 5:
                    doc_data = {field: hit[field] for field in field_list if field in hit}
                    documents.append(doc_data)
    return documents


def calculate_cost(message):
    cost_per_1k_prompt = 0.03
    cost_per_1k_message = 0.06
    message_token_count = num_tokens_from_string(message, "cl100k_base")
    billable_message_tokens = message_token_count / 1000
    rounded_up_message_tokens = math.ceil(billable_message_tokens)
    message_cost = rounded_up_message_tokens * cost_per_1k_message
    return message_cost


# ------------------------------------------------
#        start with the form and control flow
# ------------------------------------------------

image = Image.open('images/logo.png')

st.image(image, width=200)

if "chat_responses" not in st.session_state:
    st.session_state.chat_responses = []

st.title('Financial services assistant')
assistant_type = st.selectbox("Which feature do you want to use?",
                              ('Transaction analyser', 'Customer support', 'Report analyser'), key='assistant_type',
                              on_change=set_assistant_type)

with st.form("search-form"):
    st.session_state.question = st.text_input("Go ahead and ask your question:",
                                              placeholder="Please help me understand what I spend my money on...")
    if st.session_state.assistant_type == 'Transaction analyser':
        days = st.slider('Number of days', 1, 180, 90)
        opt_in = st.toggle('Opt in to see special offers')
    elif st.session_state.assistant_type == 'Report analyser':
        report_name = st.selectbox('Which report do you want to analyse?', (get_reports('search-annual-reports')))

    submitted = st.form_submit_button("Submit")

# -----------------------------------------------------------
#        Search for context and interact with the LLM
# -----------------------------------------------------------

if submitted:
    # st.write(st.session_state.assistant_type)
    if st.session_state.assistant_type == "Transaction analyser":
        # run a transaction search
        st.session_state.index = "search-transactions"
        results = transaction_search_operation(st.session_state.index, st.session_state.question, days)
        string_results = json.dumps(results)
        df_results = pd.DataFrame(results)

        # interact with the LLM
        augmented_prompt = f"""Using only the contexts below, answer the query.
        Contexts: {string_results}

        Query: {st.session_state.question}"""
        messages = [
            SystemMessage(
                content="You are a helpful financial analyst using transaction search results to give advice to customers. "
                        "If you can asnwer a question, attempt to answer it fully. Assume the context provided provides an accurate response to the query."),
            # HumanMessage(content="Hi AI, how are you today?"),
            # AIMessage(content="I am great thank you. How can I help you today?"),
            HumanMessage(content=augmented_prompt)
        ]

    elif st.session_state.assistant_type == 'Customer support':
        st.session_state.index = "search-customer-support"
        results = customer_support_search_operation(st.session_state.index, st.session_state.question)
        string_results = json.dumps(results)
        df_results = pd.DataFrame(results)
        string_results = truncate_text(string_results, 10000)
        # interact with the LLM
        augmented_prompt = f"""Using only the context below, answer the query.
        Context: {string_results}

        Query: {st.session_state.question}"""
        messages = [
            SystemMessage(
                content="You are a helpful customer support agent that answers questions based only on the context provided. "
                        "When you respond, please cite your source."),
            # HumanMessage(content="Hi AI, how are you today?"),
            # AIMessage(content="I am very good. How may I help you?"),
            HumanMessage(content=augmented_prompt)
        ]
    elif st.session_state.assistant_type == 'Report analyser':
        st.session_state.index = "search-annual-reports"
        results = report_analyser_search_operation(st.session_state.index, st.session_state.question, report_name)
        string_results = json.dumps(results)
        df_results = pd.DataFrame(results)
        reduced_string_results = truncate_text(string_results, 8000)
        # interact with the LLM
        augmented_prompt = f"""Using only the context below, answer the query.
        Context: {reduced_string_results}

        Query: {st.session_state.question}"""
        messages = [
            SystemMessage(
                content="You are a helpful analyst that answers questions based only on the context provided. "
                        "When you respond, please cite your source and where possible, always summarise your answers."),
            # HumanMessage(content="Hi AI, how are you today?"),
            # AIMessage(content="I am very good. How may I help you?"),
            HumanMessage(content=augmented_prompt)
        ]
    st.subheader('Virtual assistant:')
    chat_bot = st.chat_message("ai assistant", avatar="🤖")
    # st.write(num_tokens_from_string(string_results, "cl100k_base"))
    with st.status("Processing the data...") as status:
        result_len = len(df_results)
        status.update(label=f'Retrieved {result_len} results from Elasticsearch', state="running")
        current_chat_message = chat_model(messages).content
        st.session_state.chat_responses = current_chat_message
        status.update(label=f'Reaching out to LLM', state="running")
        chat_bot.info(st.session_state.chat_responses)
        cost_data = calculate_cost(st.session_state.chat_responses)
        st.write(f"Calculating response cost: ${cost_data}")
        status.update(label="AI response complete!", state="complete")




    # handle any context data that we want to represent
    if st.session_state.assistant_type == 'Transaction analyser':
        campaigns = get_campaigns('search-campaigns', string_results)
        if len(campaigns):
            if opt_in:
                campaign_string_results = json.dumps(campaigns)
                df_campaigns = pd.DataFrame(campaigns)
                # interact with the LLM
                augmented_prompt = f"""Using the contexts below, explain to the customer about our special offers.
                Contexts: {campaign_string_results}"""
                messages = [
                    SystemMessage(
                        content="You are a helpful customer support representative that can enthusiastically explain how our special offers can help them. "
                                "Do not simply repeat the special offer text, rephrase it to be positive and rewarding."
                                "Respond in no more than 80 words."),
                    HumanMessage(content=augmented_prompt)
                ]
                campaign_chat_bot = st.chat_message("ai assistant", avatar="🤖")
                with st.status("Contacting our experts...") as status:
                    campaign_chat_bot.info(chat_model(messages).content)
                    status.update(label="AI response complete!", state="complete")
                st.subheader('Campaign data:')
                st.dataframe(df_campaigns)
        st.subheader('Transactions:')
    st.dataframe(df_results)
