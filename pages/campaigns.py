import streamlit as st
import os
from elasticsearch import Elasticsearch, helpers
import uuid

os.environ['elastic_cloud_id'] = st.secrets['cloud_id']
os.environ['elastic_user'] = st.secrets['user']
os.environ['elastic_password'] = st.secrets['password']

es = Elasticsearch(
    cloud_id=os.environ['elastic_cloud_id'],
    basic_auth=(os.environ['elastic_user'], os.environ['elastic_password']))

def get_campaigns(index):
    query = {
        "query": {
            "match_all": {}
        }
    }
    field_list = ['campaign_name', 'campaign_description', '_score']
    campaigns = es.search(index=index, body=query)
    response_data = [{"_score": hit["_score"], **hit["_source"]} for hit in campaigns["hits"]["hits"]]
    documents = []
    # Check if there are hits
    if "hits" in campaigns and "total" in campaigns["hits"]:
        total_hits = campaigns["hits"]["total"]

        # Check if there are any hits with a value greater than 0
        if isinstance(total_hits, dict) and "value" in total_hits and total_hits["value"] > 0:
            for hit in response_data:
                if hit['_score'] > 0:
                    doc_data = {field: hit[field] for field in field_list if field in hit}
                    documents.append(doc_data)
    return documents


#------------------------------------------------
#        create new campaigns
#------------------------------------------------


st.title('Add a new campaign')
with st.form("campaign-form"):
    campaign_name = st.text_input('Campaign name')
    campaign_description = st.text_area('Campaign text')
    submit = st.form_submit_button('Add campaign')
    if submit:
        doc_id = uuid.uuid4()
        doc = {
            "campaign_name": campaign_name,
            "campaign_description": campaign_description,
            "_run_ml_inference": True
        }
        response = es.index(index='search-campaigns', id=doc_id, document=doc, pipeline="search-campaigns")

campaign_list = get_campaigns('search-campaigns')
st.dataframe(campaign_list)
