
# ------------------------------------------
#       import all dependencies
# ------------------------------------------

import os
import tqdm
import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch, helpers

os.environ['elastic_cloud_id'] = st.secrets['cloud_id']
os.environ['elastic_user'] = st.secrets['user']
os.environ['elastic_password'] = st.secrets['password']
es = Elasticsearch(
    cloud_id=os.environ['elastic_cloud_id'],
    http_auth=(os.environ['elastic_user'], os.environ['elastic_password'])
)

# ------------------------------------------
#       define all functions needed
# ------------------------------------------

# function to delete Elasticsearch index
def delete_by_query(index_name):
    query_string = {"match_all": {}}
    response = es.delete_by_query(index=index_name, query=query_string)
    return response


# define the retailers
supermarket_list = ["7Eleven", "Ahold Delhaize", "Aldi", "Coop", "Lidl", "SPAR", "Tesco", "Woolworths"]
clothing_retailer_list = ["H&M", "Zalando", "Primark", "LVMH", "Asos", "JD Sports", "Zara"]
online_retailer_list = ["Apple.com", "Amazon.com", "Bol.com", "Takealot.com"]
subscriptions_list = ["Spotify", "Netflix", "HBOMax", "Apple Music"]


# create an overriding entity dictionary to work with
entity_dict = {
    "supermarkets": supermarket_list,
    "clothing_retailers": clothing_retailer_list,
    "online_retailers": online_retailer_list,
    "subscriptions": subscriptions_list
}

# define the users' bank accounts and opening balance
account_list = [
    {"number": "ES0912345678", "balance": random.randint(1500, 10000)}, 
    {"number": "ES0287654321", "balance": random.randint(1500, 10000)}
]

# choose the account to log the transaction against
def choose_account():
    account = random.choice(account_list)
    return account["number"]

# calculate the current balance and then update the remaining balance by offsetting the value of the transaction
def calculate_balance(account_number, value, operation):
    for account in account_list:
        if account["number"] == account_number:
            current_balance = account["balance"]
            if operation == "subtract":
                new_balance = current_balance - value
            else: 
                new_balance = current_balance + value
            account["balance"] = new_balance
    return new_balance

# create a random date in the timerange
def create_random_date(total_days):
    current_date = datetime.now()
    start_date = current_date - timedelta(total_days)
    random_days = random.randint(0, total_days)
    random_date = start_date + timedelta(days=random_days)
    random_date_str = random_date.strftime("%Y-%m-%d")
    return random_date_str

# get a random retailer for the transaction
def get_random_entity():
    category_list = [supermarket_list, clothing_retailer_list, online_retailer_list, subscriptions_list]
    category = random.choice(category_list)
    entity = random.choice(category)
    return entity

# build the transaction description message
def generate_description(entity, value, transaction_date):
    current_category = ""
    for category, entities in entity_dict.items():
        for e in entities:
            if e == entity:
                current_category = category
    if current_category == "supermarkets":
        description = "Purchase at {} supermarket, for €{} on {}".format(entity, value, transaction_date)
    elif current_category == "clothing_retailers":
        description = "Purchase at {} clothing, for €{} on {}".format(entity, value, transaction_date)
    elif current_category == "online_retailers":
        description = "Payment to {} online shopping, for €{} on {}".format(entity, value, transaction_date)
    elif current_category == "subscriptions":
        description = "Payment for {} subscription, for €{} on {}".format(entity, value, transaction_date)
    else:
        description = "Deposit from ACME corp, of {} on {}".format(value, transaction_date)

    return description

# choose a transaction type randomly

def get_random_transaction_type():
    transaction_type_list = ["credit card", "debit card", "deposit"]
    transaction_type = random.choices(transaction_type_list, weights=(10, 10, 1), k=1)
    transaction_type = ' '.join(transaction_type)
    return transaction_type

# ------------------------------------------
#       this is the logic block
# ------------------------------------------

st.title('Data generation for banking demo')
columns = ['transaction_date', 'value','balance','account_number', 'description', 'entity', 'transaction_type']

with st.form("setup_form"):
    number_of_months = st.number_input('Enter the number of months to generate data for:', min_value=1, max_value=10, value=3,step=1)
    st.text("Provide the range of transactions per day:")
    start_int = st.number_input('From:', min_value=1, max_value=10, value=3,step=1)
    end_int = st.number_input('To:', min_value=1, max_value=10, value=3,step=1)
    submit = st.form_submit_button('Generate data')

if submit:
    total_days = number_of_months*30
    df = pd.DataFrame(columns=columns)

    counter = 0
    while counter <= total_days:
        transaction_per_day_count = random.uniform(start_int, end_int)
        transaction_count = 0
        while transaction_count <= transaction_per_day_count:
            transaction_type = get_random_transaction_type()
            if transaction_type != "deposit":
                entity = get_random_entity()
                value = random.randint(0, 150)
                account_number = choose_account()
                transaction_date = create_random_date(total_days - counter)
                description = generate_description(entity, value, transaction_date)
                remaining_balance = calculate_balance(account_number, value, "subtract")
                new_row = {
                    'transaction_date': transaction_date,
                    'value': value,
                    'balance': remaining_balance,
                    'account_number': account_number,
                    'description': description,
                    'entity': entity,
                    'transaction_type': transaction_type
                }
            else:
                entity = "ACME corp"
                value = random.randint(1500, 10000)
                account_number = choose_account()
                transaction_date = create_random_date(total_days - counter)
                description = generate_description(entity, value, transaction_date)
                remaining_balance = calculate_balance(account_number, value, "add")
                new_row = {
                    'transaction_date': transaction_date,
                    'value': value,
                    'balance': remaining_balance,
                    'account_number': account_number,
                    'description': description,
                    'entity': entity,
                    'transaction_type': transaction_type
                }
            df.loc[len(df)] = new_row    
            transaction_count = transaction_count + 1
        counter = counter + 1
        
    
    st.dataframe(df, use_container_width=True)
    index_name = "search-transactions"
    # clear any existing data
    delete_response = delete_by_query(index_name)
    # convert dataframe to dict
    data = df.to_dict(orient='records')

    actions = [
        {
            '_index': index_name,
            '_id': i,
            '_source': doc
        }
        for i, doc in enumerate(data)
    ]
    number_of_docs = len(actions)
    progress = tqdm.tqdm(unit="docs", total=number_of_docs)
    successes = 0
    for ok, action in helpers.streaming_bulk(
        client=es, index=index_name, actions=actions, initial_backoff=5, max_backoff=30, chunk_size=50,
    ):
        progress.update(1)
        successes += ok
    
    st.balloons()
    st.write("Indexed %d/%d documents" % (successes, number_of_docs))
    