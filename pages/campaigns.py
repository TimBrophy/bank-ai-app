import os
import tqdm
import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch, helpers
from apikey import user, password, cloud_id

os.environ['elastic_user'] = user
os.environ['elastic_password'] = password
os.environ['elastic_cloud_id'] = cloud_id
es = Elasticsearch(
    cloud_id=os.environ['elastic_cloud_id'],
    basic_auth=(os.environ['elastic_user'], os.environ['elastic_password']))


# ---------------------------------------------
#           Existing campaigns
# ---------------------------------------------

