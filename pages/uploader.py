import os
import tqdm
import streamlit as st
from elasticsearch import Elasticsearch, helpers
from PyPDF2 import PdfReader
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
import math
import uuid
import re

os.environ['elastic_cloud_id'] = st.secrets['cloud_id']
os.environ['elastic_user'] = st.secrets['user']
os.environ['elastic_password'] = st.secrets['password']

es = Elasticsearch(
    cloud_id=os.environ['elastic_cloud_id'],
    basic_auth=(os.environ['elastic_user'], os.environ['elastic_password']))


def split_doc_sections(text, max_length=1024):
    sections = []
    current_section = ""
    current_length = 0
    sentences = sent_tokenize(text)
    for sentence in sentences:
        if current_length + len(sentence) <= max_length:
            current_section += sentence + ' '
            current_length += len(sentence)
        else:
            sections.append(current_section.strip())
            current_section = sentence + ' '
            current_length = len(sentence)
    if current_section:
        sections.append(current_section.strip())
    return sections



# ---------------------------------------------
#           PDF uploader
# ---------------------------------------------
st.title('Annual report uploader')
uploaded_file = st.file_uploader("Choose a file:")
if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    number_of_pages = len(reader.pages)
    st.write(f"number of pages: {number_of_pages}")
    report_name = st.text_input("What is the name of the annual report?")
    publish_date = st.date_input("What is the date that this report was published?")
    if report_name is not None:
        import_text = st.button("Import (reliable)?")
        import_doc = st.button("Import (experimental)?")
        if import_doc:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            nltk.download('punkt')
            # Iterate through each page of the PDF
            page_num = 0
            with st.status("Uploading document") as status:
                while page_num <= len(reader.pages):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    # Split the page into sections
                    sections = split_doc_sections(page_text)
                    # Output the sections
                    for i, section in enumerate(sections):
                        trimmed_text = section.strip()
                        trimmed_text = re.sub(r'\s+', ' ', trimmed_text)
                        doc_id = uuid.uuid4()
                        doc = {
                            "report_name": report_name,
                            "text": trimmed_text,
                            "publish_date": publish_date,
                            "page": page_num,
                            "section": i + 1,
                            "_extract_binary_content": True,
                            "_reduce_whitespace": True,
                            "_run_ml_inference": True
                        }
                        response = es.index(index='search-annual-reports', id=doc_id, document=doc, pipeline="search-annual-reports")
                        st.write(f"Page: {page_num}, Section: {i + 1}")
                    page_num = page_num + 1
            status.update(label="Upload complete!", state="complete")
        if import_text:
            counter = 0
            with st.status("Uploading document") as status:
                while counter < number_of_pages:
                    selected_page = counter
                    if selected_page is not None:
                        page = reader.pages[selected_page]
                        text = page.extract_text()
                        words = text.split()
                        sentences = text.split(". ")
                        total_words = len(words)
                        doc_sections = math.ceil(total_words / 256)
                        words_per_section = total_words // doc_sections

                        sections = []
                        start_index = 0

                        for _ in range(doc_sections - 1):
                            end_index = start_index + words_per_section
                            section = " ".join(words[start_index:end_index])
                            sections.append(section)
                            start_index = end_index

                        final_section = " ".join(words[start_index:])
                        sections.append(final_section)
                        for i, section in enumerate(sections):
                            st.write(f"Page number: {selected_page + 1}")
                            # st.write(text)
                            doc_id = uuid.uuid4()
                            doc = {
                                "report_name": report_name,
                                "text": section,
                                "publish_date": publish_date,
                                "page": selected_page + 1,
                                "_extract_binary_content": True,
                                "_reduce_whitespace": True,
                                "_run_ml_inference": True
                            }
                            response = es.index(index='search-annual-reports', id=doc_id, document=doc,
                                                pipeline="search-annual-reports")
                    counter = counter + 1
            status.update(label="Upload complete!", state="complete")
