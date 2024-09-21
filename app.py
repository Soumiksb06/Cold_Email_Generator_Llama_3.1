!sudo apt-get update
!sudo apt-get install sqlite3 libsqlite3-dev

import streamlit as st
from langchain_groq import ChatGroq
import chromadb
import pandas as pd
import uuid
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Initialize LLM (Groq Model)
llm = ChatGroq(
    temperature=1,
    groq_api_key='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',
    model='llama-3.1-70b-versatile'
)

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection(name='my_collection')

st.title("AI-Powered Job Postings Scraper & Cold Email Generator")

# Section 1: LLM Query
st.header("LLM Query")
prompt = st.text_input("Enter your query for the Groq Model", "Who are you?")
if st.button("Submit Query"):
    res = llm.invoke(prompt)
    st.write(f"Response: {res.content}")

# Section 2: Vector Database - ChromaDB
st.header("ChromaDB Operations")
docs_list = ["Query is about New York", "This is Kolkata", "Hello Mumbai"]
if st.button("Add documents to ChromaDB"):
    collection.add(
        documents=docs_list,
        ids=['id4', 'id5', 'id3']
    )
    st.success("Documents added successfully!")

if st.button("Retrieve all documents"):
    all_docs = collection.get()
    st.write(all_docs)

if st.button("Delete a document by ID (id4)"):
    collection.delete(ids=['id4'])
    st.success("Document with ID 'id4' deleted!")

# Section 3: Web Scraping
st.header("Web Scraping")
url = st.text_input("Enter the job posting URL to scrape", "https://jobs.nike.com/job/R-37936?from=job%20search%20funnel")
if st.button("Scrape Job Postings"):
    loader = WebBaseLoader(url)
    page_data = loader.load().pop().page_content
    st.write(f"Scraped data: {page_data}")

    prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the
        following keys: `role`, `experience`, `skills` and `description`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):
        """
    )

    chain_extract = prompt_extract | llm
    res = chain_extract.invoke(input={"page_data": page_data})
    st.write(f"Extracted Job Postings JSON: {res.content}")

# Section 4: Cold Email Generation
st.header("Generate Cold Email")
if 'json_res' in locals():
    job_description = res.content  # Assuming job details are extracted
    df = pd.read_csv("my_portfolio.csv")
    collection = client.get_or_create_collection(name='portfolio')

    if st.button("Add Portfolio to ChromaDB"):
        if not collection.count():
            for _, row in df.iterrows():
                collection.add(
                    documents=row["Techstack"],
                    metadatas={'links': row["Links"]},
                    ids=[str(uuid.uuid4())]
                )
        st.success("Portfolio added to ChromaDB!")

    if st.button("Generate Email"):
        # Query portfolio database
        links = collection.query(
            query_texts=['experience in Python', 'experience in ML', 'experience in LLm'],
            n_results=2
        ).get('metadatas')

        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Mohan, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools.
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability,
            process optimization, cost reduction, and heightened overall efficiency.
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
            Remember you are Mohan, BDE at AtliQ.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )

        chain_email = prompt_email | llm
        email_res = chain_email.invoke(input={
            "job_description": str(job_description),
            "link_list": links
        })
        st.write(f"Generated Email: {email_res.content}")
