import streamlit as st
from langchain_groq import ChatGroq
import chromadb
import uuid
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Set up LangChain LLM (replace your API key here)
llm = ChatGroq(
    temperature=1,
    groq_api_key='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',
    model='llama-3.1-70b-versatile'
)

# Streamlit app title
st.title("LangChain + ChromaDB Demo")

# Section: Web Scraping and Job Extraction
st.header("1. Web Scraping and Job Extraction")
url = st.text_input("Enter a job page URL", "https://jobs.nike.com/job/R-37936?from=job%20search%20funnel")

if st.button("Extract Job Postings"):
    loader = WebBaseLoader(url)
    page_data = loader.load().pop().page_content

    prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the
            following keys: `role`, `experience`, `skills`, and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
        """
    )
    
    # Run the chain to extract job information
    chain_extract = prompt_extract | llm
    res = chain_extract.invoke(input={"page_data": page_data})
    
    # Parse the response
    json_parser = JsonOutputParser()
    try:
        json_res = json_parser.parse(res.content)
        st.json(json_res)
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")

# Section: Portfolio Querying with ChromaDB
st.header("2. Portfolio Querying with ChromaDB")

# Upload CSV file for portfolio
uploaded_file = st.file_uploader("Upload your portfolio CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Portfolio:")
    st.dataframe(df)

    # Initialize ChromaDB client and create a collection
    client = chromadb.PersistentClient('vectorstore')
    collection = client.get_or_create_collection(name='portfolio')

    if not collection.count():
        for _, row in df.iterrows():
            collection.add(documents=row["Techstack"],
                           metadatas={'links': row["Links"]},
                           ids=[str(uuid.uuid4())])

    # Query the portfolio
    query = st.text_input("Enter a skill or experience query", "experience in Python")
    if st.button("Query Portfolio"):
        links = collection.query(
            query_texts=[query],
            n_results=2
        ).get('metadatas')
        st.write("Matching Portfolio Links:")
        st.write(links)

# Section: Generate Cold Email
st.header("3. Generate Cold Email Based on Job Description")

if json_res:
    job_description = str(json_res)
    st.write("Extracted Job Description:")
    st.json(json_res)

    # Generate email based on job description
    if st.button("Generate Cold Email"):
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
        res = chain_email.invoke(input={
            "job_description": job_description,
            "link_list": links
        })
        
        st.subheader("Generated Cold Email:")
        st.write(res.content)

