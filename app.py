import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
import chromadb
import uuid
import pandas as pd
from langchain_core.prompts import PromptTemplate
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Set up LangChain LLM (replace with your API key)
llm = ChatGroq(
    temperature=1,
    groq_api_key='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',
    model='llama-3.1-70b-versatile'
)

# Streamlit app title
st.title("Cold Email Generator from Job Page URL")

# Section: Web Scraping and Job Extraction
st.header("1. Extract Job Description from URL")
url = st.text_input("Enter a job page URL", "https://jobs.nike.com/job/R-37936?from=job%20search%20funnel")

if st.button("Extract Job Description"):
    if url:
        # Load the web page content
        loader = WebBaseLoader(url)
        page_data = loader.load().pop().page_content

        # Set up the prompt to extract job details
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
        
        # Generate the job description using LangChain
        chain_extract = prompt_extract | llm
        res = chain_extract.invoke(input={"page_data": page_data})

        # Display the extracted job description
        st.subheader("Extracted Job Description:")
        st.write(res.content)
        job_description = res.content  # Save for later use in cold email generation
    else:
        st.error("Please enter a valid URL.")

# Section: Upload Portfolio (Optional)
st.header("2. Upload Portfolio (Optional)")
uploaded_file = st.file_uploader("Upload your portfolio CSV", type="csv")

links = []
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

    # Example: Querying for relevant portfolio links
    query = st.text_input("Enter a skill or experience query to find portfolio links", "experience in Python")
    if st.button("Query Portfolio"):
        results = collection.query(query_texts=[query], n_results=2)
        links = results.get('metadatas')
        st.write("Matching Portfolio Links:")
        st.write(links)

# Section: Cold Email Generation
st.header("3. Generate Cold Email Based on Extracted Job Description")

# Dynamic user inputs for personalization
sender_name = st.text_input("Sender's Name", "John Doe")
company_name = st.text_input("Company Name", "AtliQ")
role = st.text_input("Your Role at the Company", "Business Development Executive")
company_overview = st.text_area("Company Overview", "AtliQ is an AI & Software Consulting company dedicated to facilitating the seamless integration of business processes through automated tools. Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, process optimization, cost reduction, and heightened overall efficiency.")

if st.button("Generate Cold Email"):
    if job_description:
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are {sender_name}, a {role} at {company_name}. {company_overview}
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of {company_name}
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase {company_name}'s portfolio: {link_list}
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )

        # Run the chain to generate the cold email
        chain_email = prompt_email | llm
        res = chain_email.invoke(input={
            "job_description": job_description,
            "sender_name": sender_name,
            "role": role,
            "company_name": company_name,
            "company_overview": company_overview,
            "link_list": links
        })

        # Display the generated cold email
        st.subheader("Generated Cold Email:")
        st.write(res.content)
    else:
        st.error("Please extract the job description from the URL first.")
