import streamlit as st
import requests
from openai import OpenAI, AuthenticationError
from anthropic import Anthropic, AuthenticationError 
from bs4 import BeautifulSoup

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

# Show title and description.
st.title("Website Summarizer")
st.write(
    "Upload a URL below and an LLM will generate a summary for you! "
)

#Get API keys from secrets file
openai_api_key = st.secrets.OPENAI_API_KEY
anthropic_api_key = st.secrets.ANTHROPIC_API_KEY

#Test if OpenAI API key is valid
try:
    # Create an OpenAI client.
    open_ai_client = OpenAI(api_key=openai_api_key)

    open_ai_client.models.list()

except AuthenticationError:
    st.info("🚨 Invalid OpenAI API Key")
    st.stop()

#Tests if Anthropic API key is valid
try:

    anthropic_client = Anthropic(api_key=anthropic_api_key)

    anthropic_client.messages.create(
        model = 'claude-haiku-4-5',
        max_tokens=1,
        messages = [
            {
                'role': 'user',
                'content': 'hi'
            }
        ]
    )

except AuthenticationError:
    st.info("🚨 Invalid Anthropic API Key")
    st.stop()


#Ask the user how they want their document summarized
summarize_option = st.sidebar.selectbox(
    'How would you like your document to be summarized?',
    ('100 words', '2 connecting paragraphs', '5 bullet points')
)

#Gets users langauge option input
language_option = st.sidebar.selectbox(
    'Select the language you would like to recieve the output in.',
    ('English', 'Spanish', 'French')
)

#Define a prompt for summarization based on option selected above
if summarize_option == '100 words':
    summarize_prompt = '''
    Summarize this document in 100 words or less. NEVER exceed the 100 word summary limit. DO NOT include a header or anything else, ONLY return the 100 word summary.

    Example output: 

    Tyler Katz is an aspiring AI/ML engineer and Syracuse University graduate student pursuing an M.S. in Applied Human Centered AI. He holds a B.S. in Applied Data Analytics. At Comcast, he automated network inventory reconciliation using LLM agents and FastAPI, reducing processing time from hundreds of hours to minutes. He leads Syracuse's United AI organization and previously worked as a data analytics research assistant. Tyler specializes in AI/LLM engineering, RAG systems, and fine-tuning models. His projects include agentic RAG systems, fine-tuned LLM microservices, NLP classifiers, and machine learning pipelines, showcasing production-ready implementations.
    '''
elif summarize_option == '2 connecting paragraphs':
    summarize_prompt = '''
    Summarize this document in two individual connecting paragraphs. NEVER exceed the 2 paragraph limit. MAKE SURE the paragraphs flow into one another. DO NOT include a header or anything else, ONLY return the 2 connected paragraphs
    
    Example output:

    Tyler Katz is a graduate student at Syracuse University pursuing a Master's in Applied Human-Centered Artificial Intelligence, with a strong foundation from his undergraduate degree in Applied Data Analytics. His career trajectory reflects a deliberate shift from understanding data to building the systems that operationalize AI models in production environments. Most notably, during his data science internship at Comcast, he developed an LLM agent and MCP integration that automated a quarterly network inventory reconciliation process, reducing it from hundreds of hours to just minutes while cutting token costs by 30-40%. This achievement exemplifies his core motivation: transforming slow, manual processes into trustworthy automated systems that deliver measurable business impact.

    Beyond his professional work, Tyler demonstrates leadership and technical depth across the full AI/ML stack—from classical machine learning and data visualization to advanced techniques like RAG systems, LLM fine-tuning with LoRA/QLoRA, and agentic AI. As Engineering Lead for United AI, Syracuse's AI/ML student organization, he mentors 10 teams and helps students move from learning to shipping real projects. His portfolio includes sophisticated projects like a query-reformulation RAG agent that outperforms naive baselines, a fine-tuned LLM microservice deployed on AWS with full CI/CD infrastructure, and classical ML applications achieving 95%+ accuracy on fake news detection. Combined with his research assistant work on geospatial visualizations that contributed to an academic book presented at the UN, Tyler is building a compelling profile as an engineer who bridges rigorous AI research with practical, production-grade systems.
    '''
elif summarize_option == '5 bullet points':
    summarize_prompt = '''
    Summarize this document in 5 individual bullet points. NEVER exceed the five bullet point limit. DO NOT include a header or anything else, ONLY return the 5 bullet points.
    
    Example output:

    -Tyler Katz is a Syracuse University graduate student (M.S. Applied Human-Centered AI) with a B.S. in Applied Data Analytics, focused on becoming an AI/ML engineer specializing in agents, RAG systems, and fine-tuned LLMs.

    -As a Data Science Intern at Comcast, he automated network inventory reconciliation using an LLM agent, FastAPI, and MCP integrations—reducing a hundreds-of-hours process to minutes while cutting token costs 30–40%.

    -He serves as Engineering Lead for United AI, Syracuse's student AI/ML organization, leading 10 project teams, managing GitHub CI/CD infrastructure, and teaching 200+ members technical skills.

    -His project portfolio spans AI/LLM work (a LangGraph RAG research assistant and a QLoRA fine-tuned job-extraction microservice) and classical ML/DS projects (fake news detection, sales forecasting, and customer segmentation).

    -He previously worked as a Data Analytics Research Assistant, building visualizations and geospatial maps for a published book on crime and politics in Brazil, with some maps presented at the UN.
    '''

#Gets user LLM provider input, makes input shorter for later use
llm_provider = st.sidebar.selectbox(
    'Choose the LLM provider you would like to you.',
    ('ChatGPT (OpenAI)', 'Claude (Anthropic)')
).lower().split(' ')[0]

#Let the user select which model they want to use
if st.sidebar.checkbox('Use advanced model'):
    if llm_provider == 'chatgpt':
        model = 'gpt-5.4-mini'
    elif llm_provider == 'claude':
        model = 'claude-sonnet-5'
else:
    if llm_provider == 'chatgpt':
        model = 'gpt-5.4-nano'
    elif llm_provider == 'claude':
        model = 'claude-haiku-4-5'

# Let the user upload URL
uploaded_link = st.text_input("Enter a URL to a website")

if uploaded_link:

    #Pull the text off the page
    document = read_url_content(uploaded_link)

    if document:
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {summarize_prompt} \n\n---\n\n Generate this summary in {language_option}",
            }
        ]

        if llm_provider == 'chatgpt':
            # Generate an answer using the OpenAI API.
            stream = open_ai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

            # Stream the response to the app
            st.write_stream(stream)
        elif llm_provider == 'claude':
            # Generate an answer using the Anthropic API.
            with anthropic_client.messages.stream(
                model = model,
                max_tokens = 10000,
                messages = messages,
            ) as anthropic_output:

                # Stream the response to the app
                st.write_stream(anthropic_output.text_stream)
    else:
        st.info('🚨 URL entered is invalid or could not be read')


