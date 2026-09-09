import streamlit as st
import requests
from openai import OpenAI, AuthenticationError
from anthropic import Anthropic, AuthenticationError
from bs4 import BeautifulSoup

@st.cache_data(show_spinner=False)
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

url1 = st.sidebar.text_input('Enter a URL you would like the ChatBot to pull information from')
#Only offer the second box once the first one is filled in
url2 = st.sidebar.text_input('Enter another URL if you would like') if url1 else ''

url_1_info = read_url_content(url1) if url1 else None
url_2_info = read_url_content(url2) if url2 else None

#How many complete user/assistant exchanges to keep when not using the token buffer
BUFFER_EXCHANGES = 3

#Check the two URL case first, otherwise it can never be reached
if url1 and url2:
    SYSTEM_PROMPT = f'''
        You are a helpful assistant that answers questions based on information pulled from mutliple websites.

        Information from website 1: {url_1_info}

        Information from website 2: {url_2_info}

        Answer questions from the user using information directly from the websites. All answers should be rooted in evidence from the websites, not your own knowledge.
    '''
elif url1:
    SYSTEM_PROMPT = f'''
        You are a helpful assistant that answers questions based on information pulled from websites.

        Information from website: {url_1_info}

        Answer questions from the user using information directly from the website. All answers should be rooted in evidence from the website, not your own knowledge.

    '''
else:
    SYSTEM_PROMPT = None

# Show title and description.
st.title("ChatBot 🗣️🤖")

st.write("This ChatBot can be used in one of two ways:\n\n1. Chat with an LLM the way you would with any other chat bot.\n\n2. Upload one or two links in the sidebar, and ask the ChatBot questions about the contents of those website\n\nMemory: this ChatBot uses a buffer of 6 messages (3 user-agent exchanges) as it's memory.")

st.divider()


if 'open_ai_client' or 'anthropic_client' not in st.session_state:
    #Get OPENAI API Key from secrets file
    openai_api_key = st.secrets.OPENAI_API_KEY
    st.session_state.open_ai_client = OpenAI(api_key=openai_api_key)

    #Get ANTROPIC API Key from secrets file
    anthropic_api_key = st.secrets.ANTHROPIC_API_KEY
    st.session_state.anthropic_client = Anthropic(api_key=anthropic_api_key)

llm_provider = st.sidebar.selectbox(
    'Choose the LLM provider you would like to you.',
    ('ChatGPT 5.6 (OpenAI)', 'Claude Opus 5 (Anthropic)')
).lower().split(' ')[0]

if llm_provider == 'chatgpt':
    model = 'gpt-5.6'
elif llm_provider == 'claude':
    model = 'claude-opus-5'

#Only start the conversation over when the URLs or the model actually change, not on
#every rerun. Switching models has to wipe the memory too, otherwise the new model
#inherits the other one's answers as context
if st.session_state.get('loaded_config') != (url1, url2, llm_provider):
    st.session_state.loaded_config = (url1, url2, llm_provider)
    st.session_state.pop('messages', None)

if 'messages' not in st.session_state:
    if SYSTEM_PROMPT:
        #The system prompt is always the first thing in the queue
        st.session_state.messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'assistant', 'content': 'Hello there! What information would you like to know about the website(s)?'}
        ]
    else:
        st.session_state.messages = [
            {'role': 'assistant', 'content': 'Hello there! How can I help you?'}
        ]

try:

    st.session_state.open_ai_client.models.list()

except AuthenticationError:
    st.info("🚨 Invalid OpenAI API Key")
    st.stop()

#Tests if Anthropic API key is valid
try:

    st.session_state.anthropic_client.messages.create(
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

#Display every message in the conversation so far, apart from the system prompt
for message in st.session_state.messages:
    if message['role'] == 'system':
        continue
    with st.chat_message(message['role']):
        st.write(message['content'])

if prompt := st.chat_input('Whats up?'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.write(prompt)

    #Always keep the system prompt, then the most recent exchanges after it
    system_messages = [m for m in st.session_state.messages if m['role'] == 'system']
    chat_messages = [m for m in st.session_state.messages if m['role'] != 'system']
    window = chat_messages[-(2 * BUFFER_EXCHANGES + 1):]

    #The opening greeting is a UI message rather than memory, and Anthropic rejects a
    #message list that does not start with the user
    while window and window[0]['role'] != 'user':
        window.pop(0)

    conversation = system_messages + window

    if llm_provider == 'chatgpt':
        # Generate an answer using the OpenAI API.
        stream = st.session_state.open_ai_client.chat.completions.create(
            model=model,
            messages=conversation,
            stream=True,
            )
    
        # Stream the response to the app
        with st.chat_message('assistant'):
            response = st.write_stream(stream)

    elif llm_provider == 'claude':

        #Anthropic takes the system prompt as its own argument instead of a message
        stream_kwargs = {
            'model': model,
            'max_tokens': 10000,
            'messages': [m for m in conversation if m['role'] != 'system'],
        }
        if SYSTEM_PROMPT:
            stream_kwargs['system'] = SYSTEM_PROMPT

        # Generate an answer using the Anthropic API.
        with st.chat_message('assistant'):
            #The stream has to be read inside the with block, it closes on the way out
            with st.session_state.anthropic_client.messages.stream(**stream_kwargs) as anthropic_output:
                response = st.write_stream(anthropic_output.text_stream)


    st.session_state.messages.append({'role': 'assistant', 'content': response})

if st.sidebar.button('Clear conversation'):
    st.session_state.clear()
    st.rerun()
