import sys
import json

# Swap in pysqlite3 before chromadb loads (needed on Streamlit Cloud; local SQLite is new enough)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
from openai import OpenAI, AuthenticationError
import chromadb
from pathlib import Path
from bs4 import BeautifulSoup

#Used the same collection as HW4 to save token costs on embedding
chroma_client = chromadb.PersistentClient(path = './ChromaDB_for_HW')
collection = chroma_client.get_or_create_collection('HW4Collection')

CHROME_TAGS = ['script', 'style', 'nav', 'footer', 'header']
OVERLAP = 200

#How many complete user/assistant exchanges to keep as the chatbot's memory
BUFFER_EXCHANGES = 5

if 'open_ai_client' not in st.session_state:
    #Get OPENAI API Key from secrets file
    openai_api_key = st.secrets.OPENAI_API_KEY
    st.session_state.open_ai_client = OpenAI(api_key=openai_api_key)

def extract_clean_text(path):
    """Read one HTML file and return its visible text with site chrome removed."""
    with open(path, 'rb') as f:  # binary: let bs4 detect the encoding
        soup = BeautifulSoup(f, 'html.parser')

    for tag in soup(CHROME_TAGS):
        tag.decompose()

    return soup.get_text(' ', strip=True)


def split_in_two(text, overlap=OVERLAP):
    """Split text into exactly two overlapping chunks, cutting on word boundaries."""
    if len(text) < 2 * overlap:
        return [text]  # too short to split meaningfully

    mid = len(text) // 2

    raw_end = mid + overlap
    snapped_end = text.find(' ', raw_end)
    end_1 = raw_end if snapped_end == -1 else snapped_end

    raw_start = mid - overlap
    snapped_start = text.rfind(' ', 0, raw_start)
    start_2 = raw_start if snapped_start == -1 else snapped_start + 1

    return [text[:end_1], text[start_2:]]


def chunk_folder(folder):
    """Yield (id, document, metadata) for every chunk in every HTML file."""
    for path in sorted(Path(folder).glob('*.html')):
        chunks = split_in_two(extract_clean_text(path))
        for i, chunk in enumerate(chunks):
            yield (
                f'{path.stem}_{i}',
                chunk,
                {'source': path.name, 'chunk_index': i, 'chunk_count': len(chunks)},
            )
    

def load_html_to_collection(folder_path, collection, batch_size = 50):

    client = st.session_state.open_ai_client

    ids = []
    documents = []
    metadatas = []

    def flush():

            if len(ids) == 0:
                return
            else:
                response = client.embeddings.create(
                        input = documents,
                        model = 'text-embedding-3-small'
                )

                embeddings = [embedding.embedding for embedding in response.data]

                collection.add(
                        documents = documents,
                        ids = ids,
                        embeddings = embeddings,
                        metadatas = metadatas
                    )

                ids.clear()
                documents.clear()
                metadatas.clear()

    for chunk in chunk_folder(folder_path):
        ids.append(chunk[0])
        documents.append(chunk[1])
        metadatas.append(chunk[2])

        if len(ids) >= batch_size:
            flush()

    flush()
        


if collection.count() == 0:
    load_html_to_collection('data/su_orgs', collection)

#CHUNKING METHOD EXPLANATION
# I used the fixed size chunking method. I chose this method because semantic chunking would've been difficult with HTML files since defined blocked of text are slightly difficult to interpret
# and I didn't want to use ML-guided chunking because I did not have good training data, and I didn't want to use an LLM because that would've increased API costs for this assignment.
# I added a 200 word overlap, and made an addition so that chunks didn't split words, it would only split on white space.


def relevant_club_info(query, n_results = 3):
    """Vector search the collection for the query and return (context, source ids)."""
    client = st.session_state.open_ai_client

    #Embed the query with the same model used for the documents
    response = client.embeddings.create(
        input = query,
        model = 'text-embedding-3-small'
    )

    query_embedding = response.data[0].embedding

    #Find the documents most similar to the query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results = n_results
    )

    context = "\n\n".join(results['documents'][0])

    return context, results['ids'][0]


#Tool definition the LLM uses to decide when (and with what query) to search the club documents
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'relevant_club_info',
            'description': (
                'Search the Syracuse University student organization documents and return the passages '
                'most relevant to the query. Use this for any question about clubs or organizations at Syracuse University.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'A search query describing the club information needed, e.g. "robotics club meeting times".'
                    }
                },
                'required': ['query'],
                'additionalProperties': False
            }
        }
    }
]


# Show title and description.
st.title("HW 5: Chatbot Using RAG")

st.write("Memory: this ChatBot uses a buffer of 10 messages (5 user-agent exchanges) as it's memory.")

st.divider()

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'assistant', 'content': 'Hello there! What can I help you with?'}
    ]

try:

    st.session_state.open_ai_client.models.list()

except AuthenticationError:
    st.info("🚨 Invalid OpenAI API Key")
    st.stop()

#Display every message in the conversation so far
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

user_message = st.chat_input('Ask a question about clubs/organizations at Syracuse University...')

if user_message:

    st.session_state.messages.append({'role': 'user', 'content': user_message})

    with st.chat_message('user'):
        st.write(user_message)

    client = st.session_state.open_ai_client

    #Keep only the most recent exchanges so the conversation does not grow without bound
    window = st.session_state.messages[-(2 * BUFFER_EXCHANGES + 1):]

    #The opening greeting is a UI message rather than memory, so the window always starts on a user turn
    while window and window[0]['role'] != 'user':
        window.pop(0)

    router_prompt = (
        'You are a helpful assistant for organizations at Syracuse University. If the user asks about clubs or '
        'organizations, call relevant_club_info with a search query describing the information needed. '
        'If the user asks about something previously mentioned in the conversation, answer from the conversation memory '
        'without calling the tool.'
    )

    #First call: let the LLM decide whether to search the club documents, and with what query
    first_response = client.chat.completions.create(
        model='gpt-5-mini',
        messages=[{'role': 'system', 'content': router_prompt}] + window,
        tools=TOOLS,
        tool_choice='auto',
    )

    tool_calls = first_response.choices[0].message.tool_calls

    with st.chat_message('assistant'):

        if tool_calls:
            contexts = []
            sources = []

            #Run the vector search for every query the LLM asked for
            for tool_call in tool_calls:
                if tool_call.function.name == 'relevant_club_info':
                    query = json.loads(tool_call.function.arguments)['query']
                    context, ids = relevant_club_info(query)
                    contexts.append(context)
                    sources.extend(ids)

            context = "\n\n".join(contexts)

            system_prompt = (
                'You are a helpful assistant for organizations at Syracuse University. Answer the user\'s question using only '
                'the organization documents below. If the answer is not in the documents, say that you '
                'could not find it in the organization documents. If you found an answer, do not say that you could not find more information.'
                'If the users asks a question about something previsouly mentioned in the conversation, source the answer from the conversation memory, not the documents. \n\n'
                f'Organization documents:\n\n{context}'
            )

            #Second call: answer with the search results in the system prompt, no tools available
            stream = client.chat.completions.create(
                model='gpt-5-mini',
                messages=[{'role': 'system', 'content': system_prompt}] + window,
                stream=True,
            )

            answer = st.write_stream(stream)
            st.caption('Sources: ' + ', '.join(dict.fromkeys(sources)))

        else:
            #The LLM answered directly (e.g. from conversation memory) without searching
            answer = first_response.choices[0].message.content
            st.write(answer)

    st.session_state.messages.append({'role': 'assistant', 'content': answer})

if st.sidebar.button('Clear conversation'):
    st.session_state.messages = st.session_state.messages[:1]
    st.rerun()
