import sys

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


# Show title and description.
st.title("HW 4: Chatbot Using RAG")

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

user_message = st.chat_input('Ask a question about the course documents...')

if user_message:

    st.session_state.messages.append({'role': 'user', 'content': user_message})

    with st.chat_message('user'):
        st.write(user_message)

    client = st.session_state.open_ai_client

    #Embed the user's question with the same model used for the documents
    response = client.embeddings.create(
        input = user_message,
        model = 'text-embedding-3-small'
    )

    query_embedding = response.data[0].embedding

    #Find the documents most similar to the question
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results = 3
    )

    context = "\n\n".join(results['documents'][0])

    system_prompt = (
        'You are a helpful assistant for a course. Answer the user\'s question using only '
        'the course documents below. If the answer is not in the documents, say that you '
        'could not find it in the course documents. If you found an answer, do not say that you could not find more information.'
        'If the users asks a question about something previsouly mentioned in the conversation, source the answer from the conversation memory, not the documents. \n\n'
        f'Course documents:\n\n{context}'
    )

    #Keep only the most recent exchanges so the conversation does not grow without bound
    window = st.session_state.messages[-(2 * BUFFER_EXCHANGES + 1):]

    #The opening greeting is a UI message rather than memory, so the window always starts on a user turn
    while window and window[0]['role'] != 'user':
        window.pop(0)

    #System prompt with the retrieved documents, then the most recent exchanges
    conversation = [{'role': 'system', 'content': system_prompt}] + window

    # Generate an answer using the OpenAI API.
    stream = client.chat.completions.create(
        model='gpt-5-mini',
        messages=conversation,
        stream=True,
    )

    with st.chat_message('assistant'):
        answer = st.write_stream(stream)
        st.caption('Sources: ' + ', '.join(results['ids'][0]))

    st.session_state.messages.append({'role': 'assistant', 'content': answer})

if st.sidebar.button('Clear conversation'):
    st.session_state.messages = st.session_state.messages[:1]
    st.rerun()
