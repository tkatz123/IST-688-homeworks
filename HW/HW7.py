import sys
import json
import pandas as pd
import hashlib

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


# =============================================================================
# 1. CONFIG
# =============================================================================
# TODO: build these paths from the location of this file, not the working directory,
#       so they resolve the same locally and on Streamlit Cloud.
#       Hint: Path(__file__) is this script; the repo root is one level above HW/.
CSV_PATH = None        # -> data/hw_7_data/news.csv
CHROMA_PATH = None     # -> ChromaDB_for_HW (same folder as HW4/HW5, new collection below)

COLLECTION_NAME = 'HW7Collection'
EMBEDDING_MODEL = 'text-embedding-3-small'

# Cheap vs expensive, for step 4 of the assignment. Both OpenAI so one tool-calling loop serves both.
# TODO: confirm both names against the OpenAI models page before the build run.
CHEAP_MODEL = 'gpt-5-mini'      # what HW4/HW5 used
EXPENSIVE_MODEL = None

#How many complete user/assistant exchanges to keep as the chatbot's memory
BUFFER_EXCHANGES = 5

# TODO: write the "interesting" rubric for a global law firm monitoring its clients.
#       4 or 5 criteria, each saying what scores high and what scores low.
#       Ask for structured output: an integer score 1-10 and a one-sentence reason.
SCORING_PROMPT = """
"""

# TODO: ~20 article ids you hand-label as interesting BEFORE you look at any model's scores.
#       Used for the testing section of the write-up (step 3b).
HAND_LABELED_INTERESTING = []


# =============================================================================
# CLIENTS
# =============================================================================
if 'open_ai_client' not in st.session_state:
    #Get OPENAI API Key from secrets file
    openai_api_key = st.secrets.OPENAI_API_KEY
    st.session_state.open_ai_client = OpenAI(api_key=openai_api_key)


# =============================================================================
# 2. BUILD (runs once, only when the collection is empty)
# =============================================================================
# Run this locally, then commit the Chroma folder so the deployed app only loads it.

def load_articles(csv_path):
    """Read the CSV and return one record per unique URL (204 duplicate rows in the raw file)."""
    # TODO: read rows, keep the first row for each URL.
    # TODO: make a stable id per article from its URL (hashlib), not the row number.
    pass


def score_article(document, model):
    """Ask `model` to score one article against SCORING_PROMPT. Return (score, reason)."""
    # TODO: one chat call with structured output, parse the integer score and the reason.
    pass


def build_collection(collection, batch_size=50):
    """Dedupe, embed, score with both models, and add everything to the collection."""
    # TODO: embed `Document` whole, one vector per article, in batches (same flush pattern as HW5).
    # TODO: score every article with CHEAP_MODEL and EXPENSIVE_MODEL.
    # TODO: metadata per article: company, date, url,
    #       score_cheap, reason_cheap, score_expensive, reason_expensive.
    #       Chroma metadata values must be str / int / float / bool.
    pass


# =============================================================================
# 3. LOAD
# =============================================================================
# TODO: wrap the client + collection in a function decorated with @st.cache_resource
#       so it opens once per server instead of on every rerun.
chroma_client = None
collection = None

# TODO: if the collection is empty, call build_collection(collection).


# =============================================================================
# 4. TOOLS
# =============================================================================

def rank_interesting(n=10, company=None, score_field='score_cheap'):
    """Return the top n articles by score, optionally for one company."""
    # TODO: collection.get() with a `where` filter when company is given.
    # TODO: sort in Python by score_field (Chroma cannot sort by metadata), ties broken by date.
    # TODO: return compact JSON: company, date, url, score, reason, headline.
    pass


def search_news(query, n=5):
    """Return articles about a company or topic."""
    # TODO: if query matches a company_name, use a metadata filter.
    # TODO: otherwise embed the query and run collection.query() (semantic search, as in HW5).
    # TODO: return the same compact JSON shape as rank_interesting.
    pass


#Tool definitions the LLM uses to decide which function to call, and with what arguments
TOOLS = [
    # TODO: one function schema for rank_interesting (n, company)
    # TODO: one function schema for search_news (query, n)
]


# =============================================================================
# 6. SIDEBAR (model toggle)
# =============================================================================
# TODO: st.sidebar.radio for cheap vs expensive.
#       It switches BOTH the model that answers and the score field rank_interesting sorts by.
model = CHEAP_MODEL
score_field = 'score_cheap'


# =============================================================================
# 5. CHAT
# =============================================================================
st.title("HW 7: News Information Bot")

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

user_message = st.chat_input('Ask about the news...')

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

    # TODO: system prompt. Answer only from tool results, always cite the URL,
    #       explain "interesting" with the stored reason, say so when the articles don't cover it.
    system_prompt = ''

    # TODO: first call with tools=TOOLS, tool_choice='auto'.
    # TODO: run each tool call (rank_interesting / search_news), passing score_field to rank_interesting.
    # TODO: send the tool results back and stream the final answer.
    # TODO: if no tool was called, answer directly from conversation memory.
    answer = ''

    st.session_state.messages.append({'role': 'assistant', 'content': answer})

if st.sidebar.button('Clear conversation'):
    st.session_state.messages = st.session_state.messages[:1]
    st.rerun()
