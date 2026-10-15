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


# =============================================================================
# 1. CONFIG
# =============================================================================
CSV_PATH = 'data/hw_7_data/news.csv'
CHROMA_PATH = './ChromaDB_for_HW'

COLLECTION_NAME = 'HW7Collection'
EMBEDDING_MODEL = 'text-embedding-3-small'

# Cheap vs expensive, for step 4 of the assignment. Both OpenAI so one tool-calling loop serves both.
CHEAP_MODEL = 'gpt-5-mini'
EXPENSIVE_MODEL = 'gpt-5.2'

#How many complete user/assistant exchanges to keep as the chatbot's memory
BUFFER_EXCHANGES = 5

#Rubric for the "interesting" score. Legal and regulatory exposure is weighted above transactions on purpose.
#The user message must carry both the client company name and the article text, or the relevance gate cannot work.
SCORING_PROMPT = """
You are a news analyst at a large global law firm. The firm monitors news about its clients.
You will receive a client company name and one news article. Score how important this article
is for the firm to know about, from the perspective of a partner deciding whether to forward it
to the lawyer who manages that client relationship.

Step 1, relevance gate.
Is the article actually about the named client company? If the name refers to something else
(a different organization, a place, a sports team, a person) or the company is only mentioned
in passing, give a score of 1 and stop.

Step 2, score what the article means for the client, using these criteria in priority order:
1. Legal and regulatory exposure (highest weight): lawsuits, investigations, enforcement actions,
   fines, settlements, or new regulation that directly affects the company.
2. Corporate transactions: mergers, acquisitions, divestitures, major financings, large contracts,
   executive departures.
3. Material financial events: earnings surprises, guidance changes, large write-downs, debt or
   credit trouble.
4. Reputational or operational risk: data breaches, recalls, safety incidents, labor disputes,
   scandals.

If an article touches more than one criterion, score it by the highest-priority criterion present.
If the article is a roundup of several stories, score only the part about the named company.

Scale:
- 9 to 10: active legal or regulatory action against the company, or a major transaction that
  would require outside counsel.
- 7 to 8: a significant transaction, a material financial event, or credible emerging legal or
  regulatory risk.
- 4 to 6: notable business news with indirect legal relevance, such as leadership changes,
  large contracts, or routine earnings.
- 2 to 3: routine business news: product launches, marketing, events, stock-chart commentary.
- 1: not about the company, or no business relevance.

Use the full scale. Most articles in a news feed are routine, so expect most scores to fall
between 2 and 5. Reserve 9 and 10 for articles a partner would act on the same day.

Return JSON only, in exactly this form:
{"score": <integer from 1 to 10>, "reason": "<one sentence naming the specific legal or business angle>"}
"""

#Hand labels for testing (step 3b): a seeded random sample of 50 deduplicated articles (seed 20261014).
#First-pass hand labels (original_label in the CSV) were revised against SCORING_PROMPT with Claude.
#Mapping to the 1-10 rubric: 3 = 7-10, 2 = 4-6, 1 = 1-3 or fails the relevance gate.
#Keyed by URL. Source: data/hw_7_data/hand_label_sample.csv
HAND_LABELS = {
    'https://gulfbusiness.com/how-gcc-wealth-funds-are-dominating-the-world/': 1,
    'https://www.fnlondon.com/articles/cinven-plots-mid-market-expansion-under-new-leaders-ce3def59': 2,
    'https://ca.investing.com/news/company-news/xerox-stock-hits-52week-low-at-1018-amid-market-challenges-93CH-3533733': 1,
    'https://www.reuters.com/markets/asia/japans-mitsubishi-q1-net-profit-up-12-yy-asset-sales-weaker-yen-2024-08-01/': 2,
    'https://www.yahoo.com/tech/fujifilm-x100vi-officially-pro-camera-064958247.html': 1,
    'https://ktla.com/sports/clippers/los-angeles-clippers-to-open-flagship-store-at-intuit-dome/': 1,
    'https://www.devonlive.com/news/devon-news/employers-elmore-afc-net-over-9451891': 1,
    'https://fortune.com/2024/08/03/coca-cola-irs-6-billion-back-taxes-us-income-foreign-affiliates-federal-court/': 3,
    'https://www.livemint.com/opinion/quick-edit/mint-quick-edit-infosys-gst-notice-service-export-input-tax-credit-taxation-tax-clarity-11722799082177.html': 3,
    'https://seekingalpha.com/article/4709112-brookfield-asset-management-how-to-gain-exposure-to-alternative-assets': 1,
    'https://www.theverge.com/2024/7/31/24210036/sony-new-headphones-nfl-coaches': 1,
    'https://www.tvtechnology.com/news/commscope-launches-starline-esd-18-ghz-amplifiers': 1,
    'https://heavy.com/sports/mlb/baltimore-orioles/anthony-santander-predicted-leave-orioles/': 1,
    'https://www.bloomberg.com/news/articles/2024-07-30/samsung-begins-closing-gap-in-making-ai-memory-chips-for-nvidia': 1,
    'https://www.investors.com/ibd-data-stories/stocks-to-watch-analog-devices-sees-rs-rating-jump-to-84/': 1,
    'https://www.finsmes.com/2024/08/savvy-wealth-closes-26-5m-series-a-funding.html': 1,
    'https://www.wdwmagic.com/other/severe-weather-information/news/02aug2024-heavy-rain-and-wind-forecasted-for-walt-disney-world-as-tropical-cyclone-4-approaches.htm': 1,
    'https://www.investing.com/news/company-news/barclays-upgrades-asml-stock-citing-attractive-opportunity-amid-recent-selloff-93CH-3542975': 1,
    'https://www.channelweb.co.uk/news/4343232/computacenter-vet-nickolls-resurfaces-scc-reseller-pursues-aggressive-expansion': 1,
    'https://investorplace.com/market360/2024/08/broadcom-stock-buy-alert-the-hidden-ai-gem-with-dividend-potential/': 1,
    'https://www.manufacturingdive.com/news/nxp-semiconductors-inventory-drags-Q2-earnings/722922/': 2,
    'https://citywire.com/investment-trust-insider/news/expert-view-games-workshop-big-technologies-wpp-restore-computacenter/a2447515': 1,
    'https://www.aboutamazon.com/news/retail/amazon-just-walk-out-improves-accuracy': 1,
    'https://www.prnewswire.com/news-releases/scott-russell-and-julia-white-to-leave-sap-executive-board-302209547.html': 2,
    'https://sg.news.yahoo.com/app-annie-appoints-cindy-deng-managing-director-asia-034910270.html': 1,
    'https://www.marketbeat.com/originals/leading-oil-refiners-stock-climbs-despite-market-volatility/': 1,
    'https://www.msn.com/en-in/money/topstories/microsoft-cloud-ai-demand-up-cues-for-tcs-infosys-hcl-tech-wipro-techm/ar-BB1qWlH9': 1,
    'https://www.businesswire.com/news/home/20240730838233/en/Clutch-Security-Launches-With-8.5M-in-Seed-Funding-to-Revolutionize-the-Way-Modern-Organizations-Secure-and-Manage-Their-Non-Human-Identities': 1,
    'https://www.reit.com/news/video/tpg-re-finance-trust-focusing-recycling-capital-new-opportunities': 1,
    'https://www.investors.com/news/technology/vz-stock-verizon-stock-buy-now/': 1,
    'https://www.sportspromedia.com/news/kylian-mbappe-caen-takeover-interconnected-ventures-coalition-capital-oaktree/': 3,
    'https://www.cnbc.com/2024/07/30/farnborough-air-show-boeing-airbus-orders-supply-chain-woes.html': 2,
    'https://therealdeal.com/national/2024/08/03/deutsche-bank-set-to-dump-1-billion-in-real-estate-loans/': 3,
    'https://www.forbes.com/sites/jasonevangelho/2024/08/04/lenovo-website-update-seemingly-confirms-new-legion-go-models/': 1,
    'https://www.wired.com/review/asus-chromebook-cm14/': 1,
    'https://www.01net.it/sainsburys-partners-with-ncr-voyix-for-exceptional-customer-experience/': 2,
    'https://www.theinsurer.com/news/fortegra-gets-40mn-from-tiptree-and-warburg-pincus-to-support-further-growth/': 2,
    'https://operations.nfl.com/updates/the-game/sony-and-nfl-launch-new-technology-partnership/': 2,
    'https://www.virginiabusiness.com/article/micron-expects-to-expand-manassas-facility-soon/': 2,
    'https://www.smart-energy.com/industry-sectors/smart-energy/schneider-electric-launches-training-programme-for-electrical-engineers/': 1,
    'https://www.axios.com/2024/07/30/bronfman-still-eyeing-paramount-deal': 1,
    'https://www.newprivatemarkets.com/tom-steyers-galvanize-brings-real-estate-decarbonisation-fund-to-market/': 1,
    'https://www.tipranks.com/news/the-fly/analog-devices-call-volume-above-normal-and-directionally-bullish-3': 1,
    'https://www.reuters.com/technology/lam-research-forecasts-quarterly-revenue-above-estimates-fueled-by-ai-boom-2024-07-31/': 3,
    'https://thewaltdisneycompany.com/disney-princesses-come-together-to-inspire-girls-to-create-your-world-through-new-multi-year-brand-campaign/': 1,
    'https://www.kedglobal.com/electronics/newsView/ked202407300004': 1,
    'https://www.afr.com/companies/transport/inside-the-boardroom-civil-war-that-brought-rex-to-its-knees-20240731-p5jy4c': 3,
    'https://www.northjersey.com/story/news/2024/08/04/walt-disney-world-weather-debby/74665662007/': 1,
    'https://asiatechdaily.com/beep-expands-voltality-network-across-southeast-asia-with-3-3m-funding-boost/': 1,
    'https://www.constructionequipment.com/equipment-maintenance-and-repair/news/55130274/jpro-update-supports-caterpillar-cummins': 1,
}


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
# Run build_collection once locally on an empty collection, then commit the Chroma folder so the deployed app only loads it.

def load_articles(csv_path):
    """Read the CSV and return one record per unique URL (204 duplicate rows in the raw file)."""
    news_df = pd.read_csv(csv_path)

    #Deduplicate on URL, keep first instance
    cleaned_news_df = news_df.drop_duplicates(subset = 'URL', keep = 'first')

    #Strip and lower company name
    cleaned_news_df['company_name'] = cleaned_news_df['company_name'].str.strip().str.lower()

    #Creating a unique row_id for each row based on the URL
    cleaned_news_df["row_id"] = cleaned_news_df["URL"].apply(
    lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest()
    )

    return cleaned_news_df


def score_article(document, company, model):
    """Ask `model` to score one article against SCORING_PROMPT. Return a dict containing 'score' and 'reason'."""

    #Makes a call to an LLM to score each article's interest based on a criteria defined in the SCORING_PROMPT
    response = st.session_state.open_ai_client.chat.completions.create(
        model = model,
        messages = [
            {'role': 'system', 'content': SCORING_PROMPT},
            {'role': 'user', 'content': f'Client Company: {company} \n\n Article: \n\n {document}'}
        ],
        response_format = {
            'type': 'json_schema',
            'json_schema': {
                'name': 'article_score',
                'strict': True,
                'schema': {
                    'type': 'object',
                    'properties': {
                        'score':  {'type': 'integer'},
                        'reason': {'type': 'string'}
                    },
                    'required': ['score', 'reason'],
                    'additionalProperties': False
                }
            }
        }
    )

    #Tries to parse the output and return JSON
    #If failed will return a score of one and state in the reason that parsing failed
    try:
        return json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return {'score': 1, 'reason': 'Scoring failed'}


def build_collection(collection, batch_size=50):
    """Dedupe, embed, score with both models, and add everything to the collection."""
    
    #loading in data
    df = load_articles(CSV_PATH)

    #Initializing empty batch lists
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    counter = 0

    def flush():

        #Return early if batch is empty
        if len(ids) == 0:
            return
        else:

            #Embed all documents in batch
            response = st.session_state.open_ai_client.embeddings.create(
                input = documents,
                model = EMBEDDING_MODEL
            )

            #Extract embeddings
            embeddings = [embedding.embedding for embedding in response.data]

            #Add ids, embeddings, and metadata to collection
            collection.add(
                documents = documents,
                ids = ids,
                embeddings = embeddings,
                metadatas = metadatas
            )

            #Clear lists for next batch
            ids.clear()
            documents.clear()
            metadatas.clear()
            embeddings.clear()

    #Loop over the dataframe
    for idx, row in df.iterrows():

        #Score each document with cheap & expensive model
        cheap_score = score_article(row['Document'], row['company_name'], CHEAP_MODEL)
        expensive_score = score_article(row['Document'], row['company_name'], EXPENSIVE_MODEL)

        #Creating meta data dictionary
        metadata_dict = {
            'company': str(row['company_name']),
            'date': row['Date'],
            'url': str(row['URL']),
            'score_cheap': int(cheap_score['score']),
            'reason_cheap': str(cheap_score['reason']),
            'score_expensive': int(expensive_score['score']),
            'reason_expensive': str(expensive_score['reason'])
        }   

        ids.append(row['row_id'])
        documents.append(row['Document'])
        metadatas.append(metadata_dict)

        counter += 1

        if len(ids) >= batch_size:
            flush()
            print(f'{counter} articles added to ChromaDB')

    flush()


# =============================================================================
# 3. LOAD
# =============================================================================

@st.cache_resource
def load_db():

    chroma_client = chromadb.PersistentClient(path = CHROMA_PATH)

    return chroma_client.get_collection(COLLECTION_NAME)

#If running this build from scratch comment out guard so collection can be built without being errored out
try:
    #Load HW7 collection from Chroma DB
    collection = load_db()

    if collection.count() == 0:
        st.error('Collection is empty')
        st.stop()
except Exception:
    st.error('Unable to load database')
    st.stop()

#Gets a list of company names
COMPANY_NAMES = list(sorted({article['company'] for article in collection.get(include=['metadatas'])['metadatas']}))    

# =============================================================================
# 4. TOOLS
# =============================================================================

def rank_interesting(n=10, company=None, score_field='score_cheap'):
    """Return the top n articles by score, optionally for one company."""
    
    #Caps n at 20 for context window purposes
    n = min(n,20)

    #If a company is passed in strip whitespace and convert to lowercase, filter collection by company
    if company is not None:
        company = company.strip().lower()
        results = collection.get(where = {'company': company})
    else:
        results = collection.get()

    #Checks to make sure results is not empty
    if len(results['ids']) == 0:
        return f'No articles found for company: {company}'

    ranked = sorted(
        zip(results['documents'], results['metadatas']),
        key = lambda pair: (pair[1][score_field], pair[1]['date']),
        reverse = True
    )

    top_articles = []

    reason_field = score_field.replace('score', 'reason')

    for doc, meta in ranked[:n]:
        article = {
            'headline': doc.split('Description:')[0].strip(),
            'company': meta['company'],
            'date': meta['date'],
            'url': meta['url'],
            'score': meta[score_field],
            'reason': meta[reason_field]
        }
        top_articles.append(article)

    return json.dumps(top_articles)

    

def search_news(query, company=None, n=5, score_field='score_cheap'):
    """Return articles about a company or topic."""
    
    #Caps output at 20 articles
    n = min(n,20)

    #Get embedding of query
    response = st.session_state.open_ai_client.embeddings.create(
                    input = query,
                    model = EMBEDDING_MODEL
                )

    embeddings = response.data[0].embedding

    if company is not None:
        company = company.strip().lower()

        #Query chromaDB with the company filter
        results = collection.query(query_embeddings=[embeddings], n_results=n, where={'company': company})
    else:

        #Query chromaDB without the company filter
        results = collection.query(query_embeddings=[embeddings], n_results=n)

    #Extracting documents and metadata
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    #Making sure articles were found
    if len(documents) == 0:
        return f'No articles found'

    top_articles = []

    reason_field = score_field.replace('score', 'reason')

    for doc, meta in zip(documents, metadatas):
        article = {
            'headline': doc.split('Description:')[0].strip(),
            'company': meta['company'],
            'date': meta['date'],
            'url': meta['url'],
            'score': meta[score_field],
            'reason': meta[reason_field]
        }
        top_articles.append(article)

    return json.dumps(top_articles)


#Tool definitions the LLM uses to decide which function to call, and with what arguments
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'rank_interesting',
            'description': ('ranks the most interesting or important news for the law firm, ' 
                            'optionally for one company. Use it when the user asks what\'s ' 
                            'most interesting, important or worth knowing.'
                            ),
            'strict': True,
            'parameters': {
                'type': 'object',
                'properties': {
                    'n': {
                        'type': 'integer',
                        'description': 'Use 10 unless the user asks for a specific number. Maximum 20.'
                    },
                    'company': {
                        'type': ['string', 'null'],
                        'description': ('Use null when the user doesn\'t name a company. If the user names a company, '
                                        'choose the matching name from the list (for example, \'JP Morgan\' is \'jpmorgan chase\')'),
                        'enum': COMPANY_NAMES + [None]
                    }
                },
                'required': ['n', 'company'],
                'additionalProperties': False
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_news',
            'description': 'finds news about a topic or company. Use it when the user asks for news about something specific.',
            'strict': True,
            'parameters': {
                'type': 'object',
                'properties': {
                    'query':{
                        'type': 'string',
                        'description': 'The topic to search for, in natural language'
                    },
                    'n': {
                        'type': 'integer',
                        'description': 'Use 5 unless the user asks for a specific number. Maximum 20.'
                    },
                    'company': {
                        'type': ['string', 'null'],
                        'description': ('Use null when the user doesn\'t name a company. If the user names a company, '
                                        'choose the matching name from the list (for example, \'JP Morgan\' is \'jpmorgan chase\')'),
                        'enum': COMPANY_NAMES + [None]
                    }
                },
                'required': ['query', 'n', 'company'],
                'additionalProperties': False
            }
        }
    }
]


# =============================================================================
# 5. SIDEBAR (model toggle)
# =============================================================================
model_selection = st.sidebar.radio(
    label = 'Select a Model',
    options = [
        f'Basic: {CHEAP_MODEL}',
        f'Advanced: {EXPENSIVE_MODEL}'
    ]
)

if model_selection == f'Basic: {CHEAP_MODEL}':
    model = CHEAP_MODEL
    score_field = 'score_cheap'
elif model_selection == f'Advanced: {EXPENSIVE_MODEL}':
    model = EXPENSIVE_MODEL
    score_field = 'score_expensive'


# =============================================================================
# 6. CHAT
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

    #Grounds every answer in tool results, requires URLs, and explains "interesting" with the stored score and reason
    system_prompt = (
        'You are a news assistant for a large global law firm. The firm monitors news about its client companies. '
        'The news you can search covers July 30 to August 6, 2024. You have no news outside that window, so never '
        'present these articles as current and never answer from your own knowledge of events.\n\n'
        'Tools:\n'
        '- rank_interesting: use it when the user asks for the most interesting, important, or noteworthy news, '
        'overall or for one company.\n'
        '- search_news: use it when the user asks for news about a specific topic or company.\n'
        '- If the user asks for news about a company without naming a topic, use rank_interesting with that '
        'company, so its most important articles come first. Use search_news when there is a topic.\n\n'
        'Rules:\n'
        '- Answer only from what the tools return. Never invent an article, a detail, or a URL.\n'
        '- For every article you mention, give the headline, the company, the date, and the URL.\n'
        '- When presenting ranked results, keep the order the tool returned and explain why each article matters '
        'to a law firm, using its score (1 to 10) and its reason.\n'
        '- If a tool returns no articles, or the articles do not answer the question, say so plainly.\n'
        '- If the user asks a follow-up about articles already discussed in this conversation, answer from the '
        'conversation without calling a tool.\n'
        '- Keep answers concise. Use a numbered list for multiple articles.\n'
        '- Only offer follow-ups the two tools can actually do: rank the news or search it by topic or company. '
        'You cannot fetch full article text, filings, or anything outside these articles.'
    )


    first_response = st.session_state.open_ai_client.chat.completions.create(
        model = model,
        messages = [{'role': 'system', 'content': system_prompt}] + window,
        tools = TOOLS,
        tool_choice = 'auto'
    )

    with st.chat_message('assistant'):
        msg = first_response.choices[0].message

        tool_calls = msg.tool_calls

        if tool_calls:

            messages = [{'role': 'system', 'content': system_prompt}] + window + [msg]

            #Iterating over tool calls
            for tool in tool_calls:

                #Extracting arguments
                args = json.loads(tool.function.arguments)

                if tool.function.name == 'rank_interesting':

                    #Extracting arguments
                    n = args['n']
                    company = args['company']

                    #Calling function
                    result = rank_interesting(n = n, company = company, score_field = score_field)

                elif tool.function.name == 'search_news':

                    #Extracting arguments
                    query = args['query']
                    company = args['company']
                    n = args['n']
                    
                    result = search_news(query = query, company = company, n = n, score_field = score_field)

                messages.append({'role': 'tool', 'tool_call_id': tool.id, 'content': result})

            stream = st.session_state.open_ai_client.chat.completions.create(
                model = model,
                messages = messages,
                stream = True
            )

            #Streamlit renders text between two $ signs as LaTeX, so escape them as the answer streams in
            def escape_dollars(stream):
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content.replace('$', r'\$')

            answer = st.write_stream(escape_dollars(stream))

        else:

            answer = (msg.content or '').replace('$', r'\$')

            st.write(answer)

        st.session_state.messages.append({'role': 'assistant', 'content': answer})

if st.sidebar.button('Clear conversation'):
    st.session_state.messages = st.session_state.messages[:1]
    st.rerun()
