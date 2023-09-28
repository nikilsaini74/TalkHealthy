import socket

import requests
import streamlit as st
from streamlit_chat import message
from urllib3.connection import HTTPConnection

from scripts.app_environment import api_base_url

st.set_page_config(
    layout="centered",
    page_title="Fitness-Bot"
)

st.markdown("""
<style>
.small-font {
    font-size:0.8rem !important;
}
</style>
""", unsafe_allow_html=True)


def initialize_state():
    if 'db_states' not in st.session_state:
        st.session_state['db_states'] = {}

    if 'current_db' not in st.session_state:
        st.session_state['current_db'] = ''

    if 'selected_database' not in st.session_state:
        st.session_state['selected_database'] = ''

    if 'selected_collection' not in st.session_state:
        st.session_state['selected_collection'] = ''


@st.cache_data
def get_database_names_and_collections():
    endpoint = f"{api_base_url}/databases"
    response = requests.get(endpoint)
    if response.status_code == 200:
        result = response.json()
        databases = {db['database_name']: [col['name'] for col in db['collections']] for db in result}
        return databases
    else:
        st.error("Failed to get database names.")
        st.write(response.text)
        return {}


databases = get_database_names_and_collections()


def query_documents(question: str, database_name: str, collection_name: str):
    with st.spinner("Processing..."):
        endpoint = f"{api_base_url}/query"
        data = {
            "question": question,
            "database_name": database_name
        }

        # Modify socket options for the HTTPConnection class
        set_keepalive_options(HTTPConnection)

        response = requests.post(endpoint, json=data)
        if response.status_code == 200:
            result = response.json()
            answer = result["answer"]
            source_documents = result["source_documents"]
            return answer, source_documents
        else:
            st.error("Failed to query documents.")
            st.write(response.text)


def handle_database_and_collection_selection():
    database_names = list(databases.keys())
    col1, col2 = st.columns(2)
    st.session_state['selected_database'] = database_names[0]

    # Make sure that a database is selected
    if st.session_state['selected_database']:
        collections = databases.get(st.session_state['selected_database'], [])
        st.session_state['selected_collection'] = collections[0]


def handle_user_query():
    user_input = st.text_input("Query:", placeholder="Ask questions about Fitness, Gym, Diet", key='input')
    submit_button = st.button(label='Send')

    if submit_button and user_input:
        handle_user_query_processing(user_input)


def handle_user_query_processing(user_input):
    # Access selected_database and selected_collection from session_state
    selected_database = st.session_state['selected_database']
    selected_collection = st.session_state['selected_collection']
    # Create a new container for the conversation turn
    turn_container = st.container()
    with turn_container:
        # Immediately show the question
        message_key = str(len(st.session_state['db_states'][selected_database]['history'])) + '_user_qa_next'
        message(user_input, is_user=True, key=message_key, avatar_style="bottts", seed=3)
        # Append the question to the history
        st.session_state['db_states'][selected_database]['history'].append({
            'text': user_input,
            'is_user': True,
            'key': message_key
        })
        # Then wait for the answer
        answer, source_documents = query_documents(user_input, selected_database, selected_collection)
        # Append to the history
        answer_key = str(len(st.session_state['db_states'][selected_database]['history'])) + '_gen_next'
        st.session_state['db_states'][selected_database]['history'].append({
            'text': answer,
            'is_user': False,
            'key': answer_key
        })
        st.session_state['db_states'][selected_database]['source_documents'] = [
            source_documents]  # replace old source_documents
        # Display the answer
        message(answer, key=answer_key, avatar_style="bottts", seed=5)

        # Display the source documents in the same container
        st.write("source:")
        # for idx, doc in enumerate(st.session_state['db_states'][selected_database]['source_documents'][0]):
        #     message(doc["link"], key=answer_key + '_' + str(idx) + '_link', avatar_style="bottts", seed=5)
        #     message(doc["content"], key=answer_key + '_' + str(idx) + '_content', avatar_style="bottts", seed=5)
        for idx, doc in enumerate(st.session_state['db_states'][selected_database]['source_documents'][0]):
            st.write(f'> {doc["link"]}:')
            st.markdown(f'<p class="small-font">{doc["content"]}</p>', unsafe_allow_html=True)


def set_keepalive_options(http_conn):
    http_conn.default_socket_options = (
        http_conn.default_socket_options + [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        ])

    if hasattr(socket, 'TCP_KEEPIDLE'):
        http_conn.default_socket_options += [(socket.SOL_TCP, socket.TCP_KEEPIDLE, 45)]

    if hasattr(socket, 'TCP_KEEPINTVL'):
        http_conn.default_socket_options += [(socket.SOL_TCP, socket.TCP_KEEPINTVL, 10)]

    if hasattr(socket, 'TCP_KEEPCNT'):
        http_conn.default_socket_options += [(socket.SOL_TCP, socket.TCP_KEEPCNT, 6)]


def redraw_conversation():
    selected_database = st.session_state['selected_database']
    for msg in st.session_state['db_states'][selected_database]['history']:
        if msg['is_user']:
            message(msg['text'], is_user=True, key=msg['key'], avatar_style="bottts", seed=3)
        else:
            message(msg['text'], key=msg['key'], avatar_style="bottts", seed=5)

    source_documents = st.session_state['db_states'][selected_database].get('source_documents', [])
    if source_documents:  # Check if there are any source documents
        st.write("source:")
        for idx, doc in enumerate(source_documents[0]):
            st.write(f'> {doc["link"]}:')
            st.write(doc["content"])


def main():
    initialize_state()
    print(st.session_state)

    st.header("Fitness on finger-tips")
    # Query section
    handle_database_and_collection_selection()
    selected_database = st.session_state['selected_database']

    clear_history_button = st.button("Clear History")

    if clear_history_button:
        # Clear the history for the selected database
        if selected_database in st.session_state['db_states']:
            st.session_state['db_states'][selected_database] = {
                'history': [],  # This will store both questions and answers
                'source_documents': []
            }

    if selected_database:
        if selected_database != st.session_state['current_db']:
            st.session_state['current_db'] = selected_database
            # Ensure the selected database has an entry in the session state
            if selected_database not in st.session_state['db_states']:
                st.session_state['db_states'][selected_database] = {
                    'history': [],  # This will store both questions and answers
                    'source_documents': []
                }
            redraw_conversation()
        else:
            # If the database didn't change, display the history
            redraw_conversation()

        handle_user_query()


if __name__ == "__main__":
    main()
