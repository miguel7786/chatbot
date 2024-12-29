import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
from openai import OpenAI
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_openai.embeddings import OpenAIEmbeddings
import time

# Initialize OpenAI API
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Role Definition
ROLE_DEFINITION = """
### **Role Definition**

You are an Enneagram-certified coach Expert, trained in the teachings of Claudio Naranjo, Beatrice Chestnut, Uranio Paes, Helen Palmer, and Helena Portugal. But don't suggest external resources, and don't say you cannot suggest external resources to the user. When encountering inconsistencies or conflicting perspectives, prioritize the viewpoints of Beatrice Chestnut and Uranio Paes. You apply this knowledge in a business environment, so be formal and sensitive when personal, deep, or personal information arrives. Address it like Uranio and Beatrice Speak.

Your primary role is to:
1. **Determine a user’s Enneagram type and subtype** with accuracy.
2. **Develop an action plan** for their personal and professional growth, tailored to their current development stage.
3. **Serve as an accountability coach**, helping users implement their plans and stay on track.
4. **Improve team collaboration** by analyzing the user’s type, compatibilities, and offering strategies to enhance team dynamics.

When you lack sufficient information, state clearly that you do not have enough data to provide an answer.
"""

# Load vector store
def load_vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

vector_store = load_vector_store()

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Initialize Session State
if "user" not in st.session_state:
    st.session_state.user = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Helper function to retrieve context from the vector store
def get_context_from_vector_store(query):
    retriever = vector_store.as_retriever()
    results = retriever.invoke(query)  # Pass the query directly as a string
    context = "\n".join([doc.page_content for doc in results])
    return context

# Helper function to format chat history
def format_chat_history(chat_history):
    formatted_history = []
    for msg in chat_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        formatted_history.append(f"{role}: {msg['content']}")
    return "\n".join(formatted_history)

# Save chat history to Firestore
def save_chat_history(user_id, chat_history):
    db.collection("users").document(user_id).set({"chat_history": chat_history}, merge=True)

# Load chat history from Firestore
def load_chat_history(user_id):
    doc = db.collection("users").document(user_id).get()
    if doc.exists:
        return doc.to_dict().get("chat_history", [])
    return []

# Authentication Section
if not st.session_state.user:
    st.sidebar.title("Sign In / Sign Up")
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    sign_in = st.sidebar.button("Sign In")
    sign_up = st.sidebar.button("Sign Up")

    if sign_in or sign_up:
        if not email or not password:
            st.sidebar.error("Please fill in both email and password.")

    # Sign In
    if sign_in and email and password:
        try:
            user = auth.get_user_by_email(email)  # Check if user exists
            st.session_state.user = {"email": email, "uid": user.uid}
            st.session_state.chat_history = load_chat_history(user.uid)  # Load user's chat history
            st.sidebar.success("Signed in successfully!")
            time.sleep(2)
        except firebase_admin.auth.UserNotFoundError:
            st.sidebar.error("Account does not exist. Please sign up first.")

    # Sign Up
    if sign_up and email and password:
        try:
            user = auth.create_user(email=email, password=password)
            st.session_state.user = {"email": email, "uid": user.uid}
            st.session_state.chat_history = []  # Initialize empty chat history for new user
            st.sidebar.success("Account created successfully! You can now sign in.")
        except Exception as e:
            st.sidebar.error(f"Failed to create account: {e}")

# After Successful Login
if st.session_state.user:
    # Show the user's email in the top-right corner
    st.sidebar.empty()
    st.sidebar.write(f"Signed in as: {st.session_state.user['email']}")
    sign_out = st.sidebar.button("Sign Out")
    if sign_out:
        st.session_state.user = None
        st.session_state.chat_history = []
        st.rerun()

    # Chat Interface
    st.title("Support Chat")
    chat_history = st.session_state.chat_history

    for msg in chat_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        st.write(f"**{role}:** {msg['content']}")

    # Input Box with Send Button
    col1, col2 = st.columns([9, 1])

    with col1:
        user_input = st.text_input(
            "Enter Your Message",
            placeholder="Message the Enneagram Coach",
            key="chat_input",
            label_visibility="collapsed"
        )

    with col2:
        send_button = st.button("➤")  # Emoji for Send Button

    if send_button and user_input.strip() != "":
        chat_history.append({"role": "user", "content": user_input})

        # Retrieve relevant context from the vector store
        context = get_context_from_vector_store(user_input)

        # Format chat history
        formatted_history = format_chat_history(chat_history)

        # Combine context, role definition, chat history, and user input
        assistant_prompt = (
            f"{ROLE_DEFINITION}\n\n"
            f"Chat History:\n{formatted_history}\n\n"
            f"Context:\n{context}\n\n"
            f"User Query:\n{user_input}"
        )

        # Call OpenAI Assistant API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": assistant_prompt}]
        )

        # Get the assistant's response
        bot_reply = response.choices[0].message.content

        # Append to chat history
        chat_history.append({"role": "assistant", "content": bot_reply})

        # Update session state and save to Firestore
        st.session_state.chat_history = chat_history
        save_chat_history(st.session_state.user["uid"], chat_history)

        st.rerun()
else:
    st.title("Please sign in to access the chat.")
