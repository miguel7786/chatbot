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
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")
    retriever = vector_store.as_retriever()
    results = retriever.invoke(query.strip())  # Ensure clean string
    context = "\n".join([doc.page_content for doc in results])
    return context

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
            st.sidebar.success("Signed in successfully!")
            time.sleep(2)
        except firebase_admin.auth.UserNotFoundError:
            st.sidebar.error("Account does not exist. Please sign up first.")

    # Sign Up
    if sign_up and email and password:
        try:
            user = auth.create_user(email=email, password=password)
            st.session_state.user = {"email": email, "uid": user.uid}
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
            placeholder="Message the Support Assistant",
            key="chat_input",
            label_visibility="collapsed"
        )

    with col2:
        send_button = st.button("➤")  # Emoji for Send Button

    if send_button and user_input.strip():
        user_input = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        try:
            # Retrieve relevant context from the vector store
            context = get_context_from_vector_store(user_input)

            # Combine context, role definition, and user input to send to OpenAI assistant
            assistant_prompt = f"{ROLE_DEFINITION}\n\nContext:\n{context}\n\nUser Query:\n{user_input}"

            # Call OpenAI Assistant API
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": assistant_prompt}]
            )

            # Get the assistant's response
            bot_reply = response.choices[0].message.content

            # Append to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

            # Save chat history to Firestore
            db.collection("users").document(st.session_state.user["uid"]).set(
                {"chat_history": st.session_state.chat_history}, merge=True
            )

            st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.title("Please sign in to access the chat.")
