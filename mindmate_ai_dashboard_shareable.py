import streamlit as st
import openai
import pandas as pd
import datetime
import hashlib
import sqlite3
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="MindMate AI 🧘", page_icon="💬")

# ================== OPENAI API KEY PROMPT ==================
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = None

if not st.session_state.OPENAI_API_KEY:
    st.info("Please enter your OpenAI API key to use MindMate AI.")
    user_key = st.text_input("OpenAI API Key", type="password")
    if user_key:
        st.session_state.OPENAI_API_KEY = user_key
        st.success("API key saved! You can now use the chatbot.")
        st.experimental_refresh()  # Updated for latest Streamlit

openai.api_key = st.session_state.OPENAI_API_KEY

# ================== DATABASE ==================
def init_db():
    conn = sqlite3.connect("mindmate_users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS moods (
            username TEXT,
            date TEXT,
            message TEXT,
            mood TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ================== AUTH FUNCTIONS ==================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = sqlite3.connect("mindmate_users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        st.success("Account created! Please log in.")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
    conn.close()

def verify_login(username, password):
    conn = sqlite3.connect("mindmate_users.db")
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result and result[0] == hash_password(password)

# ================== MOOD FUNCTIONS ==================
def add_mood(username, message, mood):
    conn = sqlite3.connect("mindmate_users.db")
    c = conn.cursor()
    c.execute("INSERT INTO moods VALUES (?, ?, ?, ?)",
              (username, str(datetime.date.today()), message, mood))
    conn.commit()
    conn.close()

def get_moods(username):
    conn = sqlite3.connect("mindmate_users.db")
    df = pd.read_sql_query("SELECT * FROM moods WHERE username = ?", conn, params=(username,))
    conn.close()
    return df

# ================== CHAT FUNCTIONS ==================
def detect_emotion(message):
    msg = message.lower()
    if any(word in msg for word in ["sad", "depressed", "lonely", "cry"]):
        return "sad"
    elif any(word in msg for word in ["anxious", "nervous", "worried", "stress"]):
        return "anxious"
    elif any(word in msg for word in ["angry", "mad", "furious", "irritated"]):
        return "angry"
    elif any(word in msg for word in ["happy", "grateful", "joy", "excited"]):
        return "happy"
    else:
        return "neutral"

def crisis_check(message):
    crisis_keywords = ["suicide", "kill myself", "die", "want to die", "end it all"]
    return any(word in message.lower() for word in crisis_keywords)

def ai_response(user_message):
    openai.api_key = st.session_state.OPENAI_API_KEY
    prompt = f"""
You are MindMate, a kind and supportive wellness chatbot.
You are NOT a therapist and do not diagnose.
User: {user_message}
MindMate:
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an empathetic wellness assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].message["content"].strip()

# ================== MAIN APP ==================
st.title("💬 MindMate AI – Your Wellness Companion")

if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.username:
    tab1, tab2 = st.tabs(["🔐 Login", "🆕 Sign Up"])
    
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if verify_login(username, password):
                st.session_state.username = username
                st.success(f"Welcome back, {username}! 🌿")
                st.experimental_refresh()
            else:
                st.error("Invalid credentials.")
    
    with tab2:
        new_username = st.text_input("Create Username")
        new_password = st.text_input("Create Password", type="password")
        if st.button("Register"):
            register_user(new_username, new_password)

else:
    page = st.sidebar.radio("🧭 Navigation", ["💬 Chat", "📝 Journal", "📊 Insights", "🚪 Logout"])

    if page == "💬 Chat":
        st.subheader("🧘 How are you feeling today?")
        user_input = st.text_input("You:", placeholder="Share what’s on your mind...")

        if user_input:
            if crisis_check(user_input):
                response = (
                    "I'm really sorry you're feeling like this. You deserve support and care. "
                    "If you’re in danger, please reach out for help — In the U.S., call or text **988**."
                )
            else:
                response = ai_response(user_input)

            mood = detect_emotion(user_input)
            add_mood(st.session_state.username, user_input, mood)
            st.markdown(f"**MindMate:** {response}")

    elif page == "📝 Journal":
        st.header("📝 My Journal")
        df = get_moods(st.session_state.username)

        if not df.empty:
            emotions = ["all"] + sorted(df["mood"].unique().tolist())
            mood_filter = st.selectbox("Filter by mood", emotions)
            keyword = st.text_input("Search for a keyword")

            filtered_df = df.copy()
            if mood_filter != "all":
                filtered_df = filtered_df[filtered_df["mood"] == mood_filter]
            if keyword:
                filtered_df = filtered_df[filtered_df["message"].str.contains(keyword, case=False, na=False)]

            st.dataframe(filtered_df.sort_values("date", ascending=False), hide_index=True)
            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Journal as CSV", csv, "journal.csv", "text/csv")
        else:
            st.info("No journal entries yet 💚")

    elif page == "📊 Insights":
        st.header("📊 Interactive Mood Insights")
        df = get_moods(st.session_state.username)

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            start_date, end_date = st.date_input("Select date range:", [df["date"].min(), df["date"].max()])
            filtered = df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]

            if filtered.empty:
                st.warning("No entries in this date range.")
            else:
                st.subheader("🥧 Mood Balance")
                mood_counts = filtered["mood"].value_counts().reset_index()
                fig_pie = px.pie(mood_counts, names="index", values="mood", color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)

                st.subheader("📈 Mood Trend Over Time")
                fig_line = px.scatter(filtered, x="date", y="mood", text="message",
                                      title="Mood Tracker", hover_data=["message"],
                                      color="mood", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_line.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
                st.plotly_chart(fig_line, use_container_width=True)

                st.subheader("☁️ Common Words")
                text = " ".join(filtered["message"].dropna())
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

                st.subheader("💡 AI Mood Insight")
                summary_prompt = f"""
Analyze emotional changes between {start_date} and {end_date} based on moods: {filtered['mood'].tolist()}.
Give a short (max 60 words), compassionate reflection about how the user is doing emotionally.
"""
                reflection = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You write kind emotional summaries for users."},
                        {"role": "user", "content": summary_prompt},
                    ],
                    max_tokens=100,
                    temperature=0.8,
                ).choices[0].message["content"]
                st.info(f"🪷 *{reflection}*")
        else:
            st.info("No mood data yet — start chatting to build insights 🌿")

    elif page == "🚪 Logout":
        st.session_state.username = None
        st.success("Logged out successfully.")
        st.experimental_refresh()

