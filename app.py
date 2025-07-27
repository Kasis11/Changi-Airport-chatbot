import streamlit as st
import requests

API_URL = "https://changi-airport-chatbot.onrender.com/ask"  # ✅ correct

st.set_page_config(page_title="Changi Airport Chatbot", page_icon="🛫")
st.title("🛫 Changi Airport AI Chatbot")
st.markdown("Ask anything based on Changi Airport and Jewel Changi website content.")

query = st.text_input("Ask your question:", placeholder="e.g. What are the facilities in Terminal 3?")

if query:
    with st.spinner("Thinking..."):
        response = requests.post(API_URL, json={"question": query})
        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]
            if data.get("source_url"):
                answer += f"\n\n🔗 [Read more here]({data['source_url']})"
            st.success(answer)

            with st.expander("Source Documents (Details)"):
                for i, doc in enumerate(data["sources"]):
                    st.markdown(f"""
                    **Source {i+1}**  
                    🔗 [{doc['url']}]({doc['url']})  
                    _Excerpt:_  
                    > {doc['excerpt']} ...
                    """)
        else:
            st.error("Something went wrong. Please try again.")
