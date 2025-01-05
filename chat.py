import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

file_path=os.path.abspath("intents.json")
with open(file_path,"r") as file:
    intents=json.load(file)

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot Function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
# print("Chatbot is now online! Type 'exit' to end the conversation.")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         print("Chatbot: Goodbye! Have a great day!")
#         break
#     # The line below was changed
#     response = chatbot(user_input) # Call the defined function 'chatbot_response' instead of 'get_response'
#     print("Chatbot:", response)

counter=0

def main():
    global counter
    st.title("Intents of Chatbot using NLP")
    
    # Create a sidebar menu with options 
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the chat")

        # Check if the chat log.csv file exists, and if not, create it with column names 
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file 
            with open('chat_log.csv', 'a', newline='', encoding="utf-8") as csvfile: 
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()
    
    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        with st.expander("Click to see Conversation History"): 
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
    
    # About Menu
    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user queries.")
        st.subheader("Project Overview:")
        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm are used to train the chatbot.
        2. Streamlit web framework is used to build the chatbot interface.
        """)
        st.subheader("Dataset:")
        st.write("""
        The dataset consists of labeled intents and entities. Intents represent the purpose of the user query, while entities represent key data points extracted from the text.
        """)
        st.subheader("Conclusion:")
        st.write("This chatbot demonstrates effective use of NLP and machine learning to understand user input and provide relevant responses.")

if __name__=='__main__':
    main()