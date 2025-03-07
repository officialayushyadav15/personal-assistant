import time
import requests
from groq import Groq
from json import load, dump
import datetime
from dotenv import dotenv_values

env_vars = dotenv_values(".env")

Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")
SerpAPIKey = env_vars.get("SerpAPIKey")

client = Groq(api_key=GroqAPIKey)

System = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which has real-time up-to-date information from the internet.
*** Provide Answers In a Professional Way, make sure to add full stops, commas, question marks, and use proper grammar.***
*** Just answer the question from the provided data in a professional way. ***"""

try:
    with open(r"Data\ChatLog.json", "r") as f:
        messages = load(f)
except:
    with open(r"Data\ChatLog.json", "w") as f:
        dump([], f)

try:
    with open(r"Data\SearchCount.json", "r") as f:
        search_count_data = load(f)
except:
    search_count_data = {"month": datetime.datetime.now().strftime("%Y-%m"), "count": 0}
    with open(r"Data\SearchCount.json", "w") as f:
        dump(search_count_data, f)

def GoogleSearch(query):
    try:
        params = {
            "q": query,
            "api_key": SerpAPIKey,
            "engine": "google"
        }
        response = requests.get("https://serpapi.com/search", params=params)
        if response.status_code == 200:
            results = response.json().get("organic_results", [])
            answer = f"The search results for '{query}' are: \n[start]\n"

            for i in results:
                answer += f"Title: {i.get('title')}\nDescription: {i.get('snippet')}\n\n"

            answer += "[end]"
            return answer
        else:
            return f"Error: Unable to fetch search results (Status Code: {response.status_code})"
    except Exception as e:
        return f"Error during Google search: {e}"

def AnswerModifier(Answer):
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    modified_answer = '\n'.join(non_empty_lines)
    return modified_answer

SystemChatBot = [
    {"role": "system", "content": System},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello, how can I help you? "}
]

def information():
    current_date_time = datetime.datetime.now()
    day = current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year = current_date_time.strftime("%Y")
    hour = current_date_time.strftime("%H")
    minute = current_date_time.strftime("%M")
    second = current_date_time.strftime("%S")
    data = f"use This Real-time Information if needed: \n"
    data += f"Day: {day}\n"
    data += f"Date: {date}\n"
    data += f"Month: {month}\n"
    data += f"Year: {year}\n"
    data += f"Time: {hour} hours, {minute} minutes, {second} seconds.\n"
    return data

def check_search_limit():
    current_month = datetime.datetime.now().strftime("%Y-%m")
    with open(r"Data\SearchCount.json", "r") as f:
        search_count_data = load(f)

    if search_count_data["month"] != current_month:
        search_count_data["month"] = current_month
        search_count_data["count"] = 0

    if search_count_data["count"] >= 100:
        return False

    search_count_data["count"] += 1
    with open(r"Data\SearchCount.json", "w") as f:
        dump(search_count_data, f, indent=4)

    return True

def RealtimeSearchEngine(prompt):
    global SystemChatBot, messages

    if not check_search_limit():
        return "Monthly search limit reached. Please try again on the first of next month."

    with open(r"Data\ChatLog.json", "r") as f:
        messages = load(f)

    messages.append({"role": "user", "content": f"{prompt}"})

    search_results = GoogleSearch(prompt)
    SystemChatBot.append({"role": "system", "content": search_results})

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=SystemChatBot + [{"role": "system", "content": information()}] + messages,
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        stream=True,
        stop=None
    )

    Answer = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            Answer += chunk.choices[0].delta.content

    Answer = Answer.strip().replace("</s>", "")
    messages.append({"role": "assistant", "content": Answer})

    with open(r"Data\ChatLog.json", "w") as f:
        dump(messages, f, indent=4)

    SystemChatBot.pop()
    return AnswerModifier(Answer=Answer)

if __name__ == "__main__":
    while True:
        prompt = input("Enter your query: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Goodbye! Have a great day!")
            break
        print(RealtimeSearchEngine(prompt))
        time.sleep(10)