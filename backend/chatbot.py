from groq import Groq
import re
import os
import json
import datetime
from dotenv import dotenv_values

env_vars = dotenv_values(".env")
Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")

client = Groq(api_key=GroqAPIKey)

class EnhancedAssistant:
    def __init__(self):
        self.command_prompt = """Classify queries into:
1. open [app] - Open applications
2. close [app] - Close applications  
3. system time - Get current time
4. system date - Get current date
5. realtime [query] - Current information
6. greeting - Simple hello
7. exit - End conversation
Respond ONLY with the command format."""

        self.main_prompt = f"""You are {Assistantname}, a concise AI assistant. Follow:
- Answer in plain text (NO markdown)
- Keep responses under 3 sentences
- Current context: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"""
        
        self.load_history()

    def load_history(self):
        try:
            with open("Data/ChatLog.json", "r") as f:
                self.history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = []

    def save_history(self):
        with open("Data/ChatLog.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def classify_query(self, query):
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": self.command_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.2,
                max_tokens=50
            )
            return response.choices[0].message.content.lower().strip()
        except Exception as e:
            print(f"Classification error: {e}")
            return "general"

    def execute_command(self, command):
        
        if command.startswith("system time"):
            return datetime.datetime.now().strftime("%H:%M:%S")
        if command.startswith("system date"):
            return datetime.datetime.now().strftime("%Y-%m-%d")
            
        if command == "greeting":
            return "Hello! How can I assist you today?"
            
        return None

    def generate_response(self, query):
        try:

            command = self.classify_query(query)
            response = self.execute_command(command)
            
            if response:
                return response

            messages = [
                {"role": "system", "content": self.main_prompt},
                *self.history[-4:],  # Keep last 2 exchanges
                {"role": "user", "content": query}
            ]
            
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0.7,
                max_tokens=300,
                stream=True
            )
            
            response = "".join([chunk.choices[0].delta.content 
                              for chunk in completion if chunk.choices[0].delta.content])
            
            self.history.extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ])
            self.save_history()
            
            return response.strip()
            
        except Exception as e:
            print(f"Error: {e}")
            return "Apologies, I encountered an error. Please try again."

if __name__ == "__main__":
    assistant = EnhancedAssistant()
    print(f"{Assistantname}: Hello! How can I help you today?")
    
    while True:
        try:
            user_input = input(f"{Username}: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print(f"{Assistantname}: Goodbye! Have a great day!")
                break
                
            response = assistant.generate_response(user_input)
            print(f"\n{Assistantname}: {response}\n")
            
        except KeyboardInterrupt:
            print(f"\n{Assistantname}: Session ended abruptly.")
            break