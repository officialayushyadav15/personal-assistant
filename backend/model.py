import cohere
import re
import os
from functools import lru_cache
from typing import List
from dotenv import load_dotenv

class DecisionMaker:
    def __init__(self):
        self.co = None
        self._initialize_cohere()
        self.preamble = self._create_preamble()
        self.funcs = self._parse_functions_from_preamble()
        self.chat_history = self._create_chat_history()

    def _initialize_cohere(self):
        """Initialize Cohere client with API key from .env"""
        load_dotenv()
        api_key = os.getenv("CohereAPIKey")
        if not api_key:
            raise ValueError("CohereAPIKey not found in .env file")
        self.co = cohere.Client(api_key=api_key)

    def _create_preamble(self):
        return """You are a query classifier. Follow these rules strictly:

1. SYSTEM COMMANDS (highest priority):
   - Time/date requests
   - Volume control
   - Device operations
2. EXIT: Only respond with 'exit' for explicit goodbye messages
3. GREETINGS: Always classify as 'general'
4. MULTI-COMMANDS: Split requests using 'and' or commas
5. FUNCTION PRIORITY:
   - system
   - open/close [app]
   - play [song]
   - reminder [time+message]
   - generate image [description]
   - content [topic]
   - youtube/google search [query]
6. KNOWLEDGE:
   - realtime: People/companies, news, current events
   - general: Concepts, history, how-tos, greetings
7. PROPER NOUNS: Capitalized names → realtime
8. DEFAULT: general"""

    def _parse_functions_from_preamble(self):
        return [
            'system', 'exit', 'open', 'close', 'play',
            'reminder', 'generate image', 'content',
            'youtube search', 'google search', 'realtime', 'general'
        ]

    def _create_chat_history(self):
        return [
            {"role": "User", "message": "What time is it?"},
            {"role": "Chatbot", "message": "system time"},
            {"role": "User", "message": "What's today's date?"},
            {"role": "Chatbot", "message": "system date"},
            {"role": "User", "message": "Hello!"},
            {"role": "Chatbot", "message": "general hello"},
            {"role": "User", "message": "Open Chrome and Notepad"},
            {"role": "Chatbot", "message": "open chrome, open notepad"},
            {"role": "User", "message": "Who is Tim Cook?"},
            {"role": "Chatbot", "message": "realtime tim cook"},
        ]

    @lru_cache(maxsize=512)
    def classify_query(self, prompt: str) -> List[str]:
        try:
            response = self.co.chat(
                model='command-r-plus',
                message=self._sanitize_input(prompt),
                temperature=0.4,
                chat_history=self.chat_history,
                preamble=self.preamble
            ).text
            return self._process_response(response, prompt)
        except Exception as e:
            return [f"error: {str(e)}"]

    def _sanitize_input(self, text: str) -> str:
        return text.strip()[:500].replace('\n', ' ')

    def _process_response(self, response: str, original_prompt: str) -> List[str]:
        split_pattern = r'(?:\s+and\s+|\s*,\s*)(?=\b(?:{})\b)'.format('|'.join(self.funcs))
        commands = re.split(split_pattern, response.strip(), flags=re.IGNORECASE)
        
        valid_commands = []
        for cmd in commands:
            cmd = cmd.strip().lower()
            if any(cmd.startswith(f) for f in self.funcs):
                valid_commands.append(cmd)
        
        if not valid_commands and self._has_proper_noun(original_prompt):
            return [f"realtime {original_prompt.lower()}"]
            
        return valid_commands or [f"general {original_prompt.lower()}"]

    def _has_proper_noun(self, text: str) -> bool:
        return any(word.istitle() for word in text.split())

def main():
    try:
        dm = DecisionMaker()
        print("AI Classifier Ready. Type 'exit' to quit.\n")
        
        while True:
            try:
                user_input = input(">>> ").strip()
                if not user_input:
                    continue
                
                result = dm.classify_query(user_input)
                
                if 'exit' in result:
                    print("Goodbye!")
                    break
                
                print(f"[{'|'.join(result)}]")

            except KeyboardInterrupt:
                print("\nExiting...")
                break

    except Exception as e:
        print(f"Initialization failed: {str(e)}")

if __name__ == "__main__":
    main()