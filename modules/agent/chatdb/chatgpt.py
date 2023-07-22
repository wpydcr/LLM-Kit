import time
import openai
from modules.model.use_api import openai_api
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from colorama import Fore
from modules.agent.chatdb.config import cfg


def create_chat_completion(messages, model=None, llm_name=None,temperature=cfg.temperature, max_tokens=None) -> str:
    """Create a chat completion using the OpenAI API"""
    response = None
    num_retries = 5
    for attempt in range(num_retries):
        try:
            if llm_name == 'openai' or llm_name == 'azure openai':
                response = model.get_ones_openai([SystemMessage(content=messages[0]['content']),HumanMessage(content=messages[1]['content'])])
            elif llm_name == 'ernie bot' or llm_name == 'ernie bot turbo' or llm_name == 'chatglm api' or llm_name == 'spark api':
                response = model.get_ones_openai(messages[0]['content']+'\n'+messages[1]['content'])
            else:
                for response,_ in model._call(messages[0]['content']+'\n'+messages[1]['content'],history=[],streaming=False):
                    pass
            break
        except openai.error.RateLimitError:
            if cfg.debug_mode:
                print(Fore.RED + "Error: ", "API Rate Limit Reached. Waiting 20 seconds..." + Fore.RESET)
            time.sleep(20)
        except openai.error.APIError as e:
            if e.http_status == 502:
                if cfg.debug_mode:
                    print(Fore.RED + "Error: ", "API Bad gateway. Waiting 20 seconds..." + Fore.RESET)
                time.sleep(20)
            else:
                raise
            if attempt == num_retries - 1:
                raise
        except openai.error.InvalidRequestError:
            raise

    if response is None:
        raise RuntimeError("Failed to get response after 5 retries")

    return response
