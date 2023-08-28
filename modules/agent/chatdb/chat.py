import time
import openai
import tiktoken
from typing import List, Dict
import openai
from langchain.schema import HumanMessage, SystemMessage

def create_chat_completion(messages, model=None, llm_name=None) -> str:
    """Create a chat completion using the OpenAI API"""
    response = None
    num_retries = 5
    for attempt in range(num_retries):
        try:
            if llm_name == 'openai' or llm_name == 'azure openai':
                response = model.get_ones([SystemMessage(content=messages[0]['content']),HumanMessage(content=messages[1]['content'])])
            elif llm_name == 'ernie bot' or llm_name == 'ernie bot turbo' or llm_name == 'chatglm api' or llm_name == 'spark api' or llm_name == 'ali api':
                response = model.get_ones(messages[0]['content']+'\n'+messages[1]['content'])
            else:
                for response,_ in model._call(messages[0]['content']+'\n'+messages[1]['content'],history=[],streaming=False):
                    pass
                response = {
                    'status':0,
                    'message':response
                }
            break
        except openai.error.RateLimitError:
            time.sleep(20)
        except openai.error.APIError as e:
            if e.http_status == 502:
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

def count_message_tokens(messages : List[Dict[str, str]], model : str = "gpt-3.5-turbo-0301") -> int:
    """
    Returns the number of tokens used by a list of messages.

    Args:
    messages (list): A list of messages, each of which is a dictionary containing the role and content of the message.
    model (str): The name of the model to use for tokenization. Defaults to "gpt-3.5-turbo-0301".

    Returns:
    int: The number of tokens used by the list of messages.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        # !Node: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return count_message_tokens(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # !Note: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return count_message_tokens(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def count_string_tokens(string: str, model_name: str = "gpt-3.5-turbo-0301") -> int:
    """
    Returns the number of tokens in a text string.

    Args:
    string (str): The text string.
    model_name (str): The name of the encoding to use. (e.g., "gpt-3.5-turbo")

    Returns:
    int: The number of tokens in the text string.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def create_chat_message(role, content):
    """
    Create a chat message with the given role and content.

    Args:
    role (str): The role of the message sender, e.g., "system", "user", or "assistant".
    content (str): The content of the message.

    Returns:
    dict: A dictionary containing the role and content of the message.
    """
    return {"role": role, "content": content}


def generate_context(prompt, relevant_memory, full_message_history, model):
    current_context = [
        create_chat_message(
            "system", prompt),
        # create_chat_message(
        #     "system", f"The current time and date is {time.strftime('%c')}"),
        # create_chat_message(
        #     "system", f"This reminds you of these events from your past:\n{relevant_memory}\n\n"),
    ]

    # Add messages from the full message history until we reach the token limit
    next_message_to_add_index = len(full_message_history) - 1
    insertion_index = len(current_context)
    # Count the currently used tokens
    current_tokens_used = count_message_tokens(current_context, model)
    return next_message_to_add_index, current_tokens_used, insertion_index, current_context


# TODO: Change debug from hardcode to argument
def chat_with_ai(
        prompt,
        user_input,
        full_message_history,
        permanent_memory,
        token_limit,
        llm=None,
        llm_name=None):
    """Interact with the OpenAI API, sending the prompt, user input, message history, and permanent memory."""
    while True:
        try:
            """
            Interact with the OpenAI API, sending the prompt, user input, message history, and permanent memory.

            Args:
            prompt (str): The prompt explaining the rules to the AI.
            user_input (str): The input from the user.
            full_message_history (list): The list of all messages sent between the user and the AI.
            permanent_memory (Obj): The memory object containing the permanent memory.
            token_limit (int): The maximum number of tokens allowed in the API call.

            Returns:
            str: The AI's response.
            """
            model = 'gpt-3.5-turbo'  # TODO: Change model from hardcode to argument
            # Reserve 1000 tokens for the response

            send_token_limit = token_limit - 1000

            # relevant_memory = '' if len(full_message_history) ==0 else  permanent_memory.get_relevant(str(full_message_history[-9:]), 10)

            # logger.debug(f'Memory Stats: {permanent_memory.get_stats()}')
            relevant_memory = None

            next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
                prompt, relevant_memory, full_message_history, model)

            # while current_tokens_used > 2500:
            #     # remove memories until we are under 2500 tokens
            #     relevant_memory = relevant_memory[1:]
            #     next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
            #         prompt, relevant_memory, full_message_history, model)

            current_tokens_used += count_message_tokens([create_chat_message("user", user_input)], model) # Account for user input (appended later)

            while next_message_to_add_index >= 0:
                # print (f"CURRENT TOKENS USED: {current_tokens_used}")
                message_to_add = full_message_history[next_message_to_add_index]

                tokens_to_add = count_message_tokens([message_to_add], model)
                if current_tokens_used + tokens_to_add > send_token_limit:
                    break

                # Add the most recent message to the start of the current context, after the two system prompts.
                current_context.insert(insertion_index, full_message_history[next_message_to_add_index])

                # Count the currently used tokens
                current_tokens_used += tokens_to_add

                # Move to the next most recent message in the full message history
                next_message_to_add_index -= 1

            # Append user input, the length of this is accounted for above
            current_context.extend([create_chat_message("user", user_input)])

            # Calculate remaining tokens
            tokens_remaining = token_limit - current_tokens_used
            # assert tokens_remaining >= 0, "Tokens remaining is negative. This should never happen, please submit a bug report at https://www.github.com/Torantulino/Auto-GPT"

            # Debug print the current context
            # logger.debug(f"Token limit: {token_limit}")
            # logger.debug(f"Send Token Count: {current_tokens_used}")
            # logger.debug(f"Tokens remaining for response: {tokens_remaining}")
            # logger.debug("------------ CONTEXT SENT TO AI ---------------")
            # for message in current_context:
            #     # Skip printing the prompt
            #     # if message["role"] == "system" and message["content"] == prompt:
            #     #     continue
            #     logger.debug(f"{message['role'].capitalize()}: {message['content']}")
            #     logger.debug("")
            # logger.debug("----------- END OF CONTEXT ----------------")

            # TODO: use a model defined elsewhere, so that model can contain temperature and other settings we care about
            # print(current_context)
            assistant_reply = create_chat_completion(
                model=llm,
                llm_name=llm_name,
                messages=current_context,
            )

            if assistant_reply['status'] == 0:
                # Update full message history
                full_message_history.append(create_chat_message("user", user_input))
                full_message_history.append(create_chat_message("assistant", assistant_reply['message']))
                # logger.debug(f"{full_message_history[-1]['role'].capitalize()}: {full_message_history[-1]['content']}")
                # logger.debug("----------- END OF RESPONSE ----------------")
            return assistant_reply
        except openai.error.RateLimitError:
            # TODO: When we switch to langchain, this is built in
            print("Error: ", "API Rate Limit Reached. Waiting 10 seconds...")
            time.sleep(10)