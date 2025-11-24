from langchain.schema import HumanMessage, SystemMessage

class prompt_generator(object):
    def __init__(self) -> None:
        pass

    def generate_openai_prompt(self,message='',system_message='',history=None):
        messages = []
        if system_message != '':
            system_message = SystemMessage(content=system_message)
            messages.append(system_message)
        message = HumanMessage(content=message)
        if history is not None:
            messages.extend(history)
        messages.append(message)
        return messages
    
    def generate_ernie_prompt(self,message='',history=None):
        messages = []
        if history is not None:
            for i,content in enumerate(history):
                messages.append({'role': 'user', 'content': content} if i%2==0 else {'role': 'assistant', 'content': content})
        messages.append({'role': 'user', 'content': message})
        return messages
    
    def generate_chatglm_prompt(self,message='',history=None):
        messages = []
        if history is not None:
            for i,content in enumerate(history):
                messages.append({'role': 'user', 'content': content} if i%2==0 else {'role': 'assistant', 'content': content})
        messages.append({'role': 'user', 'content': message})
        return messages
    
    def generate_spark_prompt(self,message='',history=None):
        messages = []
        if history is not None:
            for i,content in enumerate(history):
                messages.append({'role': 'user', 'content': content} if i%2==0 else {'role': 'assistant', 'content': content})
        messages.append({'role': 'user', 'content': message})
        return messages
    
    def generate_ali_prompt(self,message='',history=None):
        pre_history = []
        if history is not None:
            for i,content in enumerate(history):
                pre_history.append({'user': content} if i%2==0 else {'bot': content})
        return message,pre_history
    
    def generate_aihubmix_prompt(self,message='',history=None):
        messages = []
        if history is not None:
            for i,content in enumerate(history):
                messages.append({'role': 'user', 'content': content} if i%2==0 else {'role': 'assistant', 'content': content})
        messages.append({'role': 'user', 'content': message})
        return messages
