from langchain.prompts import PromptTemplate
from modules.agent.chatdb.chat import chat_with_ai,create_chat_completion
from langchain.prompts import PromptTemplate
from modules.agent.chatdb.sql_examples import egs
import re
import ast

prompt_ask_steps_temp = """
Please tell me what standard SQL statements should I use in order to respond to the "USER INPUT". \
If it needs multiple SQL operations on the database, please list them step by step concisely. \
If there is no need to use the database, reply to the "USER INPUT" directly.
The output should be a markdown code snippet formatted in the following schema, \
including the leading and trailing "\`\`\`" and "\`\`\`":
```
Step1: <Description of first step>
SQL command for step1

Step2: <Description of second step>
SQL command for step2

......
```
Here are some examples:
{egs}

USER INPUT: {user_inp}
ANSWER:
"""

prompt_ask_steps = PromptTemplate(
    template=prompt_ask_steps_temp,
    input_variables=["user_inp"],
    partial_variables={
        "egs": '\n'.join(egs),
    }
)






def need_update_sql(input_string):
    pattern = r"<\S+>"
    matches = re.findall(pattern, input_string)
    return matches

# This is a magic function that can do anything with no-code. See
# https://github.com/Torantulino/AI-Functions for more info.
def call_ai_function(function, args, description, model=None):
    """Call an AI function"""
    if model is None:
        model = 'gpt-3.5-turbo'
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    args = ", ".join(args)
    messages = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}\n{function}```\n\nOnly respond with your `return` value. Do not include any other explanatory text in your response.",
        },
        {"role": "user", "content": args},
    ]
    response = create_chat_completion(
        model=model, messages=messages
    )

    return response


def populate_sql_statement(sql_str: str, previous_sql_results):
    # Try to fix the SQL using GPT:
    function_string = "def populate_sql_statement(sql_str: str, previous_sql_results: list[list[dict]]) -> list[str]:"
    args = [f"'''{sql_str}'''", f"'''{previous_sql_results}'''"]
    description_string = "Find useful information in the results of the previous sql statement, and replace <> with the corresponding information."

    result_string = call_ai_function(
        function_string, args, description_string, model='gpt-3.5-turbo'
    )
    brace_index = result_string.index("[")
    result_string = result_string[brace_index:]
    last_brace_index = result_string.rindex("]")
    result_string = result_string[:last_brace_index + 1]
    list_of_str = ast.literal_eval(result_string)
    return list_of_str



def get_steps_from_response(response):
    # Regular expression patterns to extract step number, description, and SQL query
    pattern = r"Step(\d+):\s+(.*?)\n`(.*?)`"
    matches = re.findall(pattern, response, re.DOTALL)

    # Extract information and create list of dictionaries
    result = []
    for match in matches:
        step_number = int(match[0])
        description = match[1]
        sql_query = match[2]
        # print(sql_query+'\n')
        result.append({
            "id": step_number,
            "description": description.strip(),
            "sql": sql_query.strip(),
        })

    return result


def init_system_msg(table_details=None):
    sys_temp = """
You are ChatDB, a powerful AI assistant, a variant of ChatGPT that can utilize databases as external symbolic memory. \
You are an expert in databases, proficient in SQL statements and can use the database to help users. \
The details of tables in the database are delimited by triple quotes.
\"\"\"
{table_details}
\"\"\"
"""
    sys_prompt = PromptTemplate(
        template=sys_temp,
        input_variables=[],
        partial_variables={"table_details": table_details}
    )
    sys_prompt_str = sys_prompt.format()
    return sys_prompt_str


def chain_of_memory(sql_steps, mysql_db):
    num_step = len(sql_steps)
    sql_results_history = []
    new_mem_ops = []
    output_sql = ""
    for i in range(num_step):
        cur_step = sql_steps[i]
        ori_sql_cmd = cur_step['sql']
        # print(f"\nStep{cur_step['id']}: {cur_step['description']}\n")
        if need_update_sql(ori_sql_cmd):
            list_of_sql_str = populate_sql_statement(
                ori_sql_cmd, sql_results_history)
            # print(ori_sql_cmd)
            new_mem_ops.append(list_of_sql_str)
            for sql_str in list_of_sql_str:
                # print(f"Execute: \n{sql_str}\n")
                sql_results, sql_res_str = mysql_db.execute_sql(sql_str)
                output_sql = sql_str
                # print(f"Database response:\n{sql_res_str}\n")
                if sql_results:
                    sql_results_history.append(sql_results)
        else:
            # print(f"Execute: \n{ori_sql_cmd}\n")
            sql_results, sql_res_str = mysql_db.execute_sql(ori_sql_cmd)
            output_sql = ori_sql_cmd
            new_mem_ops.append([ori_sql_cmd])
            # print(f"Database response:\n{sql_res_str}\n")
            if sql_results:
                sql_results_history.append(sql_results)
    return sql_results_history, new_mem_ops, sql_res_str, sql_results, output_sql


def generate_chat_responses(user_inp, mysql_db, historical_message, table_details=None, llm=None, llm_name=None):
    # ask steps
    prompt_ask_steps_str = prompt_ask_steps.format(user_inp=user_inp)
    response_steps = chat_with_ai(init_system_msg(table_details), prompt_ask_steps_str, historical_message, None,
                                  token_limit=4096, llm=llm, llm_name=llm_name)
    if response_steps['status'] == -1:
        return response_steps, 'no results found.', None

    response_steps_list_of_dict = get_steps_from_response(
        response_steps['message'])

    if len(response_steps_list_of_dict) == 0:
        msg = response_steps['message']
        return {
            'status': 0,
            'message': f"NOT NEED MEMORY: {msg}"
        }, 'no results found.', None

    sql_results_history, new_mem_ops, response, sql_results, output_sql = chain_of_memory(
        response_steps_list_of_dict, mysql_db)
    return {
        'status': 0,
        'message': response
    }, sql_results, output_sql


