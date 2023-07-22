import json, re, time
from modules.agent.chatdb.mysql import MySQLDB
from modules.agent.chatdb.config import cfg
from modules.agent.chatdb.chatdb_prompts import prompt_ask_steps, prompt_ask_steps_no_egs
from langchain.prompts import PromptTemplate
from modules.agent.chatdb.call_ai_function import populate_sql_statement
from modules.agent.chatdb.chat import chat_with_ai


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
            list_of_sql_str = populate_sql_statement(ori_sql_cmd, sql_results_history)
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
    return sql_results_history, new_mem_ops,sql_res_str,sql_results, output_sql


def generate_chat_responses(user_inp, mysql_db, historical_message, table_details=None,llm=None,llm_name=None):
    # ask steps
    prompt_ask_steps_str = prompt_ask_steps.format(user_inp=user_inp)
    response_steps = chat_with_ai(init_system_msg(table_details), prompt_ask_steps_str, historical_message, None,
                                  token_limit=cfg.fast_token_limit, llm=llm,llm_name=llm_name)

    historical_message[-2]["content"] = prompt_ask_steps_no_egs.format(user_inp=user_inp)

    response_steps_list_of_dict = get_steps_from_response(response_steps)

    if len(response_steps_list_of_dict) == 0:
        return f"NOT NEED MEMORY: {response_steps}",'no results found.',None
        

    sql_results_history, new_mem_ops,response,sql_results,output_sql = chain_of_memory(response_steps_list_of_dict, mysql_db)
    return response,sql_results,output_sql

def need_update_sql(input_string):
    pattern = r"<\S+>"
    matches = re.findall(pattern, input_string)
    return matches