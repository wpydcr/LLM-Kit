import json
import datetime
import os

real_path = os.path.split(os.path.realpath(__file__))[0]
new_path = os.path.join(real_path, "..", "data",'modeldata')

def download_org_data(a,download_path):
    if download_path=='':
        download_path=f"""data_{datetime.datetime.now().strftime("%m%d_%H%M%S")}"""
    if len(a)>0:   
        lsa=[]
        for i in a:
            lsa.append({"que":i, "ans": a[i]})
        with open(os.path.join(new_path, 'json',download_path+'.json'), "w", encoding="utf-8") as file:
            json.dump(lsa, file, ensure_ascii=False, indent=4)


def download_jsonl_data(a,download_path,type_=None):
    if download_path=='':
        download_path=f"""data_{datetime.datetime.now().strftime("%m%d_%H%M%S")}"""

    if type_==None:
        path = os.path.join(new_path, "LLM")
    else:
        path = os.path.join(new_path, "Embedding")
    if not os.path.exists(path):
        os.mkdir(path)
    data_path = os.path.join(path, download_path)
    if len(a)==2:
        if len(a[0])>0:
            if not os.path.exists(data_path):
                os.mkdir(data_path)
            with open(os.path.join(path, download_path,'valid.jsonl'),mode='w',encoding='utf-8') as f:
                for i in a[0]:
                    lsa = {"instruction":i, "output": a[0][i], "input": ""}
                    f.write(json.dumps(lsa,ensure_ascii=False) + '\n')
        if len(a[1])>0:
            if not os.path.exists(data_path):
                os.mkdir(data_path)
            with open(os.path.join(path, download_path,'train.jsonl'),mode='w',encoding='utf-8') as f:
                for i in a[1]:
                    lsa = {"instruction":i, "output": a[1][i], "input": ""}
                    f.write(json.dumps(lsa,ensure_ascii=False) + '\n')
    elif len(a)==1:
        if len(a[0])>0:
            a=a[0]
            if not os.path.exists(data_path):
                    os.mkdir(data_path)
            with open(os.path.join(path, download_path,'train.jsonl'),mode='w',encoding='utf-8') as f:
                for i in a:
                    lsa = {"instruction":i, "output": a[i], "input": ""}
                    f.write(json.dumps(lsa,ensure_ascii=False) + '\n')

            