#coding=utf8

from langchain.utilities import BingSearchAPIWrapper
from langchain.utilities import GoogleSerperAPIWrapper

class internet_search():
    def __init__(self):
        self.BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
        self.KEY = None
        self.search_device = 'bing'

    def set_v(self,search,key,result_len):
        self.search_device = search
        self.KEY = key
        self.result_len = result_len

    def bing_search(self,text):
        search = BingSearchAPIWrapper(bing_subscription_key=self.KEY,
                                    bing_search_url=self.BING_SEARCH_URL)
        return search.results(text, self.result_len)
    
    def google_search(self,text):
        search = GoogleSerperAPIWrapper(serper_api_key=self.KEY, gl='cn', hl='zh-cn', k=self.result_len)
        return search.run(text)
        
    def search_text(self,text):
        if self.search_device == 'bing':
            result=''
            rep=self.bing_search(text)
            for t in rep:
                result+=t["snippet"].replace('<b>','').replace('</b>','')+'\n'  if "snippet" in t.keys() else ""
            return result,rep
        elif self.search_device == 'google':
            result=''
            rep=self.google_search(text)
            for t in rep:
                result+=t["snippet"].replace('<b>','').replace('</b>','')+'\n'  if "snippet" in t.keys() else ""
            return result,rep
