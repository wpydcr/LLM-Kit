import requests
import time
import zhtts
import os

real_path = os.path.split(os.path.realpath(__file__))[0]

lang_dict = {
    '中文':'zh',
    '英语':'en',
    '日语':'jp'
}

def tts(text, spd, lang, filename):
    # url = f"https://fanyi.baidu.com/gettts?lan={lang_dict[lang]}&text={text}&spd={spd}&source=web"

    # payload = {}
    # headers = {
    #     'Cookie': 'BAIDUID=543CBD0E4FB46C2FD5F44F7D81911F15:FG=1'
    # }

    # res = requests.request("GET", url, headers=headers, data=payload)
    # cs=0
    # while res.content == b'' and cs<11:
    #     cs+=1
    #     res = requests.request("GET", url, headers=headers, data=payload)
    #     time.sleep(0.1)
    # if res.status_code == 200:
    #     return res.content
    # else:
    #     return None

    url = f"https://yntts.qq.com/generateTTSURL"
    
    payload = {'txt': text, 'volume': 50, 'speed': 50, 'speaker': "25", 'pitch': 50, 'type': 1, 'tone': 42}

    res = requests.get(url, json=payload)
    cs=0
    while res.status_code != 200 and cs<11:
        cs+=1
        res = requests.get(url, json=payload)
        time.sleep(0.1)

    if res.status_code != 200:
        return None

    try:
        os.remove(filename+'.mp3')
    except FileNotFoundError as e:
        pass
    except Exception as e:
        return None

    session_id = res.json().get('session_id')
    url2 = f"https://yntts.qq.com/tts.mp3?session_id={session_id}&speaker=42&language=91&sampling_rate=816"
    res2 = requests.get(url2, stream=True)
    with open(filename+'.mp3', 'wb') as fd:
        for chunk in res2.iter_content():
            fd.write(chunk)
    
    # return filename+'.mp3'


def get_voice(text, spd, filename, gen_type, lang):
    if gen_type == '在线':
        tts(text, spd, lang, filename)
        # if voice is None:
        #     print("TTS failed")
        #     return None
        # with open(filename+ ".mp3", "wb") as f:
        #     f.write(voice)
        # audio = AudioSegment.from_mp3(filename+'.mp3')
        # audio.export(filename+'.wav', format="wav")
        # os.remove(filename+'.mp3')
    elif gen_type == '本地':
        # 暂时支持中文
        tts_model = zhtts.TTS()
        tts_model.text2wav(text, filename+'.wav')

    return filename
