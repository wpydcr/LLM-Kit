import io
import logging
import os

import numpy as np
import soundfile

from modules.agent.svc.inference import slicer
from modules.agent.svc.inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
# chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")

real_path = os.path.split(os.path.realpath(__file__))[0]
model_path = os.path.join(real_path, "..", "..", "..", "models", 'svc_models', 'Haruhi_54000.pth')

# model_path = "../../../models/svc_models/Haruhi_54000.pth"
config_path = os.path.join(real_path, "configs", "config.json")




def infer_to(spk, tran, voice,model):
    slice_db = -40
    
    wav_format = 'wav'
    # audio_file = io.BytesIO(voice)
    audio_file = voice
    chunks = slicer.cut(audio_file, db_thresh=slice_db)
    # audio_file = io.BytesIO(voice)
    audio_data, audio_sr = slicer.chunks2audio(audio_file, chunks)
    audio = []


    for (slice_tag, data) in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
        length = int(np.ceil(len(data) / audio_sr * model.target_sample))
        raw_path = io.BytesIO()
        soundfile.write(raw_path, data, audio_sr, format="wav")
        raw_path.seek(0)
        if slice_tag:
            print('jump empty segment')
            _audio = np.zeros(length)
        else:
            out_audio, out_sr = model.infer(spk, tran, raw_path)
            _audio = out_audio.cpu().numpy()
        audio.extend(list(_audio))
    # infer_tool.mkdir(["./vits_results"])
    # res_path = f'./vits_results/{tran}key_{spk}_{str(uuid.uuid4())}.{wav_format}'
    soundfile.write(voice, audio, model.target_sample, format=wav_format)

    return voice