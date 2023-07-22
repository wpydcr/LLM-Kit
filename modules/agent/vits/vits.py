# coding=utf-8
import argparse
import json
import os

import numpy as np
import torch
from scipy.io import wavfile
from torch import no_grad, LongTensor

from modules.agent.vits.models import SynthesizerTrn
from modules.agent.vits.text import text_to_sequence, _clean_text
from modules.agent.vits import commons, utils


def save_as_wav(data, filename, sample_rate):
    # Scale the float values to the range of 16-bit signed integers (-32768 to 32767)
    scaled_data = np.int16(data * 32767)

    # Save the data as a WAV file
    wavfile.write(filename, sample_rate, scaled_data)


real_path = os.path.split(os.path.realpath(__file__))[0]
new_path = os.path.join(real_path, "config", "config.json")

hps_ms = utils.get_hparams_from_file(new_path)


#
# audio_postprocess_ori = gr.Audio.postprocess
#
# def audio_postprocess(self, y):
#     data = audio_postprocess_ori(self, y)
#     if data is None:
#         return None
#     return gr_processing_utils.encode_url_or_file_to_base64(data["name"])


# gr.Audio.postprocess = audio_postprocess

def get_text(text, hps, is_symbol):
    text_norm, clean_text = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text


def create_tts_fn(net_g_ms, speaker_id):
    def tts_fn(text, language, noise_scale, noise_scale_w, length_scale, is_symbol):
        text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")

        if not is_symbol:
            if language == 0:
                text = f"[ZH]{text}[ZH]"
            elif language == 1:
                text = f"[JA]{text}[JA]"
            else:
                text = f"{text}"
        stn_tst, clean_text = get_text(text, hps_ms, is_symbol)
        with no_grad():
            device = "cuda"
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                   length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

        return "Success", (22050, audio)

    return tts_fn


def tts_fnD(net_g_ms, speaker_id, text, language, noise_scale, noise_scale_w, length_scale, is_symbol):
    text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
    if not is_symbol:
        if language == 0:
            text = f"[ZH]{text}[ZH]"
        elif language == 1:
            text = f"[JA]{text}[JA]"
        else:
            text = f"{text}"
    stn_tst, clean_text = get_text(text, hps_ms, is_symbol)
    with no_grad():
        device = "cuda"
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        sid = LongTensor([speaker_id]).to(device)
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                               length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

        return audio


def create_to_symbol_fn(hps):
    def to_symbol_fn(is_symbol_input, input_text, temp_lang):
        if temp_lang == 0:
            clean_text = f'[ZH]{input_text}[ZH]'
        elif temp_lang == 1:
            clean_text = f'[JA]{input_text}[JA]'
        else:
            clean_text = input_text
        return _clean_text(clean_text, hps.data.text_cleaners) if is_symbol_input else ''

    return to_symbol_fn


def change_lang(language):
    if language == 0:
        return 0.6, 0.668, 1.2
    elif language == 1:
        return 0.6, 0.668, 1
    else:
        return 0.6, 0.668, 1


def model_switch(models, text, path):
    model = "Genshin Impact-刻晴"
    output_audio = tts_fnD(models[model]["net_g_ms"], models[model]["sid"], text, models[model]["outputLanguage"],
                           models[model]["noise_scale"],
                           models[model]["noise_scale_w"], models[model]["length_scale"], is_symbol=False)

    sample_rate = 22050  # Example sample rate, modify according to your needs
    output_filename = path
    save_as_wav(output_audio, output_filename, sample_rate)


def vits_factory(text, outputPath, outputLanguage):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--api', action="store_true", default=False)
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    parser.add_argument("--all", action="store_true", default=False, help="enable all models")
    args = parser.parse_args()
    device = torch.device(args.device)
    categories = ["Genshin Impact"]
    models = {}

    real_path = os.path.split(os.path.realpath(__file__))[0]
    new_path = os.path.join(real_path, "../../apply", "..", "..", "models", "vits_pretrained_models", "info.json")
    with open(new_path, "r", encoding="utf-8") as f:
        models_info = json.load(f)
    for i, info in models_info.items():
        if info['title'].split("-")[0] not in categories or not info['enable']:
            continue
        sid = info['sid']
        name_en = info['name_en']
        name_zh = info['name_zh']
        title = info['title']
        cover = f"pretrained_models/{i}/{info['cover']}"
        example = info['example']
        language = info['language']
        net_g_ms = SynthesizerTrn(
            len(hps_ms.symbols),
            hps_ms.data.filter_length // 2 + 1,
            hps_ms.train.segment_size // hps_ms.data.hop_length,
            n_speakers=hps_ms.data.n_speakers if info['type'] == "multi" else 0,
            **hps_ms.model)

        real_path = os.path.split(os.path.realpath(__file__))[0]
        new_path = os.path.join(real_path, "../../apply", "..", "..", "models", "vits_pretrained_models", i, i + ".pth")
        print(new_path)

        utils.load_checkpoint(new_path, net_g_ms, None)
        _ = net_g_ms.eval().to(device)
        models[title] = {"sid": sid, "name_en": name_en, "name_zh": name_zh,
                         "title": title, "cover": cover, "example": example, "language": language,
                         "net_g_ms": net_g_ms, "outputLanguage": outputLanguage, "noise_scale": 0.6,
                         "noise_scale_w": 0.668,
                         "length_scale": 1.2}
    model_switch(models, text, outputPath)

    # a = tts_fnD(net_g_ms, speaker_id = 115, text=text,language = language, noise_scale = 0.6, noise_scale_w = 0.668, length_scale = 1.2, is_symbol = False)
