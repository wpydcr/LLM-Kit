import json
import os
import torch
from modules.agent.vits.models import SynthesizerTrn
from modules.agent.vits.vits import hps_ms, tts_fnD,save_as_wav
from modules.agent.vits import utils
class VITS:
    """
    This class is used for the vits setup and inference.
    这个类用于vits模型的配置以及推理。
    """
    def __init__(self):
        self.vits_model = None
        self.model_path = ""
        self.model_info = {}

    def load(self,model_name,outputLanguage):
        """
            Loads the specified VITS model.

            Args:
                model_name (str): The name of the VITS model.
                outputLanguage (str): The desired output language.

            Returns:
                None
        """

        real_path = os.path.split(os.path.realpath(__file__))[0]
        model_path = os.path.join(real_path,  "..", "models", "vits_pretrained_models",
                                       model_name,
                                       model_name + ".pth")
        if self.model_path != model_path:
            self.model_path = model_path
            new_path = os.path.join(real_path,  "..", "models", "vits_pretrained_models", "info.json")
            with open(new_path, "r", encoding="utf-8") as f:
                models_info = json.load(f)
            sid = models_info[model_name]['sid']
            name_en = models_info[model_name]['name_en']
            name_zh = models_info[model_name]['name_zh']
            title = models_info[model_name]['title']
            cover = f"pretrained_models/{model_name}/{models_info[model_name]['cover']}"
            example = models_info[model_name]['example']
            language = models_info[model_name]['language']
            self.vits_model = SynthesizerTrn(
                len(hps_ms.symbols),
                hps_ms.data.filter_length // 2 + 1,
                hps_ms.train.segment_size // hps_ms.data.hop_length,
                n_speakers=hps_ms.data.n_speakers if models_info[model_name]['type'] == "multi" else 0,
                **hps_ms.model)

            utils.load_checkpoint(self.model_path, self.vits_model, None)
            _ = self.vits_model.eval().to(torch.device("cuda"))
            self.model_info = {"sid": sid, "name_en": name_en, "name_zh": name_zh,
                             "title": title, "cover": cover, "example": example, "language": language,
                              "outputLanguage": outputLanguage, "noise_scale": 0.6,
                             "noise_scale_w": 0.668,
                             "length_scale": 1.2}


    def to_audio(self,response,output_path):
        """
                Converts the text response to audio using the loaded VITS model.

                Args:
                    response (str): The text response to be converted.
                    output_path (str): The path to save the generated audio file.

                Returns:
                    str: The path of the saved audio file.
        """


        output_audio = tts_fnD(self.vits_model, self.model_info["sid"], response, self.model_info["outputLanguage"],
                               self.model_info["noise_scale"],
                               self.model_info["noise_scale_w"], self.model_info["length_scale"], is_symbol=False)
        sample_rate = 22050  # Example sample rate, modify according to your needs
        save_as_wav(output_audio, output_path+".wav", sample_rate)
        return output_path+".wav"