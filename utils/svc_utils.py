import os
from modules.agent.svc import vits_haruhi
from modules.agent.svc.inference.infer_tool import Svc


class SVC:
    """
        This class used for SVC setup and inference.
        这个类用于配置svc模型以及推理。
    """

    def __init__(self):
        self.svc_model = None
        self.model_path = ""

    def load(self, voice_style):
        """Loads the SVC model.

        Args:
            voice_style (str): The voice style.

        """
        if voice_style != "默认":
            real_path = os.path.split(os.path.realpath(__file__))[0]
            path = os.path.join(real_path, "..", "models", 'svc_models', "svc", voice_style)
            config_path = os.path.join(real_path, "..", "modules", "agent", "svc", "configs", "config.json")

            if self.svc_model is None or path != self.model_path:
                self.svc_model = Svc(path, config_path)
                self.model_path = path

    def voice_process(self, voice_style, filename):
        """Performs voice processing using the SVC model.

        Args:
            voice_style (str): The voice style.
            filename (str): The filename of the input audio.

        """
        if voice_style != "默认":
            vits_haruhi.vits_haruhi(voice_style, filename + ".wav", 8, self.svc_model)
