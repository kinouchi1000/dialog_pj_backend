import logging
from typing import Dict, List

from common.constants import DICT_PATH, MODEL_PATH, SENTENCEPIECE_MODEL_PATH

from inference_server.model.dialog import Generator


class Controller:
    def __init__(self) -> None:
        self.generator = Generator(
            data_path=DICT_PATH,
            checkpoint_path=MODEL_PATH,
            sentencepiece_model=SENTENCEPIECE_MODEL_PATH
        )

    def get_reply(self, context: List[Dict]) -> str:
        
        logging.info(f"context: {context}")
        if len(context)<1:
            return self.generator.favot.execute([{"utt":"||init||","spk":"spker"}])
        limit = 3
        while limit>0:
            ret = self.generator.favot.execute(context)
            if ret is None or len(ret) != 2:
                continue
            ret, ret_debug = ret
            if ret is not None:
                return ret
            limit-=1
        logging.info(f"ret: {ret}")

        return "ごめんなさい。応答に困りました。"
