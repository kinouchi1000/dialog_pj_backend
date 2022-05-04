import logging
from typing import Dict, List

from inference_server.model.dialog import Generator


class Controller:
    def __init__(self) -> None:
        self.generator = Generator(
            data_path="../docker/inference_server/data/sample/bin/",
            checkpoint_path="../docker/inference_server/model/japanese-dialog-transformer-1.6B-persona50k.pt",
            sentencepiece_model="../docker/inference_server/data/dicts/sp_oall_32k.model"
        )

    def get_reply(self, context: List[Dict]) -> str:
        
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

        return "ごめんなさい。応答に困りました。"
