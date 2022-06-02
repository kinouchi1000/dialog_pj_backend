# coding: utf-8
"""
対応version
fairseq v0.10.2
python-telegram-bot v13.1
全体を依存少なくリライト

"""
import logging
import math
import re
import sys
from datetime import datetime
from logging import DEBUG, basicConfig
from typing import Dict, List

import fairseq
import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils

sys.path.append("src")
from common.constants import BOT_NAME, MAX_WORKER
from inference_server.model.favot import Favot
from inference_server.model.favot_model import FavotModel

SEPARATOR = "[SEP]"
SPK1 = "[SPK1]"
SPK2 = "[SPK2]"
hiragana = re.compile("[\u3041-\u309F，、．。？！\?\!]+")


def add_local_args(parser):
    parser.add_argument(
        "--max-contexts", type=int, default=4, help="max length of used contexts"
    )
    parser.add_argument(
        "--suppress-duplicate",
        action="store_true",
        default=False,
        help="suppress duplicate sentences",
    )
    parser.add_argument(
        "--show-nbest", default=3, type=int, help="# visible candidates"
    )
    parser.add_argument(
        "--starting-phrase",
        default="こんにちは。よろしくお願いします。",
        type=str,
        help="starting phrase",
    )
    return parser


# serverから呼び出す用
class Generator:
    def __init__(
        self, data_path: str, checkpoint_path: str, sentencepiece_model: str
    ) -> None:

        # parser
        self.parser = options.get_interactive_generation_parser()
        add_local_args(self.parser)
        self.parser.set_defaults(
            path=checkpoint_path,
            num_worker=MAX_WORKER,
            beam=80,
            min_len=10,
            source_lang="src",
            target_lang="dst",
            tokenizer="space",
            no_repeat_ngram_size=3,
            nbest=80,
            sampling=True,
            sampling_topp=0.9,
            temperature=1.0,
            show_nbest=5,
        )

        self.args = options.parse_args_and_arch(
            self.parser,
            input_args=[
                data_path,
                "--bpe",
                "sentencepiece",
                "--sentencepiece-model",
                sentencepiece_model,
            ],
        )
        utils.import_user_module(self.args)

        # inference model
        self.fm = FavotModel(self.args)
        self.favot = Favot(self.args, self.fm, parser=self.parser)

        # logger
        rootname = "log/dialog.log"
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = rootname + "." + dt

        basicConfig(
            filename=fname,
            level=DEBUG,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        logging.info("setup is done")

    def inference_reply(
        self,
        context: List[Dict],
    ) -> str:
        """function run dialog system
        Args:
            context: list of dict [{"spk":"name","utt":"content"}]
        """

        if len(context) < 1:
            ret, ret_debug = self.favot.execute([{"utt": "||init||", "spk": "spker"}])
            return ret
        limit = 3
        while limit > 0:
            ret, ret_debug = self.favot.execute(context)

            if ret is not None:
                logging.info("sys_uttr: " + ret)
                print("\n".join(ret_debug))
                return ret
            limit -= 1

        return "ごめんなさい。応答に困りました。"

    def cancel(self):
        self.favot.reset()


#### test 用スクリプト###########
def _main():
    generator = Generator(
        data_path="docker/inference_server/data/sample/bin/",
        checkpoint_path="docker/inference_server/model/japanese-dialog-transformer-1.6B-persona50k.pt",
        sentencepiece_model="docker/inference_server/data/dicts/sp_oall_32k.model",
    )
    uttr_list = []
    ret = generator.inference_reply(uttr_list)
    uttr_list.append({"spk": SPK1, "utt": ret})

    while True:
        uttr = input(">>")
        uttr = uttr.rstrip("/n")
        uttr_list.append({"spk": SPK2, "utt": uttr})
        logging.info(uttr_list)

        ret = generator.inference_reply(uttr_list)
        print(f"sys:{ret}")
        uttr_list.append({"spk": SPK1, "utt": ret})


if __name__ == "__main__":
    _main()
