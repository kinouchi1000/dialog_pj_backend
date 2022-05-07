#coding: utf-8
"""
対応version
fairseq v0.10.2
python-telegram-bot v13.1
全体を依存少なくリライト

"""
import ast
import collections
import copy
import difflib
import logging
import math
import re
import sys
import time
from datetime import datetime
from logging import DEBUG, basicConfig
from typing import Dict, List

import fairseq
import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders
# from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import (pack_constraints,
                                                  unpack_constraints)
from fairseq_cli.generate import get_symbols_to_strip_from_output
#from fairseq_cli import interactive as intr
from fairseq_cli.interactive import make_batches

sys.path.append("src")
from common.constants import BOT_NAME, MAX_WORKER
from inference_server.model.favot_model import FavotModel

SEPARATOR = "[SEP]"
SPK1 = "[SPK1]"
SPK2 = "[SPK2]"
hiragana = re.compile('[\u3041-\u309F，、．。？！\?\!]+')



class Favot(object):

    def encode_fn(self, x):
        if self.fm.tokenizer is not None:
            x = self.fm.tokenizer.encode(x)
        if self.fm.bpe is not None:
            x = self.fm.bpe.encode(x)
        return x

    def decode_fn(self, x):
        if self.fm.bpe is not None:
            x = self.fm.bpe.decode(x)
        if self.fm.tokenizer is not None:
            x = self.fm.tokenizer.decode(x)
        return x

    def __init__(self, args, favot_model, *, parser=None):
        self.parser = parser
        self.fm = favot_model
        self.args = args
        self.cfg = convert_namespace_to_omegaconf(args)
        # TODO この２つをDBから取得
        self.contexts = [] # 対話文　
        self.sent_contexts = [] 
        self.debug = False
        self.delimiter = "．。 　?？!！♪☆★"
        self.sent_splitter = re.compile(".*?[{}]".format(self.delimiter), re.DOTALL)
        self.alphs = "abcdefghijklmnopqrstuvwyz"

        self.make_input_func = self.make_input

        utils.import_user_module(args)

        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.batch_size is None:
            args.batch_size = 1

    def sent_split(self, line):
        """文章と文に分割"""
        _rets = self.sent_splitter.findall(line)
        rets = [r for r in _rets if r != ""]
        if "".join(rets) != line:
            c = re.sub(re.escape("".join(rets)), "", line)

            if c != "":
                rets.append(c)
        rets = [r.strip(" \n\t") for r in rets]

        return rets

    def common_word(self, word)->bool:
        word = word.strip("．。？?！!・")
        common = [
            "です",
            "ます",
            "ありがとう",
            "趣味",
            "(笑)",
        ]

        ## 本当はコーパス内の出現頻度で足きり
        if len(word) <= 1:
            return True
        if len(word) <= 2:
            hira = hiragana.findall(word)
            if len(hira) == 0:
                pass
            elif len("".join(hira)) >= 1:
                return True
            if word in ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]:
                return True
        if len(word) <= 3:
            if word[-1] == "い" or word[-1] == "る":
                return True
        for c in common:
            if c in word:
                return True
        if hiragana.fullmatch(word) is not None:
            return True

        return False

    def set_generator_parameters(self, args):
        for k, v in args.items():
            cur_v = self.args.__dict__[k]
            if v == "None":
                self.args.__setattr__(k, None)
            elif type(cur_v) == int:
                self.args.__setattr__(k, int(v))
            elif type(cur_v) == float:
                self.args.__setattr__(k, float(v))
            elif type(cur_v) == bool:
                if v == "False" or v == "false":
                    self.args.__setattr__(k, False)
                else:
                    self.args.__setattr__(k, True)
            elif type(cur_v) == str:
                self.args.__setattr__(k, str(v))
            else:
                raise TypeError("Unknown type of generator parameter")
            print(self.args)
        self.fm.generator = self.fm.task.build_generator(self.fm.models, self.args)
        _args = copy.deepcopy(self.args)
        _args.__setattr__("score_reference", True)
        _args.__setattr__("beam", 1)
        _args.__setattr__("nbest", 1)

        self.fm.scorer = self.fm.task.build_generator(self.fm.models, _args)
        logging.info("update generator parameter:" + str(args))
        return

    def make_single_sample(self, inputs, args, task, max_positions):
        ret = []

        for batch in make_batches(inputs, args, task, max_positions, self.encode_fn):
            bsz = batch.src_tokens.size(0)
            tokens = batch.src_tokens
            lengths = batch.src_lengths
            constraints = batch.constraints
            if self.fm.use_cuda:
                tokens = tokens.cuda()
                lengths = lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                'net_input': {
                    'src_tokens': tokens,
                    'src_lengths': lengths,
                    'prev_output_tokens': tokens,
                },
            }
            ret.append(sample)
        return ret
    
    # 実行
    def execute(self, context:List[Dict], mode="normal"):
        logging.info(f"execute input : {context}")
        ret = self._execute(context, mode=mode)
        if ret is not None:
            ret_scores, ret_debug = ret
        else:
            return
        if len(ret_scores) == 0:
            return "", ret_debug
        ret_utt, ret_score = ret_scores.most_common(1)[0]
        print(ret_score, ret_utt)
        if mode == "prefinish":
            ret_utt = ret_utt + "\nあ、すみません。そろそろ時間ですね。今日はありがとうございました。"
        context.append({'spk':SPK1,'utt':ret_utt})
        logging.info(str(ret_scores.most_common(5)))
        self.reset()
        return ret_utt, ret_debug

    # 実行
    def _execute(self, contexts:List[Dict], **kwargs):
        """対話を実行"""
        current_uttr = contexts[-1]["utt"]

        # 設定コマンド各種
        mode = "normal"
        if "mode" in kwargs:
            mode = kwargs["mode"]
        if current_uttr.startswith("/help"):
            logging.info(str(self.args))
            return collections.Counter(), [str(self.args)]
        if current_uttr.startswith("/debug"):
            if current_uttr == "/debug off" or current_uttr == "/debug False" or current_uttr == "/debug false":
                self.debug = False
            else:
                self.debug = True
            return

        if current_uttr.startswith("/sys "):
            toks = current_uttr.split(" ")
            key = toks[1]
            val = toks[2]
            args = {key: val}
            self.set_generator_parameters(args)
            return


        # 初期対話
        if current_uttr == "||init||":
            start_utt = self.args.starting_phrase
            ret_scores = collections.Counter()
            ret_scores[start_utt] = 0.0
            return ret_scores, ""

        ret_debug = []
        start_id = 0

        inputs = [
            self.make_input_func(contexts),
        ]
        self.add_contexts(contexts)

        # if current_uttr.startswith("/input "):
        #     if "終了処理" in current_uttr:
        #         mode = "finish"
        #     _input = current_uttr[7:]
        #     inputs = [
        #         _input,
        #     ]

        logging.info("input_seq: " + str(inputs))
        if self.debug:
            ret_debug.append("input_seq: " + str(inputs))
        results = []

        
        args = self.fm.cfg
        task = self.fm.task
        max_positions = self.fm.max_positions
        use_cuda = self.fm.use_cuda
        # inference
        for i, batch in enumerate(make_batches(inputs, args, task, max_positions, self.encode_fn)):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            constraints = batch.constraints
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                    'prev_output_tokens': src_tokens
                },
            }
            translate_start_time = time.time()
            translations = task.inference_step(self.fm.generator, self.fm.models, sample, constraints=constraints)
            translate_time = time.time() - translate_start_time
            list_constraints = [[] for _ in range(bsz)]

            if args.generation.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.fm.tgt_dict.pad())
                constraints = list_constraints[i]
                results.append((start_id + id, src_tokens_i, hypos, {
                    "constraints": constraints,
                    "time": translate_time / len(translations)
                }))

        ret_scores = collections.Counter()


        # sort output to match input order
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
            if self.fm.src_dict is not None:
                src_str = self.fm.src_dict.string(src_tokens, args.common_eval.post_process)
                print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))
                for constraint in info["constraints"]:
                    print("C-{}\t{}".format(id_, self.fm.tgt_dict.string(constraint, args.common_eval.post_process)))
                    if self.debug:
                        ret_debug.append("C-{}\t{}".format(
                            id_, self.fm.tgt_dict.string(constraint, args.common_eval.post_process)))
            # Process top predictions

            _cand_counter = collections.Counter()
            for i, hypo in enumerate(hypos[:min(len(hypos), min(args.generation.nbest, self.args.show_nbest))]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=self.fm.align_dict,
                    tgt_dict=self.fm.tgt_dict,
                    remove_bpe=args.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.fm.generator),
                )
                detok_hypo_str = self.decode_fn(hypo_str)

                score = hypo['score'] / math.log(2)  # convert to base 2

                # remove duplicate candidates
                dup_flag, nodup_cand = self.contain_duplicate(detok_hypo_str, self.contexts)
                if dup_flag and self.args.suppress_duplicate:
                    logging.info("duplicated pattern: {}".format(detok_hypo_str))
                    if nodup_cand != "":
                        logging.info("no dup cand: {}".format(nodup_cand))
                        score -= 100
                    else:
                        score = score - 100000
                # original hypothesis (after tokenization and BPE)
                logging.info("system_utt_cands: " + 'H-{}\t{}\t{}'.format(id_, score, hypo_str))
                logging.info("system_utt_cands: " + 'D-{}\t{}\t{}'.format(id_, score, detok_hypo_str))
                if self.debug:
                    ret_debug.append("system_utt_cands: " + 'D-{}\t{}\t{}'.format(id_, score, detok_hypo_str))

                _scores = hypo['positional_scores'].div_(math.log(2)).tolist()
                _contexts = self.contexts

                if "<ex>" in detok_hypo_str:
                    detok_hypo_str = detok_hypo_str.replace("<ex>", "").replace("</ex>", "")
                    if "<" in detok_hypo_str or ">" in detok_hypo_str:
                        score -= 1000
                detok_hypo_str = detok_hypo_str.replace("(笑)", " ").replace("(笑）", " ").replace("（笑)", " ")
                if "unk" in detok_hypo_str and len(_contexts) > 0:
                    c1 = re.findall("(..<unk>)", detok_hypo_str)
                    c2 = re.findall("(.<unk>.)", detok_hypo_str)
                    c3 = re.findall("(..<unk>)", detok_hypo_str)
                    logging.info("{}/{}/{}".format(str(c1), str(c2), str(c3)))
                    try:
                        if len(c1) > 0:
                            c1 = c1[0]
                            cc = re.findall("{}(.)".format(c1[0:2]), _contexts[-1]["utt"], re.DOTALL)
                            if len(cc) > 0:
                                detok_hypo_str = detok_hypo_str.replace(c1[0:2] + "<unk>", c1[0:2] + cc[0])
                        elif len(c2) > 0:
                            c2 = c2[0]
                            cc = re.findall("{}(.){}".format(c2[0], c2[1]), _contexts[-1]["utt"], re.DOTALL)
                            if len(cc) > 0:
                                detok_hypo_str = detok_hypo_str.replace(c2[0] + "<unk>" + c2[1], c2[0] + cc[0] + c2[1])
                        elif len(c3) > 0:
                            c3 = c3[0]
                            cc = re.findall("(.){}".format(c3[0:2]), _contexts[-1]["utt"], re.DOTALL)
                            if len(cc) > 0:
                                detok_hypo_str = detok_hypo_str.replace("<unk>" + c3[0:2], cc[0] + c3[0:2])
                        else:
                            score -= 1000
                    except:
                        score -= 1000
                if "呼べば" in detok_hypo_str or "呼ん" in detok_hypo_str or "呼び" in detok_hypo_str:
                    score -= 2
                if mode != "prefinish" and mode != "finish":
                    if "時間で" in detok_hypo_str:
                        score -= 2
                        if "そろそろ" in detok_hypo_str:
                            score -= 1000000
                nodup_cand = nodup_cand.replace("(笑)", " ").replace("(笑）", " ").replace("（笑)", " ")

                ret_scores[detok_hypo_str] = score

                logging.info("system_utt_cands: " + 'P-{}\t{}'.format(
                    id_,
                    ' '.join(
                        map(
                            lambda x: '{:.4f}'.format(x),
                            # convert from base e to base 2
                            hypo['positional_scores'].div_(math.log(2)).tolist(),
                        ))))

                if args.generation.print_alignment:
                    alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                    print('A-{}\t{}'.format(id_, alignment_str))

        return ret_scores, ret_debug

    # 重複を含む
    def contain_duplicate(self, hypo,context):
        sents = self.sent_split(hypo)
        nodup_cand = []
        ff = False
        
        for orgs in sents:
            f = False
            s = orgs.rstrip("!?！？。．　・")
            spk2_skip = 0
            for i, cdic in enumerate(self.sent_contexts[::-1]):
                if cdic["spk"] == SPK2 and spk2_skip < 2:
                    continue
                elif cdic["spk"] == SPK1:
                    spk2_skip += 1

                c = cdic["utt"].rstrip("!?！？。．　・")
                hiras = hiragana.findall(s)
                hira = "".join(hiras)
                if len(hira) >= len(c) - 1 and (len(c) < 7 or len(s) < 7):
                    continue
                if "そう" in c and len(c) < 10:
                    continue
                e = difflib.SequenceMatcher(None, s, c).ratio()
                if e > 0.5:
                    logging.info("sim: {}, cand: {}, contexts: {}".format(e, s, c))
                if e > 0.65:
                    f = True
                    ff = True
                    break
            if not f:
                nodup_cand.append(orgs)

        ## 文全体チェック: nodup_candでかけるように変更
        f = False
        for cdic in context:
            if cdic["spk"] == SPK2:
                continue
            c = cdic["utt"]
            e = difflib.SequenceMatcher(None, "".join(nodup_cand), c).ratio()
            if e > 0.5:
                logging.info("all sim: {}, cand: {}, contexts: {}".format(e, hypo, c))
            if e > 0.5:
                f = True
                ff = True
                break

        ## check duplicate tokens within the sentence itself
        _contexts = []

        for i, s in enumerate(nodup_cand):
            _contexts.append({"spk": SPK1, "utt": s, "id": i})

        new_nodup_cand = []
        for i, s in enumerate(nodup_cand):
            f = False
            for j, cdic in enumerate(_contexts):
                c = cdic["utt"]
                ## skip too short sentences
                s = s.strip(" ")
                c = c.strip(" ")

                if len(c) < 2:
                    continue
                e = difflib.SequenceMatcher(None, s, c).ratio()
                if i == j:
                    continue
                if e > 0.5:
                    logging.info("self: sim: {}, cand: {}, contexts: {}".format(e, s, c))
                if e > 0.65:
                    f = True
                    ff = True
                    break
            if not f:
                new_nodup_cand.append(s)
        ret_flag = ff
        return ret_flag, "".join(new_nodup_cand)

    def add_contexts(self, contexts:List[Dict]):
        self._add_contexts(contexts)
        return

    def _add_contexts(self, contexts:List[Dict]):
        for c in contexts:
            if c["spk"] == BOT_NAME:
                c["spk"] = SPK1
            else: 
                c["spk"] = SPK2
            
            for s in self.sent_split(c["utt"]):
                self.sent_contexts.append({"spk": c["spk"], "utt": s})
        self.contexts = contexts


    def make_input(self,contexts:List[Dict]):

        _lines = []

        # 後ろのmax_contexts+1(現在の発話分)だけを取得
        for c in contexts[-self.args.max_contexts+1:]:
            spk = c["spk"]
            utt = "".join(c["utt"])

            if spk == BOT_NAME:
                spk = SPK1
            else:
                spk = SPK2
            logging.info(f"spk:{spk} utt:{utt}")
            _line = spk + utt + SEPARATOR
            _lines.append(_line)

        line = ""
        # 文字列が512以上なら切り捨て
        for _line in _lines[::-1]:
            if len(line) + len(_line) > 512:
                break
            line = _line + line

        line = line[:-len(SEPARATOR)]

        logging.info(line)

        return line

    def reset(self):
        self.contexts = []
        self.sent_contexts = []
        return

