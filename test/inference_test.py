from ..src.inference_server.controller import Controller

controller = Controller()

l = [
    {'spk':"bot","utt":"はじめまして"},
    {'spk':"spk","utt":"こんにちは"},
    {'spk':"bot","utt":"名前はなんと言うんですか？"},
    {'spk':"spk","utt":"木内です。"},
]

ret = controller.get_reply(l)

print(ret)
