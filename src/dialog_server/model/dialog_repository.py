from grpc_service.dialog_server_pb2 import Reply

demo_reply_histories = [
    Reply(speaker_id="S1",  comment="こんにちは。"),
    Reply(speaker_id="bot", comment="こんにちは。元気？"),
    Reply(speaker_id="S1",  comment="元気だよ！"),
    Reply(speaker_id="bot", comment="お名前はなんていうの？"),
    Reply(speaker_id="S1",  comment="木内です。よろしくおねがいします"),
    Reply(speaker_id="bot", comment="よろしくね！ご職業はなんですか？"),
    Reply(speaker_id="S1",  comment="職業はプログラマーです"),
    Reply(speaker_id="bot", comment="へー！すごいですね。"),
    Reply(speaker_id="S1",  comment="それほどでも無いですよ！"),
    Reply(speaker_id="bot", comment="でもなんで、プログラマーになろうとしたんですか？"),
    Reply(speaker_id="S1",  comment="中学校のときから進路を決めてました。"),
    Reply(speaker_id="bot", comment="じゃあいつから頑張ってたんですか？"),
]


class DialogRepository:
    def __init__(self) -> None:
        pass

    def store_dialog(self, speaker_id: str, comment: str, reply: str):
        # TODO store to Redis
        demo_reply_histories.append(Reply(speaker_id=speaker_id, comment=comment))
        demo_reply_histories.append(Reply(speaker_id="bot", comment=reply))

    def get_history(self, speaker_id: str, history_from: int = -4, history_end: int = None):

        if history_end is None:
            history_end = len(demo_reply_histories)
        # TODO get from Redis
        if speaker_id == "S1" and history_end <= len(demo_reply_histories):
            return demo_reply_histories[history_from:history_end]
        else:
            return []

    def get_history_all(self, speaker_id):
        # TODO get from Redis

        if speaker_id == "S1":
            return demo_reply_histories
        else:
            return []
