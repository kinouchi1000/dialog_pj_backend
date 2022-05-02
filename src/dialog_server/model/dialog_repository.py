from grpc_service.dialog_server_pb2 import Reply

demo_reply_histories = [
    Reply(speaker_id="S1", comment="History1"),
    Reply(speaker_id="bot", comment="History2"),
    Reply(speaker_id="S1", comment="History3"),
    Reply(speaker_id="bot", comment="History4"),
    Reply(speaker_id="S1", comment="History5"),
    Reply(speaker_id="bot", comment="History6"),
    Reply(speaker_id="S1", comment="History7"),
    Reply(speaker_id="bot", comment="History8"),
    Reply(speaker_id="S1", comment="History9"),
    Reply(speaker_id="bot", comment="History10"),
    Reply(speaker_id="S1", comment="History11"),
    Reply(speaker_id="bot", comment="History12"),
    Reply(speaker_id="S1", comment="History13"),
    Reply(speaker_id="bot", comment="History14"),
]


class DialogRepository:
    def __init__(self) -> None:
        pass

    def store_dialog(self, speaker_id: str, comment: str, reply: str):
        # TODO store to Redis
        demo_reply_histories.append(Reply(speaker_id=speaker_id, comment=comment))
        demo_reply_histories.append(Reply(speaker_id="bot", comment=reply))

    def get_history(self, speaker_id: str, history_from: int, history_end: int):
        # TODO get from Redis
        if speaker_id == "S1" and history_end < len(demo_reply_histories):
            return demo_reply_histories[history_from:history_end]
        else:
            return []

    def get_history_all(self, speaker_id):
        # TODO get from Redis

        if speaker_id == "S1":
            return demo_reply_histories
        else:
            return []
