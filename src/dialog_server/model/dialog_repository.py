import logging

from common.constants import REPLY_HISTORY_LIMIT
from grpc_service.dialog_server_pb2 import Reply


class DialogRepository:
    def __init__(self) -> None:
        self.demo_reply_histories = []

    def store_dialog(self, speaker_id: str, comment: str, reply: str):
        
        # TODO store to Redis
        self.demo_reply_histories.append(Reply(speaker_id=speaker_id, comment=comment))
        self.demo_reply_histories.append(Reply(speaker_id="bot", comment=reply))

    def get_history(self, speaker_id: str, history_from: int = -4):

        history_end = history_from + REPLY_HISTORY_LIMIT
        if history_end >len(self.demo_reply_histories):
            history_end = len(self.demo_reply_histories)

        # TODO get from Redis
        if speaker_id == "S1":
            return self.demo_reply_histories[history_from:history_end]
        else:
            logging.info("speaker doesn't exist")
            return []

    def get_history_all(self, speaker_id):
        
        # TODO get from Redis
        if speaker_id == "S1":
            return self.demo_reply_histories
        else:
            logging.info("speaker doesn't exist")
            return []
