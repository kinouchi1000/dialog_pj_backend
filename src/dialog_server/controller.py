import logging
from typing import List

from common.constants import REPLY_HISTORY_LIMIT
from grpc_service.dialog_server_pb2 import Reply

from dialog_server.model.dialog_repository import DialogRepository
from dialog_server.model.inference_client import InferenceClient


class Controller:
    def __init__(self) -> None:
        self.inference_client = InferenceClient()
        self.repository = DialogRepository()

    def get_reply(self, speaker_id: str, comment: str) -> str:

        # get Infenrece (responce)
        history = self.repository.get_history(speaker_id=speaker_id, history_from=-4)
        reply = self.inference_client.get_reply(comment, history)

        # store data
        self.repository.store_dialog(speaker_id, comment=comment, reply=reply)
        return reply

    def get_reply_history_limited(
        self, speaker_id: str, history_from: int
    ) -> List[Reply]:

        history: List[Reply] = self.repository.get_history(speaker_id, history_from)

        return history

    def get_reply_history(self, speaker_id) -> List[Reply]:
        history: List[Reply] = self.repository.get_history_all(speaker_id)
        logging.info(f"limit history: {history}")
        return history
