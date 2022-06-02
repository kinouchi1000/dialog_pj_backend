import logging
from typing import List

import grpc
from common.constants import INFERENCE_SERVER_IP, INFERENCE_SERVER_PORT
from grpc_service.dialog_server_pb2 import Reply
from grpc_service.inference_server_pb2 import Comment
from grpc_service.inference_server_pb2_grpc import InferenceServiceStub


class InferenceClient:
    def __init__(self) -> None:
        self._channel = grpc.insecure_channel(
            f"{INFERENCE_SERVER_IP}:{INFERENCE_SERVER_PORT}",
            options=(("grpc.enable_http_proxy", 0),),
        )
        self._stub = InferenceServiceStub(self._channel)

    def get_reply(self, comment: str, history: List[Reply]) -> str:
        """get reply from bot
        Note that history is ascending order. Right is newest
        """
        reply = Reply(speaker_id="speaker", comment=comment)
        history.append(reply)
        comments = Comment(comments=history)

        ret = self._stub.GetReply(comments)

        return ret.comment
