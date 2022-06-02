import logging
from concurrent import futures

import grpc
from common.constants import (
    INFERENCE_SERVER_IP,
    INFERENCE_SERVER_PORT,
    MAX_WORKER,
    REPLY_HISTORY_LIMIT,
)
from google.protobuf import json_format
from grpc_service import inference_server_pb2, inference_server_pb2_grpc
from grpc_service.dialog_server_pb2 import Reply
from grpc_service.inference_server_pb2 import Comment

from inference_server.controller import Controller


class InferenceServicer(inference_server_pb2_grpc.InferenceServiceServicer):
    def __init__(self) -> None:
        self.controller = Controller()

    def GetReply(self, request, con):

        comment_list = []
        histories_dict = json_format.MessageToDict(request)

        for comment in histories_dict["comments"]:
            comment_list.append(self.get_dict(comment["speakerId"], comment["comment"]))

        comment_list = list(filter(None, comment_list))

        reply = self.controller.get_reply(comment_list)
        return Reply(comment=reply)

    @staticmethod
    def get_dict(id: str, comment: str):
        if id != None and comment != None and id != "" and comment != "":
            return {"spk": id, "utt": comment}
        else:
            return None


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKER))
    inference_server_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(), server
    )

    server.add_insecure_port(f"{INFERENCE_SERVER_IP}:{INFERENCE_SERVER_PORT}")
    server.start()

    logging.info(f"server running with {INFERENCE_SERVER_IP}:{INFERENCE_SERVER_PORT}")
    logging.info(f"max worker is {MAX_WORKER}")

    server.wait_for_termination()
