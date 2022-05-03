import logging
from concurrent import futures

import grpc
from common.constants import INFERENCE_SERVER_IP, INFERENCE_SERVER_PORT, MAX_WORKER, REPLY_HISTORY_LIMIT
from grpc_service import inference_server_pb2, inference_server_pb2_grpc
from grpc_service.inference_server_pb2 import Comment, Reply

from inference_server.controller import Controller


class InferenceServicer(inference_server_pb2_grpc.InferenceServiceServicer):
    def __init__(self) -> None:
        self.controller = Controller()

    def GetReply(self, request: Comment, context):
        comment_list = [request.comment_past1, request.comment_past2, request.comment_past3, request.comment]
        reply = self.controller.get_reply(comment_list)

        return Reply(reply=reply)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKER))
    inference_server_pb2_grpc.add_DialogServiceServicer_to_server(InferenceServicer(), server)

    server.add_insecure_port(f"{INFERENCE_SERVER_IP}:{INFERENCE_SERVER_PORT}")
    server.start()

    logging.info(f"server running with {INFERENCE_SERVER_IP}:{INFERENCE_SERVER_PORT}")
    logging.info(f"max worker is {MAX_WORKER}")

    server.wait_for_termination()
