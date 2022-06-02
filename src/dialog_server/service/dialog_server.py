import logging
from concurrent import futures

import grpc
from common.constants import (
    DIALOG_SERVER_IP,
    DIALOG_SERVER_PORT,
    MAX_WORKER,
    REPLY_HISTORY_LIMIT,
)
from grpc_service import dialog_server_pb2, dialog_server_pb2_grpc
from grpc_service.dialog_server_pb2 import (
    GetReplyHistoryLimitedParam,
    Reply,
    ReplyHistoryLimited,
)

from dialog_server.controller import Controller


class DialogServicer(dialog_server_pb2_grpc.DialogServiceServicer):
    def __init__(self) -> None:
        self.controller = Controller()

    def SendReply(self, request: Reply, context):
        """This function reply responce from dialog bot"""
        logging.debug("called send reply")
        logging.debug(f"{request}")

        speaker_id: str = request.speaker_id
        comment: str = request.comment

        reply = self.controller.get_reply(speaker_id, comment)
        logging.debug(f"reply:{reply}")

        return Reply(speaker_id="bot", comment=reply)

    def GetReplyHistoryLimited(self, request: GetReplyHistoryLimitedParam, context):
        """This fuction responce the reply histories"""
        logging.debug("called get reply history limited")

        speaker_id = request.speaker_id
        history_from = request.history_from

        history = self.controller.get_reply_history_limited(speaker_id, history_from)

        return ReplyHistoryLimited(replies=history)

    def GetReplyHistory(self, request, context):
        """This function responce the itterable reply history"""
        logging.debug("called get reply history")
        speaker_id = request.speaker_id

        histories = self.controller.get_reply_history(speaker_id)

        for history in histories:
            yield history


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKER))
    dialog_server_pb2_grpc.add_DialogServiceServicer_to_server(DialogServicer(), server)

    server.add_insecure_port(f"{DIALOG_SERVER_IP}:{DIALOG_SERVER_PORT}")
    server.start()

    logging.info(f"server running with {DIALOG_SERVER_IP}:{DIALOG_SERVER_PORT}")
    logging.info(f"max worker is {MAX_WORKER}")

    server.wait_for_termination()
