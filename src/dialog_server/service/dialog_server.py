import logging
from concurrent import futures

import grpc
from common.constants import DIALOG_SERVER_IP, DIALOG_SERVER_PORT, MAX_WORKER, REPLY_HISTORY_LIMIT
from grpc_service import dialog_server_pb2, dialog_server_pb2_grpc
from grpc_service.dialog_server_pb2 import GetReplyHistoryLimitedParam, Reply, ReplyHistoryLimited

demo_reply_histories = [
    Reply(speaker_id="S1", comment="History1"),
    Reply(speaker_id="S2", comment="History2"),
    Reply(speaker_id="S1", comment="History3"),
    Reply(speaker_id="S2", comment="History4"),
    Reply(speaker_id="S1", comment="History5"),
    Reply(speaker_id="S2", comment="History6"),
    Reply(speaker_id="S1", comment="History7"),
    Reply(speaker_id="S2", comment="History8"),
    Reply(speaker_id="S1", comment="History9"),
    Reply(speaker_id="S2", comment="History10"),
    Reply(speaker_id="S1", comment="History11"),
    Reply(speaker_id="S2", comment="History12"),
    Reply(speaker_id="S1", comment="History13"),
    Reply(speaker_id="S2", comment="History14"),
]


class DialogServicer(dialog_server_pb2_grpc.DialogServiceServicer):
    def SendReply(self, request: Reply, context):
        """This function reply responce from dialog bot"""
        speaker_id: str = request.speaker_id
        comment: str = request.comment

        logging.info(f"{speaker_id} talking '{comment}'")

        return Reply(speaker_id="bot", comment=f"your responce is {comment}")

    def GetReplyHistoryLimited(self, request: GetReplyHistoryLimitedParam, context):
        """This fuction responce the reply histories"""
        speaker_id = request.speaker_id
        history_from = request.history_from

        history_end = history_from + REPLY_HISTORY_LIMIT
        if speaker_id == "S1" and history_end < len(demo_reply_histories):
            limited_history = demo_reply_histories[history_from:history_end]
        else:
            limited_history = []

        return ReplyHistoryLimited(replies=limited_history)

    def GetReplyHistory(self, request, context):

        pass


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKER))
    dialog_server_pb2_grpc.add_DialogServiceServicer_to_server(DialogServicer(), server)

    server.add_insecure_port(f"{DIALOG_SERVER_IP}:{DIALOG_SERVER_PORT}")
    server.start()

    logging.info(f"server running with {DIALOG_SERVER_IP}:{DIALOG_SERVER_PORT}")
    logging.info(f"max worker is {MAX_WORKER}")

    server.wait_for_termination()
