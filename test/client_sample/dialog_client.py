from typing import List

import grpc
from common.constants import DIALOG_SERVER_IP, DIALOG_SERVER_PORT
from google.protobuf import json_format
from grpc_service import dialog_server_pb2_grpc
from grpc_service.dialog_server_pb2 import (Empty, GetReplyHistoryLimitedParam,
                                            GetReplyHistoryParam, Reply)

DIALOG_SERVER_IP="0.0.0.0"
class DialogClient:
    def __init__(self):
        self.channel = grpc.insecure_channel(f"{DIALOG_SERVER_IP}:{DIALOG_SERVER_PORT}")
        self.stub = dialog_server_pb2_grpc.DialogServiceStub(self.channel)

    def __del__(self):
        if self.channel:
            self.channel.close()

    def SendReply(self, speaker_id: str, comment: str) -> str:
        # Generate object
        reply = Reply(speaker_id=speaker_id, comment=comment)
        # request
        responce: Reply = self.stub.SendReply(reply)

        return responce.comment

    def GetReplyHistoryLimited(self, speaker_id: str, history_from: int) -> List[str]:

        # generate parameter
        param = GetReplyHistoryLimitedParam(speaker_id=speaker_id, history_from=history_from)

        # get histories
        histories: List[Reply] = self.stub.GetReplyHistoryLimited(param)
        # convert to list
        histories_dict = json_format.MessageToDict(histories)
        comments = []
        if "replies" in histories_dict:
            return comments
        for history in histories_dict["replies"]:
            comments.append(history["comment"])

        return comments

    def GetReplyHistory(self, speaker_id) -> List[str]:

        param = GetReplyHistoryParam(speaker_id=speaker_id)

        histories = self.stub.GetReplyHistory(param)
        comments = []
        
        for history in histories:
            comments.append(history.comment)

        return comments
