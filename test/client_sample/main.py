import _path
from dialog_client import DialogClient

client = DialogClient()


def main():
    # test reply
    speaker_id = "speaker1"
    comment = "hello!!!"

    reply = client.SendReply(speaker_id=speaker_id, comment=comment)
    print(reply)

    # test get history
    comments = client.GetReplyHistoryLimited("S1", history_from=3)
    print(comments)

    # test get all history with stream
    comments = client.GetReplyHistory("S1")
    print(comments)


if __name__ == "__main__":
    main()
