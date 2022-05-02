import logging

import path
from dialog_server.service.dialog_server import serve

logging.basicConfig(level=logging.DEBUG)

serve()
