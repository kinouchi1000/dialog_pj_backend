import logging

import path
from inference_server.service.inference_server import serve

logging.basicConfig(level=logging.DEBUG)

serve()
