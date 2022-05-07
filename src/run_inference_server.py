import sys

sys.path.append('/fairseq')
import logging

import path
from inference_server.service.inference_server import serve

logging.basicConfig(level=logging.DEBUG)

serve()
