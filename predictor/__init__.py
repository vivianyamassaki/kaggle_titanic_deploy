import logging


logging.basicConfig(level='INFO')
logger = logging.getLogger('uvicorn.error')
logger.propagate = False
