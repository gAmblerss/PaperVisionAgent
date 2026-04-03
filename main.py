import logging
from src.logger_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

logger.info("Logging started")
logger.debug(f"Logging level: {logger.level}")