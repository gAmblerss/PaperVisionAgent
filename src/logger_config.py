import logging
from datetime import datetime

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(f'../logs/{datetime.now().strftime("%Y%m%d-%H")}.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    #这里都产出的原因是 DEBUG级别低于INFO，console_handle包含INFO