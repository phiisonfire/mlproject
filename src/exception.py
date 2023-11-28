import sys
from src.logger import logging # load the custom logging instead of the default

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    line = exc_tb.tb_lineno
    error_message = f'Error in file {filename}, line {line}, error message: {str(error)}'
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        # Overwrite the error_message property of Exception
        self.error_message = error_message_detail(error=error_message, error_detail=error_detail)
    
    def __str__(self) -> str:
        return self.error_message

if __name__ == '__main__':
    try:
        a = 1/0
    except Exception as e:
        logging.info("Test the CustomException")
        raise CustomException(e, sys)
        