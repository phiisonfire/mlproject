import sys

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
        