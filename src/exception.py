'''
This module helps us to raise the custom exception in our entire project.
We will just raise an exception and error message will be returned.
'''
import sys

def error_message_details(error,error_detail:sys):
    '''
    This function returns the formatted error message details of given error.
    '''
    _,_,exc_tb = error_detail.exc_info()
    # in what file, what code we got the exception will be returned via this function
    error_message = f"Error occured in script {exc_tb.tb_frame.f_code.co_filename}\
                      at line number {exc_tb.tb_lineno}\
                      and error message {str(error)}."
    return error_message

class CustomException(Exception):
    '''
    This is a custom exception class which is the base class of all the custom exceptions.
    '''
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail)

    def __str__(self) -> str:
        return self.error_message
