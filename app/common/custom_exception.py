import sys 
# gives access to runtime exception information
# using here to extract file name , line number and traceback

class CustomException(Exception): #creating a custom exception class inheriting from base Exception class

    def __init__(self, message : str, error_detail: Exception = None): #Constructor with message and error_detail is the Original Exception parameter

        self.error_message = self.get_detailed_error_message(message,error_detail) #building detailed error message

        super().__init__(self.error_message) #calling parent class constructor with the detailed error message


    @staticmethod
    def get_detailed_error_message(message,error_detail): #utlility method to build detailed error message
            _,_, exec_tb = sys.exc_info() #extracting traceback object from the current exception info
            file_name = exec_tb.tb_frame.f_code.co_filename if exec_tb else "Unknown File" #extracting file name from traceback where error has occured
            line_number = exec_tb.tb_lineno if exec_tb else "unkown Line" #extracting line number from traceback where error has occured
            return f"{message} | Error : {error_detail} | File:{file_name} | Line: {line_number}" #building detailed error message string
        
    def __str__(self):
            return self.error_message
