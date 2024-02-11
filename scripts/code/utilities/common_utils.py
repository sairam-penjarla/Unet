from code.logging import logger  # Importing logger module for logging

class CommonUtils:
    def __init__(self) -> None:
        pass  # Constructor does nothing
    
    def adder(self, a:int, b:int) -> int:
        logger.info(f"Value of a: {a}")  # Logging value of a
        logger.info(f"Value of b: {b}")  # Logging value of b
        return a+b  # Returning the sum of a and b
    
    def multiply(self, a:int, b:int) -> list:
        return [a*b]  # Returning a list containing the product of a and b
