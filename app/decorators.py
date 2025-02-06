import traceback
from functools import wraps
import inspect
from datetime import datetime as dt
import sys
sys.path.append("..")


def validate_input_types(func):
    """
    Decorator that validates the types of input arguments for a function
    based on its type annotations.

    Raises:
        TypeError: If the type of an argument does not match its annotation.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Validate each argument's type
        for name, value in bound_args.arguments.items():
            if name in sig.parameters:
                expected_type = sig.parameters[name].annotation
                if expected_type is not inspect._empty and not isinstance(value, expected_type):
                    raise TypeError(
                        f"Argument '{name}' must be of type {expected_type}, got {type(value)} instead."
                    )
        
        # Call the original function if validation passes
        return func(*args, **kwargs)
    
    return wrapper


def class_errors(func):
    """
    Decorator to log errors occurring in a class method to a file.
    
    Logs the following details for any exception:
    - Symbol (assumes the first argument is an instance with a `symbol` attribute)
    - Timestamp
    - Class name
    - Function name
    - Full traceback
    
    If a RecursionError occurs, the program prints "Exit", waits for user input, and exits.

    Args:
        func (callable): The class method to be decorated.

    Returns:
        callable: The wrapped function.
    """
    def just_log(*args, **kwargs):
        symbol = args[0].symbol
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            time = dt.now()
            class_name = args[0].__class__.__name__
            function_name = func.__name__
            with open("class_errors.txt", "a") as log_file:
                log_file.write("\n\nSymbol {}, Time: {} Error in class {}, function {}:\n"
                            .format(symbol, time, class_name, function_name))
                traceback.print_exc(file=log_file)
            if isinstance(e, RecursionError):
                print("Exit")
                input()
                exit()
            raise e
    return just_log
  
