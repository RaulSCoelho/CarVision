import functools
import sys
import threading
import time

def loading_message(message: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Flag to control the spinner animation
            running = True
            # Define the spinner animation function
            def spin():
                while running:
                    for char in '|/-\\':
                        sys.stdout.write('\r' + message + ' ' + char)
                        time.sleep(0.1)
                        sys.stdout.flush()

            # Start the spinner animation in a separate thread
            spinner_thread = threading.Thread(target=spin)
            spinner_thread.start()

            try:
                # Call the original function
                result = func(*args, **kwargs)
            finally:
                # Set flag to stop animation on exit
                running = False
                # Once the function is complete, stop the spinner
                spinner_thread.join()
                # Move the cursor back to the beginning of the line and clear it
                sys.stdout.write('\r\033[K')
                sys.stdout.flush()

            return result
        return wrapper
    return decorator
