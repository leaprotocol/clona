from nicegui import ui
import logging
from typing import Callable

class UILogger:
    def __init__(self):
        self.log_display = None
        self.setup_logging()

    def setup_logging(self):
        """Set up logging display in the UI"""
        self.log_display = ui.textarea(
            label='Application Logs',
            value=''
        ).classes('w-full h-40')

        class UILogHandler(logging.Handler):
            def __init__(self, callback: Callable[[str], None]):
                super().__init__()
                self.callback = callback

            def emit(self, record):
                log_entry = self.format(record)
                self.callback(log_entry + "\n")

        def update_log(msg):
            current = self.log_display.value
            self.log_display.value = current + msg

        handler = UILogHandler(update_log)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        ))
        logging.getLogger().addHandler(handler)

    def update_log(self, message: str, aperture: str = None):
        """Update the log display with a new message"""
        if aperture:
            logging.info(f"{message} for aperture {aperture}")
        else:
            logging.info(message)
            
        if self.log_display:
            current = self.log_display.value
            self.log_display.value = current + message + "\n" 