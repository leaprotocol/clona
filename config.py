import gphoto2 as gp
import logging

# Global variables
main_container = None
image_display = None
camera = None  # Global variable to hold the camera object
context = gp.Context()
lock_camera_connection = False
status_label = None

def set_camera_connection_lock(state):
    global lock_camera_connection
    lock_camera_connection = state

def update_ui_safely(container, action):
    action()

def notify_error(e):
    global status_label
    logging.error(f"An error occurred: {e}")
    if status_label:
        status_label.text = f"Error: {e}. Waiting for connection"
