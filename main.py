from nicegui import ui
import threading
import gphoto2 as gp
from config import *
from camera import initialize_camera, list_connected_cameras, start_camera_check, debug, zoom_out, zoom_in, \
    enable_cancel_auto_focus, disable_cancel_auto_focus, enable_auto_focus_drive, disable_auto_focus_drive, \
    enable_viewfinder, disable_viewfinder, set_eos_remote_release
from image_processing import capture_photo_and_display, analyze_image
from ui import setup_ui


def main():
    global main_container, status_label, image_display, context
    context = gp.Context()

    main_container = ui.column()
    list_connected_cameras()
    ui.label('Clona')
    status_label = ui.label('Camera connection not initialized yet')

    setup_ui()

    ui.run()


if __name__ in {"__main__", "__mp_main__"}:
    main()
