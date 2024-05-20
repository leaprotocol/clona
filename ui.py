from nicegui import ui
import threading
from config import *
from camera import list_connected_cameras, start_camera_check, debug, zoom_out, zoom_in, enable_cancel_auto_focus, \
    disable_cancel_auto_focus, enable_auto_focus_drive, disable_auto_focus_drive, enable_viewfinder, disable_viewfinder, \
    set_eos_remote_release
from image_processing import capture_photo_and_display, analyze_image


def setup_ui():
    global image_display

    ui.button('Start Camera Connection', on_click=start_camera_check)
    ui.button('List Connected Cameras', on_click=lambda: threading.Thread(target=list_connected_cameras).start())
    ui.button('Zoom Out', on_click=lambda: threading.Thread(target=zoom_out).start())
    ui.button('Zoom In', on_click=lambda: threading.Thread(target=zoom_in).start())
    ui.button('Capture Photo and Display', on_click=lambda: threading.Thread(target=capture_photo_and_display).start())
    ui.button('Enable Cancel Auto Focus', on_click=lambda: threading.Thread(target=enable_cancel_auto_focus).start())
    ui.button('Disable Cancel Auto Focus', on_click=lambda: threading.Thread(target=disable_cancel_auto_focus).start())
    ui.button('Enable Auto Focus Drive', on_click=lambda: threading.Thread(target=enable_auto_focus_drive).start())
    ui.button('Disable Auto Focus Drive', on_click=lambda: threading.Thread(target=disable_auto_focus_drive).start())
    ui.button('Enable Viewfinder', on_click=lambda: threading.Thread(target=enable_viewfinder).start())
    ui.button('Disable Viewfinder', on_click=lambda: threading.Thread(target=disable_viewfinder).start())
    ui.button('Set EOS Remote Release', on_click=lambda: threading.Thread(target=set_eos_remote_release).start())
    ui.button('Analyze', on_click=lambda: threading.Thread(target=analyze_image).start())
    ui.button('Debug', on_click=lambda: threading.Thread(target=debug).start())
    ui.button('BUBBON', on_click=lambda: ui.notify('Button was pressed'))

    image_display = ui.image('')
