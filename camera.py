import gphoto2 as gp
import logging
import time
import threading
from config import *
from nicegui import ui

def initialize_camera():
    global camera, context
    try:
        camera = gp.Camera()
        camera.init(context)
    except gp.GPhoto2Error as e:
        logging.error(f"Failed to initialize camera: {e}")
        camera = None

def exit_camera():
    global camera, context
    if camera:
        try:
            camera.exit(context)
        except gp.GPhoto2Error as e:
            logging.error(f"Failed to release camera: {e}")
        camera = None

def list_connected_cameras():
    global context
    camera_list = gp.Camera.autodetect(context)
    cameras_str = "No cameras detected." if not camera_list else "Connected cameras:\n" + "\n".join([f"{name} on {port}" for name, port in camera_list])
    update_ui_safely(main_container, lambda: ui.notify(cameras_str, duration=5))
    print(cameras_str)

def check_camera_connection_periodically():
    global camera, context, status_label
    while True:
        try:
            camera.get_config(context)
            if status_label:
                status_label.text = 'Camera is connected'
        except Exception as e:
            notify_error(e)
            if not lock_camera_connection:
                try:
                    camera.exit(context)
                except Exception as e:
                    notify_error(e)
                finally:
                    try:
                        camera = gp.Camera()
                        context = gp.Context()
                        camera.init(context)
                    except Exception as e:
                        notify_error(e)
        time.sleep(2)

def start_camera_check():
    threading.Thread(target=check_camera_connection_periodically, daemon=True).start()

def debug():
    global camera
    print(camera.get_config(context))
    child_count = camera.get_config(context).count_children()
    print(child_count)
    for child in camera.get_config(context).get_children():
        print(child.get_label())
        print(child.get_name())
        label = '{} ({})'.format(child.get_label(), child.get_name())
        print(len(label))

def enable_cancel_auto_focus():
    global camera, context
    try:
        config = camera.get_config(context)
        autofocus_cfg = config.get_child_by_name('cancelautofocus')
        if autofocus_cfg:
            autofocus_cfg.set_value(1)
            camera.set_config(config, context)
        else:
            notify_error('cancelautofocus not found')
    except Exception as e:
        notify_error(e)

def disable_cancel_auto_focus():
    global camera, context
    try:
        config = camera.get_config(context)
        autofocus_cfg = config.get_child_by_name('cancelautofocus')
        if autofocus_cfg:
            autofocus_cfg.set_value(0)
            camera.set_config(config, context)
        else:
            notify_error('cancelautofocus not found')
    except Exception as e:
        notify_error(e)

def enable_auto_focus_drive():
    global camera, context
    try:
        config = camera.get_config(context)
        autofocusdrive_cfg = config.get_child_by_name('autofocusdrive')
        if autofocusdrive_cfg:
            autofocusdrive_cfg.set_value(1)
            camera.set_config(config, context)
        else:
            notify_error('autofocusdrive not found')
    except Exception as e:
        notify_error(e)

def disable_auto_focus_drive():
    global camera, context
    try:
        config = camera.get_config(context)
        autofocusdrive_cfg = config.get_child_by_name('autofocusdrive')
        if autofocusdrive_cfg:
            autofocusdrive_cfg.set_value(0)
            camera.set_config(config, context)
        else:
            notify_error('autofocusdrive not found')
    except Exception as e:
        notify_error(e)

def disable_viewfinder():
    set_camera_connection_lock(True)
    global camera, context
    try:
        config = camera.get_config(context)
        viewfinder_cfg = config.get_child_by_name('viewfinder')
        if viewfinder_cfg:
            viewfinder_cfg.set_value(0)
            camera.set_config(config, context)
        else:
            notify_error('viewfinder not found')
    except Exception as e:
        notify_error(e)
    finally:
        set_camera_connection_lock(False)

def enable_viewfinder():
    set_camera_connection_lock(True)
    global camera, context
    try:
        config = camera.get_config(context)
        viewfinder_cfg = config.get_child_by_name('viewfinder')
        if viewfinder_cfg:
            viewfinder_cfg.set_value(1)
            camera.set_config(config, context)
        else:
            notify_error('viewfinder not found')
    except Exception as e:
        notify_error(e)
    finally:
        set_camera_connection_lock(False)

def set_eos_remote_release():
    set_camera_connection_lock(True)
    global camera, context
    try:
        config = camera.get_config(context)
        eosremoterelease_cfg = config.get_child_by_name('eosremoterelease')
        if eosremoterelease_cfg:
            eosremoterelease_cfg.set_value('Immediate')
            camera.set_config(config, context)
        else:
            notify_error('eosremoterelease not found')
    except Exception as e:
        notify_error(e)
    finally:
        set_camera_connection_lock(False)

def zoom_out():
    global camera, context
    set_camera_connection_lock(True)
    config = camera.get_config(context)
    manualfocusdrive_cfg = config.get_child_by_name('manualfocusdrive')
    manualfocusdrive_cfg.set_value("Far 3")
    camera.set_config(config, context)
    set_camera_connection_lock(False)

def zoom_in():
    global camera, context
    set_camera_connection_lock(True)
    config = camera.get_config(context)
    manualfocusdrive_cfg = config.get_child_by_name('manualfocusdrive')
    manualfocusdrive_cfg.set_value("Near 1")
    camera.set_config(config, context)
    set_camera_connection_lock(False)
