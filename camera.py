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
        logging.info("Camera initialized successfully.")
    except gp.GPhoto2Error as e:
        logging.error(f"Failed to initialize camera in initialize_camera: {e}")
        camera = None

def exit_camera():
    global camera, context
    if camera:
        try:
            camera.exit(context)
            logging.info("Camera released successfully.")
        except gp.GPhoto2Error as e:
            logging.error(f"Failed to release camera in exit_camera: {e}")
        camera = None

def list_connected_cameras():
    global context
    try:
        camera_list = gp.Camera.autodetect(context)
        cameras_str = "No cameras detected." if not camera_list else "Connected cameras:\n" + "\n".join([f"{name} on {port}" for name, port in camera_list])
        update_ui_safely(main_container, lambda: ui.notify(cameras_str, duration=5))
        print(cameras_str)
    except gp.GPhoto2Error as e:
        logging.error(f"Failed to list connected cameras in list_connected_cameras: {e}")

def check_camera_connection_periodically():
    global camera, context, status_label
    while True:
        try:
            if camera:
                camera.get_config(context)
                if status_label:
                    status_label.text = '   Camera is connected'
            else:
                logging.error("Camera is not initialized in check_camera_connection_periodically.")
        except Exception as e:
            notify_error(f"Exception in check_camera_connection_periodically: {e}")
            if not lock_camera_connection:
                try:
                    if camera:
                        camera.exit(context)
                except Exception as e:
                    notify_error(f"Exception while exiting camera in check_camera_connection_periodically: {e}")
                finally:
                    try:
                        camera = gp.Camera()
                        context = gp.Context()
                        camera.init(context)
                    except Exception as e:
                        notify_error(f"Exception while initializing camera in check_camera_connection_periodically: {e}")
        time.sleep(2)

def start_camera_check():
    threading.Thread(target=check_camera_connection_periodically, daemon=True).start()

def debug():
    global camera
    if camera:
        try:
            print(camera.get_config(context))
            child_count = camera.get_config(context).count_children()
            print(child_count)
            for child in camera.get_config(context).get_children():
                print(child.get_label())
                print(child.get_name())
                label = '{} ({})'.format(child.get_label(), child.get_name())
                print(len(label))
        except Exception as e:
            logging.error(f"Failed to debug in debug: {e}")
    else:
        logging.error("Camera is not initialized for debugging in debug.")

def enable_cancel_auto_focus():
    global camera, context
    try:
        config = camera.get_config(context)
        autofocus_cfg = config.get_child_by_name('cancelautofocus')
        if autofocus_cfg:
            autofocus_cfg.set_value(1)
            camera.set_config(config, context)
        else:
            notify_error('cancelautofocus not found in enable_cancel_auto_focus')
    except Exception as e:
        notify_error(f"Exception in enable_cancel_auto_focus: {e}")

def disable_cancel_auto_focus():
    global camera, context
    try:
        config = camera.get_config(context)
        autofocus_cfg = config.get_child_by_name('cancelautofocus')
        if autofocus_cfg:
            autofocus_cfg.set_value(0)
            camera.set_config(config, context)
        else:
            notify_error('cancelautofocus not found in disable_cancel_auto_focus')
    except Exception as e:
        notify_error(f"Exception in disable_cancel_auto_focus: {e}")

def enable_auto_focus_drive():
    global camera, context
    try:
        config = camera.get_config(context)
        autofocusdrive_cfg = config.get_child_by_name('autofocusdrive')
        if autofocusdrive_cfg:
            autofocusdrive_cfg.set_value(1)
            camera.set_config(config, context)
        else:
            notify_error('autofocusdrive not found in enable_auto_focus_drive')
    except Exception as e:
        notify_error(f"Exception in enable_auto_focus_drive: {e}")

def disable_auto_focus_drive():
    global camera, context
    try:
        config = camera.get_config(context)
        autofocusdrive_cfg = config.get_child_by_name('autofocusdrive')
        if autofocusdrive_cfg:
            autofocusdrive_cfg.set_value(0)
            camera.set_config(config, context)
        else:
            notify_error('autofocusdrive not found in disable_auto_focus_drive')
    except Exception as e:
        notify_error(f"Exception in disable_auto_focus_drive: {e}")

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
            notify_error('viewfinder not found in disable_viewfinder')
    except Exception as e:
        notify_error(f"Exception in disable_viewfinder: {e}")
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
            notify_error('viewfinder not found in enable_viewfinder')
    except Exception as e:
        notify_error(f"Exception in enable_viewfinder: {e}")
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
            notify_error('eosremoterelease not found in set_eos_remote_release')
    except Exception as e:
        notify_error(f"Exception in set_eos_remote_release: {e}")
    finally:
        set_camera_connection_lock(False)

def zoom_out():
    global camera, context
    set_camera_connection_lock(True)
    try:
        config = camera.get_config(context)
        manualfocusdrive_cfg = config.get_child_by_name('manualfocusdrive')
        manualfocusdrive_cfg.set_value("Far 3")
        camera.set_config(config, context)
    except Exception as e:
        notify_error(f"Exception in zoom_out: {e}")
    finally:
        set_camera_connection_lock(False)

def zoom_in():
    global camera, context
    set_camera_connection_lock(True)
    try:
        config = camera.get_config(context)
        manualfocusdrive_cfg = config.get_child_by_name('manualfocusdrive')
        manualfocusdrive_cfg.set_value("Near 1")
        camera.set_config(config, context)
    except Exception as e:
        notify_error(f"Exception in zoom_in: {e}")
    finally:
        set_camera_connection_lock(False)
