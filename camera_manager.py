from datetime import datetime

import gphoto2 as gp
from threading import Lock, Thread
import logging
import time

class CameraManager:
    def __init__(self):
        self.camera = None
        self.camera_lock = Lock()
        self.connected = False
        self.context = gp.Context()
        self.lock_camera_connection = False
        self._status_check_thread = None
        self._stop_status_check = False


    def wait_for_camera_ready(self, timeout=10):
        """Wait until camera is ready for I/O operations"""
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            try:
                # Try to get a simple property - if it works, camera is ready
                config = self.camera.get_config()
                OK, widget = gp.gp_widget_get_child_by_name(config, 'datetime')
                if OK >= gp.GP_OK:
                    return True
                time.sleep(0.5)  # Short sleep between attempts
            except gp.GPhoto2Error as e:
                if e.code == -110:  # GP_ERROR_IO_IN_PROGRESS
                    logging.debug("Camera busy, waiting...")
                    time.sleep(0.5)  # Wait before retry
                    continue
                else:
                    logging.error(f"Unexpected camera error: {e}")
                    return False
            except Exception as e:
                logging.error(f"Error checking camera status: {e}")
                return False

        logging.error("Timeout waiting for camera to be ready")
        return False

    def set_exposure_mode(self, mode='AV'):
        """Set camera exposure mode (AV for Aperture Priority)"""
        if not self.connected:
            logging.error("Cannot set mode: Camera not connected")
            return False

        try:
            with self.camera_lock:
                # Wait for camera to be ready
                if not self.wait_for_camera_ready():
                    logging.error("Camera not ready for exposure mode change")
                    return False

                config = self.camera.get_config()
                # Try different possible widget names for exposure mode
                for mode_setting in ['autoexposuremode', 'expprogram', 'exposuremode']:
                    OK, mode_widget = gp.gp_widget_get_child_by_name(config, mode_setting)
                    if OK >= gp.GP_OK:
                        mode_widget.set_value(mode)

                        # Wait again before applying config
                        if not self.wait_for_camera_ready():
                            logging.error("Camera not ready to apply config")
                            return False

                        self.camera.set_config(config)
                        logging.info(f"Set exposure mode to {mode}")
                        return True

                logging.error("Could not find exposure mode setting")
                return False
        except Exception as e:
            logging.error(f"Error setting exposure mode: {e}")
            return False

    def get_current_settings(self):
        """Get all current camera settings"""
        if not self.connected:
            return None

        try:
            if not self.wait_for_camera_ready():
                logging.error("Camera not ready to get settings")
                return None

            config = self.camera.get_config()
            settings = {}

            # List of settings to capture
            settings_to_check = [
                'aperture', 'iso', 'shutterspeed', 'imageformat',
                'capturetarget', 'capture', 'exposurecompensation',
                'autoexposuremode', 'expprogram', 'exposuremode',
                'datetimeutc', 'datetime'
            ]

            for setting in settings_to_check:
                try:
                    OK, widget = gp.gp_widget_get_child_by_name(config, setting)
                    if OK >= gp.GP_OK:
                        if widget.get_type() in [gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU]:
                            settings[setting] = {
                                'current': widget.get_value(),
                                'available': [widget.get_choice(i)
                                              for i in range(widget.count_choices())]
                            }
                        else:
                            settings[setting] = {
                                'current': widget.get_value(),
                                'available': None
                            }
                except:
                    continue

            return settings

        except Exception as e:
            logging.error(f"Error getting camera settings: {e}")
            return None

    def set_camera_connection_lock(self, state):
        """Lock camera connection during operations"""
        self.lock_camera_connection = state

    def start_status_check(self):
        """Start periodic camera status checking"""
        if self._status_check_thread is None:
            self._stop_status_check = False
            self._status_check_thread = Thread(
                target=self._check_camera_connection_periodically,
                daemon=True
            )
            self._status_check_thread.start()
            logging.info("Camera status checking started")

    def _check_camera_connection_periodically(self):
        """Periodically check if camera is still connected"""
        while not self._stop_status_check:
            if not self.lock_camera_connection:
                try:
                    with self.camera_lock:
                        if self.camera:
                            # Use a shorter timeout for connection check
                            if not self.wait_for_camera_ready(timeout=2):
                                self.connected = False
                                continue

                            config = self.camera.get_config(self.context)
                            if not self.connected:
                                logging.info("Camera connection restored")
                            self.connected = True
                        else:
                            self.connected = False
                except Exception as e:
                    if self.connected:
                        logging.error(f"Camera connection lost: {e}")
                    self.connected = False
                    self.camera = None

            time.sleep(2)

    def capture_image(self, save_path):
        """Capture an image and save it to specified path with metadata"""
        if not self.connected:
            logging.error("Cannot capture image: Camera not connected")
            return None

        try:
            if not self.camera_lock.acquire(timeout=5):
                logging.error("Could not acquire camera lock")
                return None

            try:
                # Wait for camera to be ready before capture
                if not self.wait_for_camera_ready():
                    logging.error("Camera not ready for capture")
                    return None

                file_path = self.camera.capture(gp.GP_CAPTURE_IMAGE)

                # Wait for camera to be ready before getting file
                if not self.wait_for_camera_ready():
                    logging.error("Camera not ready to retrieve file")
                    return None

                camera_file = self.camera.file_get(
                    file_path.folder,
                    file_path.name,
                    gp.GP_FILE_TYPE_NORMAL
                )
                target = f"{save_path}/{file_path.name}"
                camera_file.save(target)

                return {
                    'path': target,
                    'metadata': {
                        'capture_time': datetime.now().isoformat(),
                        'camera_settings': self.get_current_settings()
                    }
                }
            finally:
                self.camera_lock.release()
        except Exception as e:
            logging.error(f"Failed to capture image: {e}")
            return None


    def initialize_camera(self):
        """Initialize camera connection"""
        with self.camera_lock:
            try:
                self.camera = gp.Camera()
                self.camera.init(self.context)

                # Wait for camera to be ready after init
                if not self.wait_for_camera_ready():
                    logging.error("Camera not ready after initialization")
                    self.camera = None
                    self.connected = False
                    return False

                self.connected = True
                logging.info("Camera initialized successfully")
                return True
            except gp.GPhoto2Error as e:
                logging.error(f"Failed to initialize camera: {e}")
                self.camera = None
                self.connected = False
                return False

    def release_camera(self):
        """Release camera resources"""
        self._stop_status_check = True
        with self.camera_lock:
            if self.camera:
                try:
                    if self.wait_for_camera_ready():
                        self.camera.exit(self.context)
                    else:
                        logging.warning("Camera busy during release, forcing exit")
                    self.camera.exit(self.context)
                except gp.GPhoto2Error as e:
                    logging.error(f"Error releasing camera: {e}")
                finally:
                    self.camera = None
                    self.connected = False
                    logging.info("Camera released")


    def list_connected_cameras(self):
        """List all connected cameras"""
        try:
            camera_list = gp.Camera.autodetect(self.context)
            if not camera_list:
                logging.info("No cameras detected")
                return []
            else:
                cameras = [{"name": name, "port": port} for name, port in camera_list]
                logging.info(f"Found cameras: {cameras}")
                return cameras
        except gp.GPhoto2Error as e:
            logging.error(f"Error listing cameras: {e}")
            return []

    def get_camera_config(self):
        """Get current camera configuration"""
        if not self.connected:
            logging.error("Cannot get config: Camera not connected")
            return None

        try:
            with self.camera_lock:
                if not self.wait_for_camera_ready():
                    logging.error("Camera not ready to get config")
                    return None

                config = self.camera.get_config(self.context)
                return config
        except gp.GPhoto2Error as e:
            logging.error(f"Failed to get camera config: {e}")
            return None

    def set_camera_config(self, setting, value):
        """Set camera configuration value"""
        if not self.connected:
            logging.error("Cannot set config: Camera not connected")
            return False

        try:
            with self.camera_lock:
                if not self.wait_for_camera_ready():
                    logging.error("Camera not ready to set config")
                    return False

                config = self.camera.get_config(self.context)
                setting_config = config.get_child_by_name(setting)
                if setting_config:
                    setting_config.set_value(value)

                    # Wait again before applying config
                    if not self.wait_for_camera_ready():
                        logging.error("Camera not ready to apply config")
                        return False

                    self.camera.set_config(config, self.context)
                    logging.info(f"Set {setting} to {value}")
                    return True
                else:
                    logging.error(f"Setting {setting} not found")
                    return False
        except gp.GPhoto2Error as e:
            logging.error(f"Failed to set {setting} to {value}: {e}")
            return False