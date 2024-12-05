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
            if not self.lock_camera_connection:  # Don't check if camera is locked for operations
                try:
                    with self.camera_lock:
                        if self.camera:
                            config = self.camera.get_config(self.context)
                            if not self.connected:
                                logging.info("Camera connection restored")
                            self.connected = True
                        else:
                            self.connected = False
                except gp.GPhoto2Error as e:
                    if self.connected:
                        logging.error(f"Camera connection lost: {e}")
                    self.connected = False
                    # Don't try to reconnect immediately - let the user do it
                    self.camera = None

            time.sleep(2)  # Check every 2 seconds

    def capture_image(self, save_path):
        """Capture an image and save it to specified path"""
        if not self.connected:
            logging.error("Cannot capture image: Camera not connected")
            return None

        try:
            # Try to acquire lock with timeout
            if not self.camera_lock.acquire(timeout=5):
                logging.error("Could not acquire camera lock")
                return None

            try:
                file_path = self.camera.capture(gp.GP_CAPTURE_IMAGE)
                camera_file = self.camera.file_get(
                    file_path.folder,
                    file_path.name,
                    gp.GP_FILE_TYPE_NORMAL
                )
                target = f"{save_path}/{file_path.name}"
                camera_file.save(target)
                logging.info(f"Image saved to {target}")
                return target
            finally:
                self.camera_lock.release()
        except gp.GPhoto2Error as e:
            logging.error(f"Failed to capture image: {e}")
            # If we get a lock error, try to reinitialize the camera
            if e.code == gp.GP_ERROR_LOCK_FAILED:
                try:
                    self.release_camera()
                    time.sleep(2)  # Wait a bit before reinitializing
                    self.initialize_camera()
                except:
                    logging.error("Failed to reinitialize camera after lock error")
            return None
        except Exception as e:
            logging.error(f"Unexpected error capturing image: {e}")
            return None

    def initialize_camera(self):
        """Initialize camera connection"""
        with self.camera_lock:
            try:
                self.camera = gp.Camera()
                self.camera.init(self.context)
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
                config = self.camera.get_config(self.context)
                setting_config = config.get_child_by_name(setting)
                if setting_config:
                    setting_config.set_value(value)
                    self.camera.set_config(config, self.context)
                    logging.info(f"Set {setting} to {value}")
                    return True
                else:
                    logging.error(f"Setting {setting} not found")
                    return False
        except gp.GPhoto2Error as e:
            logging.error(f"Failed to set {setting} to {value}: {e}")
            return False