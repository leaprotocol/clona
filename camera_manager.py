from datetime import datetime

import gphoto2 as gp
from threading import Lock, Thread
import logging
import time
import exifread
import io




class CameraManager:
    def __init__(self):
        self.camera = None
        self.camera_lock = Lock()
        self.connected = False
        self.context = gp.Context()
        self.lock_camera_connection = False
        self._status_check_thread = None
        self._stop_status_check = False

    # camera_manager.py

    def get_focal_length_from_capture(self, image_data):
        """Extract focal length, camera info and exposure settings from EXIF data"""
        try:
            image_io = io.BytesIO(image_data)
            tags = exifread.process_file(image_io)

            results = {}

            # Camera model
            if 'Image Model' in tags:
                results['camera_model'] = str(tags['Image Model'])

            # Lens info
            lens_tags = ['EXIF LensModel', 'MakerNote LensType']
            for tag in lens_tags:
                if tag in tags:
                    results['lens_name'] = str(tags[tag])
                    break

            # Focal length

            if 'EXIF FocalLength' in tags:
                results['focal_length'] = str(tags['EXIF FocalLength'])

            # Focus distance
            focus_tags = ['MakerNote FocusDistance', 'EXIF SubjectDistance', 'EXIF FocusDistanceLower', 'EXIF FocusDistanceLower']
            for tag in focus_tags:
                if tag in tags:
                    try:
                        nums = str(tags[tag]).split('/')
                        if len(nums) == 2:
                            results['focus_distance'] = float(nums[0]) / float(nums[1])
                        else:
                            results['focus_distance'] = float(nums[0])
                        break
                    except ValueError:
                        continue

            # ISO - get actual value even if Auto was used
            if 'EXIF ISOSpeedRatings' in tags:
                results['iso'] = str(tags['EXIF ISOSpeedRatings'])

            # Shutter speed
            if 'EXIF ExposureTime' in tags:
                shutter = str(tags['EXIF ExposureTime'])
                if '/' in shutter:
                    results['shutterspeed'] = shutter  # Already in format like "1/60"
                else:
                    results['shutterspeed'] = str(float(shutter))  # Convert to string

            # Aperture
            if 'EXIF FNumber' in tags:
                nums = str(tags['EXIF FNumber']).split('/')
                if len(nums) == 2:
                    results['aperture'] = float(nums[0]) / float(nums[1])
                else:
                    results['aperture'] = float(nums[0])

            return results

        except Exception as e:
            logging.error(f"Error reading EXIF data: {e}")
            return {}

    def wait_for_camera_ready(self, timeout=10):
        """Wait until camera is ready for I/O operations"""
        # Note: This should be called before any camera operation
        start_time = time.time()
        ready = False

        while (time.time() - start_time) < timeout and not ready:
            try:
                # Try to get a simple property - if it works, camera is ready
                config = self.camera.get_config()
                OK, widget = gp.gp_widget_get_child_by_name(config, 'datetime')
                if OK >= gp.GP_OK:
                    ready = True
                    break
                time.sleep(2)  # Wait 2 seconds between checks
            except gp.GPhoto2Error as e:
                if e.code == -110:  # GP_ERROR_IO_IN_PROGRESS
                    logging.debug("Camera busy, waiting...")
                    time.sleep(2)  # Wait 2 seconds before retry
                    continue
                else:
                    logging.error(f"Unexpected camera error: {e}")
                    return False
            except Exception as e:
                logging.error(f"Error checking camera status: {e}")
                return False

        if not ready:
            logging.error("Camera not ready within timeout period")
        return ready


    def list_all_properties(self):
        """List all available camera properties recursively"""
        if not self.connected:
            logging.error("Cannot list properties: Camera not connected")
            return None

        def get_widget_properties(widget, prefix=''):
            """Recursively get properties from widget and its children"""
            properties = {}

            try:
                # Get current widget's properties
                name = widget.get_name()
                full_name = f"{prefix}{name}" if prefix else name

                # Only process if it's a leaf node (has no children) or has a value
                if widget.count_children() == 0:
                    try:
                        properties[full_name] = {
                            'label': widget.get_label(),
                            'value': widget.get_value(),
                            'readonly': widget.get_readonly(),
                            'type': widget.get_type(),
                            'choices': ([widget.get_choice(i) for i in range(widget.count_choices())]
                                        if widget.get_type() in [gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU]
                                        else None)
                        }
                    except Exception as e:
                        logging.debug(f"Error getting properties for {full_name}: {e}")

                # Recursively process children
                for child in widget.get_children():
                    new_prefix = f"{full_name}/" if full_name else ''
                    child_props = get_widget_properties(child, new_prefix)
                    properties.update(child_props)

            except Exception as e:
                logging.error(f"Error processing widget: {e}")

            return properties

        try:
            config = self.camera.get_config()
            properties = get_widget_properties(config)

            if not properties:
                logging.error("No properties were found")
            else:
                logging.info(f"Successfully retrieved {len(properties)} properties")

            return properties

        except Exception as e:
            logging.error(f"Error listing camera properties: {e}")
            return None

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

            camera_config = self.camera.get_config()
            settings = {}

            # List of settings to check
            settings_to_check = ['aperture', 'iso', 'shutterspeed', 'imageformat',
                                 'capturetarget', 'capture', 'exposurecompensation',
                                 'autoexposuremode', 'expprogram', 'exposuremode']

            for setting in settings_to_check:
                try:
                    OK, widget = gp.gp_widget_get_child_by_name(camera_config, setting)
                    if OK >= gp.GP_OK:
                        # Just store the current value, not the available choices
                        value = widget.get_value()
                        settings[setting] = value
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
            try:
                if not self.camera:
                    self.connected = False
                    time.sleep(2)
                    continue

                # Check if camera is ready
                ready = self.wait_for_camera_ready(timeout=2)
                if ready != self.connected:
                    if ready:
                        logging.info("Camera connection restored")
                    else:
                        logging.warning("Camera connection lost")
                self.connected = ready

            except Exception as e:
                if self.connected:
                    logging.error(f"Camera connection lost: {e}")
                self.connected = False
                self.camera = None

            time.sleep(2)  # Always wait 2 seconds between checks

    def do_camera_operation(self, operation_func, *args, **kwargs):
        """Execute a camera operation with readiness check"""
        if not self.connected:
            raise ConnectionError("Camera not connected")

        # Wait for camera to be ready
        if not self.wait_for_camera_ready():
            raise RuntimeError("Camera not ready for operation")

        # Execute the operation
        return operation_func(*args, **kwargs)

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

                # Get all settings before capture
                camera_settings = {}
                config = self.camera.get_config()

                # List of settings to capture with their correct paths
                settings_to_get = [
                    ('aperture', 'aperture'),
                    ('iso', 'iso'),
                    ('shutterspeed', 'shutterspeed'),
                    ('cameramodel', 'camera_model'),
                    ('lensname', 'lens_name'),
                    ('focal_length', 'focal_length'),
                    # These are alternative paths that might exist
                    ('d01c', 'lens_name'),  # Some cameras use this for lens name
                    ('d002', 'camera_model'),  # Some cameras use this for camera model
                ]

                for setting_name, result_key in settings_to_get:
                    try:
                        OK, widget = gp.gp_widget_get_child_by_name(config, setting_name)
                        if OK >= gp.GP_OK:
                            value = widget.get_value()
                            if value:  # Only store if we got a value
                                camera_settings[result_key] = value
                                logging.debug(f"Got {result_key}: {value}")
                    except Exception as e:
                        logging.debug(f"Could not get {setting_name}: {e}")


                # Try to get lens and camera info another way if not found
                if 'lens_name' not in camera_settings:
                    try:
                        OK, summary = self.camera.get_summary()
                        if OK >= gp.GP_OK:
                            summary_text = str(summary)
                            # Try to extract lens info from summary
                            if 'Lens:' in summary_text:
                                lens_line = [line for line in summary_text.split('\n') if 'Lens:' in line]
                                if lens_line:
                                    camera_settings['lens_name'] = lens_line[0].split('Lens:')[1].strip()
                    except Exception as e:
                        logging.debug(f"Could not get lens info from summary: {e}")

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
                # Get image data for EXIF reading
                file_data = camera_file.get_data_and_size()


                target = f"{save_path}/{file_path.name}"
                camera_file.save(target)

                # Get EXIF data
                exif_data = self.get_focal_length_from_capture(file_data)

                # Get current camera settings
                camera_settings = self.get_current_settings()

                # Merge EXIF data into camera settings, preferring EXIF values when available
                if camera_settings is None:
                    camera_settings = {}

                # Override 'Auto' ISO with actual value from EXIF if available
                if 'iso' in exif_data:
                    camera_settings['iso'] = exif_data['iso']

                if 'focal_length' in exif_data:
                    camera_settings['focal_length'] = exif_data['focal_length']

                # Add camera and lens info from EXIF
                if 'camera_model' in exif_data:
                    camera_settings['camera_model'] = exif_data['camera_model']
                if 'lens_name' in exif_data:
                    camera_settings['lens_name'] = exif_data['lens_name']

                return {
                    'path': target,
                    'metadata': {
                        'capture_time': datetime.now().isoformat(),
                        'focal_length': exif_data.get('focal_length'),
                        'focus_distance': exif_data.get('focus_distance'),
                        'camera_settings': camera_settings
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