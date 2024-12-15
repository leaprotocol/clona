from datetime import datetime

import gphoto2 as gp
from threading import Lock, Thread
import logging
import time
import exifread
import io
import os
import subprocess




class CameraManager:
    def __init__(self):
        self.context = gp.Context()
        self.camera = None
        self.camera_lock = Lock()
        self.connected = False
        self._stop_status_check = False
        self.honor_camera_settings = False
        self.autofocus_enabled = True

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

            # Lens info - Nikon specific tags
            lens_tags = [
                'EXIF LensModel',
                'MakerNote LensType',
                'MakerNote Lens',
                'MakerNote LensIDNumber',
                'MakerNote LensData'
            ]
            for tag in lens_tags:
                if tag in tags:
                    results['lens_name'] = str(tags[tag])
                    break

            # Aperture - try multiple tags
            aperture_tags = ['EXIF FNumber', 'EXIF ApertureValue', 'MakerNote FNumber']
            for tag in aperture_tags:
                if tag in tags:
                    try:
                        nums = str(tags[tag]).split('/')
                        if len(nums) == 2:
                            results['aperture'] = f"f/{float(nums[0]) / float(nums[1]):.1f}"
                        else:
                            results['aperture'] = f"f/{float(nums[0]):.1f}"
                        break
                    except (ValueError, IndexError):
                        continue

            # Focal length
            if 'EXIF FocalLength' in tags:
                focal_str = str(tags['EXIF FocalLength'])
                try:
                    # Handle fraction format (e.g., "50/1")
                    if '/' in focal_str:
                        nums = focal_str.split('/')
                        focal_length = float(nums[0]) / float(nums[1])
                    else:
                        focal_length = float(focal_str)
                    results['focal_length'] = str(int(focal_length))  # Convert to integer string
                except (ValueError, IndexError):
                    logging.error(f"Could not parse focal length: {focal_str}")

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

            return results

        except Exception as e:
            logging.error(f"Error reading EXIF data: {e}")
            return {}

    def wait_for_camera_ready(self, timeout=10):
        """Wait until camera is ready for I/O operations"""
        start_time = time.time()
        ready = False

        while (time.time() - start_time) < timeout and not ready:
            try:
                # Try to get camera config as a readiness test
                config = self.camera.get_config()
                # If we can get the config, the camera is ready
                ready = True
                break
            except gp.GPhoto2Error as e:
                if e.code in [-110, -53]:  # Common error codes for busy camera
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

    def set_honor_camera_settings(self, value):
        """Set whether to honor existing camera settings"""
        self.honor_camera_settings = value
        logging.info(f"Honor camera settings: {value}")
        
    def set_autofocus_enabled(self, value):
        """Set whether autofocus should be performed before capture"""
        self.autofocus_enabled = value
        logging.info(f"Autofocus enabled: {value}")
        
    def capture_image(self, save_path):
        """Modified capture_image method to respect new settings"""
        if not self.connected:
            logging.error("Cannot capture image: Camera not connected")
            return None
            
        try:
            with self.camera_lock:
                if not self.wait_for_camera_ready():
                    return None
                    
                # Only modify camera settings if not honoring existing ones
                if not self.honor_camera_settings:
                    if self.autofocus_enabled and not self.autofocus_enabled:
                        self.trigger_autofocus()
                    
                # Capture the image
                file_path = self.do_camera_operation(self.camera.capture, gp.GP_CAPTURE_IMAGE)
                
                # Define where to save the image
                target = os.path.join(save_path, file_path.name)
                
                # Get the camera file
                camera_file = self.do_camera_operation(
                    self.camera.file_get, file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL
                )
                
                # Save the camera file to the target destination
                camera_file.save(target)
                
                # Fetch and save metadata
                metadata = self.get_camera_settings()
                
                logging.info(f"Image captured and saved to {target}")
                return {"path": target, "metadata": metadata}
                
        except gp.GPhoto2Error as e:
            logging.error(f"Capture failed: {e}")
            return None

    def initialize_camera(self):
        """Initialize the camera with filesystem lock detection"""
        try:
            # First check if any cameras are detected
            cameras = self.list_connected_cameras()
            if not cameras:
                logging.error("No cameras detected")
                return False

            logging.info(f"Detected cameras: {cameras}")
            
            # Initialize the camera
            self.camera = gp.Camera()
            
            # Initialize with port if available
            if 'port' in cameras[0]:
                port_info_list = gp.PortInfoList()
                port_info_list.load()
                idx = port_info_list.lookup_path(cameras[0]['port'])
                port_info = port_info_list[idx]
                self.camera.set_port_info(port_info)
            
            # Try to initialize with shorter timeout first
            self.connected = self.wait_for_camera_ready(timeout=5)
            
            if not self.connected:
                # If failed, try to unmount camera from filesystem
                try:
                    # subprocess.run(['gvfs-mount', '-s', 'gphoto2'], 
                    #              stderr=subprocess.DEVNULL)
                    time.sleep(2)  # Wait for unmount
                    self.connected = self.wait_for_camera_ready(timeout=5)
                except Exception as e:
                    logging.warning(f"Failed to unmount camera: {e}")
            
            if not self.connected:
                logging.error("Failed to initialize camera: Camera not ready")
                return False
            
            logging.info("Camera initialized and ready")
            return True
            
        except gp.GPhoto2Error as e:
            logging.error(f"Failed to initialize camera: {e}")
            self.connected = False
            return False
        except Exception as e:
            logging.error(f"Unexpected error during camera initialization: {e}")
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

    def set_config_value(self, config_name, value):
        """Set a camera config value"""
        try:
            if not self.connected:
                logging.error("Camera not connected")
                return

            config = self.camera.get_config(self.context)
            widget = config.get_child_by_name(config_name)

            # Check if the widget is read-only
            if widget.get_readonly():
                logging.warning(f"Config '{config_name}' is read-only. Cannot be set to '{value}'.")
                return

            # Handle different widget types
            if widget.get_type() == gp.GP_WIDGET_RADIO:
                # Find the choice that matches the provided value
                choice_found = False
                for i in range(widget.count_choices()):
                    choice = widget.get_choice(i)
                    if choice.lower() == value.lower():
                        widget.set_value(choice)
                        choice_found = True
                        break
                if not choice_found:
                    logging.warning(f"Value '{value}' not found in choices for '{config_name}'.")
                    return
            else:
                widget.set_value(value)

            self.camera.set_config(config, self.context)
            logging.info(f"Set {config_name} to {value}")
        except gp.GPhoto2Error as e:
            logging.error(f"Failed to set {config_name} to {value}: {e}")

    def get_camera_settings(self):
        """Retrieve current camera settings"""
        camera_settings = {}
        try:
            config = self.camera.get_config()

            # List of settings to capture with their correct paths
            settings_to_get = [
                ('aperture', 'aperture'),
                ('iso', 'iso'),
                ('shutterspeed', 'shutter_speed'),
                ('cameramodel', 'camera_model'),
                ('lensname', 'lens_name'),
                ('focal_length', 'focal_length'),
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

        except Exception as e:
            logging.error(f"Error retrieving camera settings: {e}")

        return camera_settings

    def set_aperture(self, aperture_value):
        """Set camera aperture to specified value"""
        try:
            config = self.camera.get_config()
            
            # Additional Nikon-specific setting names
            aperture_settings = [
                'aperture',
                'f-number',
                'fnumber',
                'aperturevalue',
                'd002',              # Nikon PTP code
                'capturesettings/f-number',
                'capturesettings/aperture',
                '5007',             # Another Nikon PTP code
                'exposureprogram'   # Some Nikons require this first
            ]
            
            # First try to set exposure program to aperture priority
            try:
                OK, expprogram = gp.gp_widget_get_child_by_name(config, 'expprogram')
                if OK >= gp.GP_OK:
                    expprogram.set_value('A')
                    self.camera.set_config(config)
                    logging.debug("Set camera to Aperture Priority mode")
                    # Get fresh config after mode change
                    config = self.camera.get_config()
            except Exception as e:
                logging.debug(f"Could not set exposure program: {e}")

            # Try each possible setting name
            for setting_name in aperture_settings:
                try:
                    OK, widget = gp.gp_widget_get_child_by_name(config, setting_name)
                    if OK >= gp.GP_OK:
                        # For Nikon, we might need to format the aperture value
                        formatted_value = f"f/{aperture_value}"
                        try:
                            widget.set_value(formatted_value)
                        except:
                            # If formatted value fails, try numeric
                            widget.set_value(str(aperture_value))
                        
                        self.camera.set_config(config)
                        logging.debug(f"Aperture set to {aperture_value}")
                        return True
                except Exception as e:
                    logging.debug(f"Could not set {setting_name}: {e}")
                    continue
                
            raise ValueError(f"Could not set aperture to {aperture_value}")
            
        except Exception as e:
            logging.error(f"Error setting aperture: {e}")
            return False