from nicegui import ui
import logging
from typing import Callable, Optional

class CameraControls:
    def __init__(self, camera_manager, on_capture: Optional[Callable] = None):
        self.camera_manager = camera_manager
        self.on_capture = on_capture
        self.status_label = None
        self.connect_button = None
        self.disconnect_button = None
        self.capture_button = None
        self.settings_button = None
        self._initialized = False

    async def setup_controls(self):
        """Initialize camera control buttons"""
        with ui.card().classes('w-full p-4'):
            self.status_label = ui.label('○ Camera Disconnected').classes('text-red-500')
            
            with ui.row().classes('gap-2'):
                self.connect_button = ui.button(
                    'Connect Camera',
                    on_click=self.handle_connect_camera
                ).classes('bg-blue-500 text-white')
                
                self.disconnect_button = ui.button(
                    'Disconnect',
                    on_click=self.handle_disconnect_camera
                ).classes('bg-red-500 text-white')
                
                self.capture_button = ui.button(
                    'Capture Photo',
                    on_click=self.handle_capture_photo
                ).classes('bg-green-500 text-white')
                
                self.settings_button = ui.button(
                    'Camera Settings',
                    on_click=self.show_camera_settings
                ).classes('bg-gray-500 text-white')
            
            self._initialized = True
            self.update_camera_status()
        return True

    def handle_connect_camera(self):
        """Handle camera connection"""
        try:
            cameras = self.camera_manager.list_connected_cameras()
            if not cameras:
                ui.notify('No cameras detected', type='warning')
                return

            success = self.camera_manager.initialize_camera()
            if success:
                logging.info("Waiting for camera to be ready...")
                if self.camera_manager.wait_for_camera_ready(timeout=2):
                    ui.notify('Camera connected and ready', type='positive')
                else:
                    ui.notify('Camera connected but not ready yet', type='warning')
            else:
                ui.notify('Failed to connect camera', type='negative')
            
            self.update_camera_status()
        except Exception as e:
            logging.error(f"Error connecting camera: {e}")
            ui.notify('Error connecting to camera', type='negative')

    def handle_disconnect_camera(self):
        """Handle camera disconnection"""
        try:
            self.camera_manager.release_camera()
            ui.notify('Camera disconnected')
            self.update_camera_status()
        except Exception as e:
            logging.error(f"Error disconnecting camera: {e}")
            ui.notify('Error disconnecting camera', type='negative')

    def handle_capture_photo(self):
        """Handle photo capture"""
        if not self.camera_manager.connected:
            ui.notify('Camera not connected', type='negative')
            return
            
        if self.on_capture:
            self.on_capture()
        else:
            ui.notify('Capture handler not configured', type='warning')

    def show_camera_settings(self):
        """Show camera settings dialog"""
        if not self.camera_manager.connected:
            ui.notify('Camera not connected', type='negative')
            return
            
        dialog = ui.dialog()
        with dialog, ui.card().classes('p-4'):
            ui.label('Camera Settings').classes('text-xl font-bold')
            settings = self.camera_manager.get_camera_settings()
            if settings:
                for key, value in settings.items():
                    ui.label(f"{key}: {value}")
            else:
                ui.label('No settings available')
            ui.button('Close', on_click=dialog.close)

    def update_camera_status(self):
        """Update camera status display and button states"""
        if not self._initialized:
            return

        try:
            if not self.camera_manager.connected:
                self.status_label.text = '○ Camera Disconnected'
                self.status_label.classes('text-red-500', remove='text-green-500 text-yellow-500')
                self.connect_button.enable()
                self.disconnect_button.disable()
                self.capture_button.disable()
                self.settings_button.disable()
                return

            is_ready = self.camera_manager.wait_for_camera_ready(timeout=2)

            if is_ready:
                self.status_label.text = '● Camera Ready'
                self.status_label.classes('text-green-500', remove='text-red-500 text-yellow-500')
            else:
                self.status_label.text = '● Camera Busy'
                self.status_label.classes('text-yellow-500', remove='text-red-500 text-green-500')

            self.connect_button.disable()
            self.disconnect_button.enable()
            self.capture_button.enable() if is_ready else self.capture_button.disable()
            self.settings_button.enable() if is_ready else self.settings_button.disable()
            
        except Exception as e:
            logging.error(f"Error updating camera status: {e}")
            self.status_label.text = '○ Camera Error'
            self.status_label.classes('text-red-500', remove='text-green-500 text-yellow-500')
            self.connect_button.enable()
            self.disconnect_button.disable()
            self.capture_button.disable()
            self.settings_button.disable()
        