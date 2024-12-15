import os
from nicegui import app, ui, __version__
import logging
from typing import Callable
from datetime import datetime
import gphoto2 as gp
import shutil
from analysis import convert_raw_to_jpeg
from nicegui.events import UploadEventArguments
import exifread
import asyncio

class LensAnalysisUI:
    def __init__(self, camera_manager, dataset_manager):
        self.camera_manager = camera_manager
        self.dataset_manager = dataset_manager
        self.apertures = []
        self.current_dataset = None
        self.current_scenario = None
        self.log_display = None
        self.status_label = None
        self.capture_button = None
        self.connect_button = None
        self.disconnect_button = None
        self.settings_button = None
        self.dataset_list = None
        self.scenario_details = None



    def setup_ui_logging(self):
        """Set up logging display in the UI"""
        self.log_display = ui.textarea(
            label='Application Logs',
            value=''
        ).classes('w-full h-40')

        class UILogHandler(logging.Handler):
            def __init__(self, callback: Callable[[str], None]):
                super().__init__()
                self.callback = callback

            def emit(self, record):
                log_entry = self.format(record)
                self.callback(log_entry + "\n")

        def update_log(msg):
            current = self.log_display.value
            self.log_display.value = current + msg

        handler = UILogHandler(update_log)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        ))
        logging.getLogger().addHandler(handler)

    def update_camera_status(self):
        """Update camera status display and button states"""
        if not self.camera_manager.connected:
            self.status_label.text = '○ Camera Disconnected'
            self.status_label.classes('text-red-500', remove='text-green-500 text-yellow-500')
            self.connect_button.enable()
            self.disconnect_button.disable()
            self.settings_button.disable()
            return

        # Check if camera is ready
        is_ready = self.camera_manager.wait_for_camera_ready(timeout=2)

        if is_ready:
            self.status_label.text = '● Camera Ready'
            self.status_label.classes('text-green-500', remove='text-red-500 text-yellow-500')
        else:
            self.status_label.text = '● Camera Busy'
            self.status_label.classes('text-yellow-500', remove='text-red-500 text-green-500')

        self.connect_button.disable()
        self.disconnect_button.enable()
        self.settings_button.enable() if is_ready else self.settings_button.disable()

    def handle_connect_camera(self):
        """Handle camera connection"""
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

    def handle_disconnect_camera(self):
        """Handle camera disconnection"""
        self.camera_manager.release_camera()
        ui.notify('Camera disconnected')
        self.update_camera_status()

    def handle_capture_photo(self):
        """Handle photo capture"""
        try:
            if not self.camera_manager.connected :
                ui.notify('Camera not connected or unavailable', type='negative')
                return

            if not self.current_scenario:
                ui.notify('Please select a scenario first', type='warning')
                return

            if self.current_scenario['type'] in ['vignette', 'bokeh', 'distortion', 'sharpness', 'chromatic']:
                self.show_aperture_selection_dialog()
            else:
                dataset_path = os.path.join("datasets", self.current_dataset['id'], "photos")
                aperture = self.current_scenario.get('aperture', 'default_aperture')
                self.do_capture(dataset_path, aperture)
                
        except Exception as e:
            logging.error(f"Error in handle_capture_photo: {e}")
            self.handle_disconnect_camera()  # Ensure camera is properly disconnected on error
            ui.notify('Camera error - please reconnect', type='negative')

    def show_aperture_selection_dialog(self):
        """Show dialog for selecting multiple apertures for batch capture"""
        available_apertures = self.print_available_apertures()
        if not available_apertures:
            ui.notify(
                'Could not get aperture settings from camera - make sure camera is connected and supports aperture control',
                type='negative')
            return

        dialog = ui.dialog().classes('dialog-class')

        def force_close_dialog():
            """Force close the dialog and clean up"""
            try:
                dialog.close()
            except:
                pass
            ui.notify('Dialog closed', type='warning')

        with dialog, ui.card().classes('p-4 min-w-[300px]'):
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('Select Apertures').classes('text-xl')
                ui.button(
                    text='✕',
                    on_click=force_close_dialog,
                    color='green'
                )

            aperture_checkboxes = []
            for aperture in available_apertures:
                checkbox = ui.checkbox(f'{aperture}')
                aperture_checkboxes.append((aperture, checkbox))

            with ui.row().classes('gap-2 justify-end mt-4'):
                ui.button(
                    'Cancel',
                    on_click=force_close_dialog,
                    color='red'
                )

                ui.button(
                    'Capture Series',
                    on_click=lambda: self.do_batch_capture(
                        dialog,
                        [ap for ap, cb in aperture_checkboxes if cb.value]
                    )
                ).classes('bg-blue-500 text-white')

        # Add escape key and outside click handlers with force close
        dialog.on_escape = force_close_dialog
        dialog.on_click_outside = force_close_dialog

        dialog.open()

    def do_capture(self, dataset_path, aperture):
        """Regular capture with full metadata"""
        try:
            dataset_path = os.path.join("datasets", self.current_dataset['id'], "photos")
            os.makedirs(dataset_path, exist_ok=True)

            capture_result = self.camera_manager.capture_image("captures")
            if capture_result:
                temp_path = capture_result['path']
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_filename = f"{self.current_scenario['type']}_{timestamp}_{os.path.basename(temp_path)}"
                final_path = os.path.join(dataset_path, final_filename)

                shutil.move(temp_path, final_path)

                # Get all camera settings from the capture result
                camera_settings = capture_result['metadata']['camera_settings']

                # Create photo info with all metadata
                photo_info = {
                    'filename': final_filename,
                    'path': final_path,
                    'timestamp': timestamp,
                    'metadata': {
                        'scenario_type': self.current_scenario['type'],
                        'scenario_id': self.current_scenario['id'],
                        'camera_settings': camera_settings,
                        # Also store at top level for easier access
                        'aperture': camera_settings.get('aperture'),
                        'shutter_speed': camera_settings.get('shutter_speed'),
                        'iso': camera_settings.get('iso'),
                        'lens_name': camera_settings.get('lens_name'),
                        'camera_model': camera_settings.get('camera_model'),
                        'focal_length': camera_settings.get('focal_length')
                    }
                }

                if 'photos' not in self.current_scenario:
                    self.current_scenario['photos'] = []
                self.current_scenario['photos'].append(photo_info)

                if self.dataset_manager.update_scenario(
                        self.current_dataset['id'],
                        self.current_scenario
                ):
                    ui.notify('Photo captured and saved with full metadata', type='positive')
                    self.select_scenario(self.current_scenario)
                else:
                    ui.notify('Photo captured but failed to update dataset', type='warning')
            else:
                ui.notify('Failed to capture photo', type='negative')

        except Exception as e:
            ui.notify(f'Error capturing photo: {str(e)}', type='negative')
            logging.error(f"Error capturing photo: {e}")

    def handle_import_raw(self):
        """Handle RAW file import"""
        if not self.current_scenario:
            ui.notify('Please select a scenario first', type='warning')
            return

        def handle_upload(e: UploadEventArguments):
            try:
                if not e.name.lower().endswith(('.cr2', '.nef', '.arw')):
                    ui.notify('Please select a RAW file (.CR2, .NEF, or .ARW)', type='warning')
                    return

                # Create dataset directory
                dataset_path = os.path.join("datasets", self.current_dataset['id'], "photos")
                os.makedirs(dataset_path, exist_ok=True)

                # Import and process the RAW file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_filename = f"{self.current_scenario['type']}_{timestamp}_{e.name}"
                final_path = os.path.join(dataset_path, final_filename)

                # Save the uploaded file
                with open(final_path, 'wb') as f:
                    e.content.seek(0)
                    shutil.copyfileobj(e.content, f)

                # Generate JPEG preview
                preview_filename = f"{os.path.splitext(final_filename)[0]}_preview.jpg"
                preview_path = os.path.join(dataset_path, preview_filename)
                convert_raw_to_jpeg(final_path, preview_path)

                # Extract EXIF metadata using ExifRead
                with open(final_path, 'rb') as f:
                    tags = exifread.process_file(f)

                camera_settings = {
                    'aperture': str(tags.get('EXIF FNumber', '')).split('/')[0] if 'EXIF FNumber' in tags else None,
                    'shutter_speed': str(tags.get('EXIF ExposureTime', '')),
                    'iso': str(tags.get('EXIF ISOSpeedRatings', '')),
                    'lens_name': str(tags.get('EXIF LensModel', '')),
                    'camera_model': str(tags.get('Image Model', '')),
                    'focal_length': str(tags.get('EXIF FocalLength', '')).split('/')[0] if 'EXIF FocalLength' in tags else None
                }

                # Create photo info with metadata
                photo_info = {
                    'filename': final_filename,
                    'path': final_path,
                    'preview_path': preview_path,
                    'timestamp': timestamp,
                    'metadata': {
                        'scenario_type': self.current_scenario['type'],
                        'scenario_id': self.current_scenario['id'],
                        'import_method': 'direct_raw_import',
                        'camera_settings': camera_settings,
                        # Also store at top level for easier access
                        'aperture': camera_settings['aperture'],
                        'shutter_speed': camera_settings['shutter_speed'],
                        'iso': camera_settings['iso'],
                        'lens_name': camera_settings['lens_name'],
                        'camera_model': camera_settings['camera_model'],
                        'focal_length': camera_settings['focal_length']
                    }
                }

                if 'photos' not in self.current_scenario:
                    self.current_scenario['photos'] = []
                self.current_scenario['photos'].append(photo_info)

                if self.dataset_manager.update_scenario(
                        self.current_dataset['id'],
                        self.current_scenario
                ):
                    ui.notify('RAW file imported successfully', type='positive')
                    # Schedule UI refresh after upload
                    async def refresh_ui():
                        await self.select_scenario(self.current_scenario)
                    ui.timer(0.1, refresh_ui, once=True)
                else:
                    ui.notify('File imported but failed to update dataset', type='warning')

            except Exception as e:
                ui.notify(f'Error importing RAW file: {str(e)}', type='negative')
                logging.error(f"Error importing RAW file: {e}")

        # Create upload dialog
        dialog = ui.dialog()
        with dialog, ui.card().classes('p-4'):
            with ui.row().classes('w-full justify-between items-center mb-4'):
                ui.label('Import RAW File').classes('text-xl')
                ui.button(text='✕', on_click=dialog.close, color='red')
                
            # File upload component
            ui.upload(
                label='Select RAW file',
                on_upload=handle_upload,
                auto_upload=True,
                multiple=False
            ).classes('w-full mb-4')
            
            ui.label('Supported formats: .CR2, .NEF, .ARW').classes('text-sm text-gray-500')

        dialog.open()

    def create_camera_controls(self):
        """Create camera control section"""
        with ui.card().classes('w-full mb-4'):
            with ui.row().classes('w-full items-center'):
                ui.label('Camera Control').classes('text-xl mr-4')
                self.status_label = ui.label('○ Camera Disconnected').classes('text-red-500')

            with ui.row().classes('gap-2 mt-2'):
                self.connect_button = ui.button(
                    'Connect Camera',
                    on_click=self.handle_connect_camera,
                    color='blue'
                )

                self.disconnect_button = ui.button(
                    'Disconnect Camera',
                    on_click=self.handle_disconnect_camera,
                    color='red'
                )
                self.disconnect_button.disable()

            with ui.row().classes('gap-2 mt-2'):

                self.settings_button = ui.button(
                    '⚙️ Camera Settings',
                    on_click=self.show_camera_settings,
                    color='blue'
                )
                self.settings_button.disable()
                
                ui.button(
                    'List All Properties',
                    on_click=self.show_properties_dialog,
                    color='blue'
                )

            # Camera Settings Controls
            with ui.card().classes('w-full mt-2 p-2'):
                ui.label('Camera Settings').classes('text-xl mb-4')
                
                # Capture Target Setting
                with ui.row().classes('w-full items-center mb-4'):
                    ui.label('Storage Location:').classes('mr-4')
                    capture_target = ui.select(
                        options=['Memory card', 'Internal RAM'],
                        value='Memory card',
                        on_change=lambda e: self.camera_manager.set_config_value('capturetarget', e.value)
                    ).classes('w-40')
                
                # Autofocus Setting
                with ui.row().classes('w-full items-center mb-4'):
                    ui.label('Enable Autofocus:').classes('mr-4')
                    autofocus_switch = ui.switch(
                        value=False,  # Default to off
                        on_change=lambda e: self.camera_manager.set_autofocus_enabled(e.value)
                    )
                
                # Honor Camera Settings
                with ui.row().classes('w-full items-center mb-4'):
                    ui.label('Honor Camera Settings:').classes('mr-4')
                    honor_settings_switch = ui.switch(
                        value=True,  # Default to on
                        on_change=lambda e: self.camera_manager.set_honor_camera_settings(e.value)
                    )

            # Add focus control section
            with ui.card().classes('w-full mt-2 p-2'):
                ui.label('Focus Controls').classes('font-bold mb-2')

                # Auto Focus controls
                with ui.row().classes('gap-2 mt-2'):
                    ui.button(
                        'Enable Auto Focus Cancel',
                        on_click=lambda: self.handle_autofocus(True),
                        color='blue'
                    )

                    ui.button(
                        'Disable Auto Focus Cancel',
                        on_click=lambda: self.handle_autofocus(False),
                        color='blue'
                    )

                # Auto Focus Drive controls
                with ui.row().classes('gap-2 mt-2'):
                    ui.button(
                        'Enable AF Drive',
                        on_click=lambda: self.handle_autofocus_drive(True),
                        color='blue'
                    )

                    ui.button(
                        'Disable AF Drive',
                        on_click=lambda: self.handle_autofocus_drive(False),
                        color='red'
                    )

                # Viewfinder controls
                with ui.row().classes('gap-2 mt-2'):
                    ui.button(
                        'Enable Viewfinder',
                        on_click=lambda: self.handle_viewfinder(True),
                        color='blue'
                    )

                    ui.button(
                        'Disable Viewfinder',
                        on_click=lambda: self.handle_viewfinder(False),
                        color='red'
                    )

                # Focus controls
                with ui.row().classes('gap-2 mt-2'):
                    ui.label('Focus Control:').classes('font-bold self-center')

                    # Far focus controls
                    ui.button(
                        'Far 3',
                        on_click=lambda: self.handle_focus('Far 3'),
                        color='blue'
                    )

                    ui.button(
                        'Far 2',
                        on_click=lambda: self.handle_focus('Far 2'),
                        color='blue'
                    )

                    ui.button(
                        'Far 1',
                        on_click=lambda: self.handle_focus('Far 1'),
                        color='blue'
                    )

                    # Near focus controls
                    ui.button(
                        'Near 1',
                        on_click=lambda: self.handle_focus('Near 1'),
                        color='blue'
                    )

                    ui.button(
                        'Near 2',
                        on_click=lambda: self.handle_focus('Near 2'),
                        color='blue'
                    )

                    ui.button(
                        'Near 3',
                        on_click=lambda: self.handle_focus('Near 3'),
                        color='blue'
                    )

    def show_properties_dialog(self):
        """Show dialog with all camera properties"""
        if not self.camera_manager.connected:
            ui.notify('Camera not connected', type='warning')
            return

        try:
            properties = self.camera_manager.list_all_properties()
            if not properties:
                ui.notify('No camera properties found. Check logs for details.', type='negative')
                return

            dialog = ui.dialog()
            with dialog, ui.card().classes('p-4 max-w-[800px] max-h-[600px] overflow-auto'):
                ui.label('Camera Properties').classes('text-xl mb-4')

                # Properties container
                props_container = ui.column().classes('w-full gap-2')

                # Search input with proper event binding
                def on_search(e):
                    search_text = e.value.lower()
                    props_container.clear()

                    for name, details in sorted(properties.items()):
                        if search_text and search_text not in name.lower() and search_text not in str(
                                details['value']).lower():
                            continue

                        with props_container:
                            with ui.card().classes('w-full p-2'):
                                ui.label(f"Name: {name}").classes('font-bold')
                                ui.label(f"Label: {details['label']}")
                                ui.label(f"Value: {details['value']}")
                                if details.get('choices'):
                                    ui.label(f"Available choices: {', '.join(str(c) for c in details['choices'])}")
                                ui.label(f"Read-only: {'Yes' if details.get('readonly') else 'No'}")

                search = ui.input('Search properties', on_change=on_search).classes('w-full mb-4')

                # Initial population
                on_search(type('Event', (), {'value': ''})())  # Create dummy event with empty search

                with ui.row().classes('w-full justify-end mt-4'):
                    ui.button('Close', on_click=dialog.close, color='blue')

            dialog.open()

        except Exception as e:
            ui.notify(f'Error showing properties: {str(e)}', type='negative')
            logging.error(f"Error in show_properties_dialog: {e}")

    # ui.py

    def show_metadata_card(self, metadata):
        """Helper method to show consistent metadata display"""
        with ui.card().classes('w-full p-4 mb-4'):
            ui.label('Capture Settings').classes('font-bold mb-2')
            with ui.grid(columns=2).classes('gap-2'):
                cam_settings = metadata.get('camera_settings', {})

                # Get shutter speed and format it correctly
                shutter_speed = cam_settings.get('shutterspeed', 'Unknown')
                if shutter_speed != 'Unknown':
                    if isinstance(shutter_speed, str) and shutter_speed.startswith('1/'):
                        shutter_display = shutter_speed  # Already in correct format
                    else:
                        shutter_display = f"1/{shutter_speed}"
                else:
                    shutter_display = 'Unknown'

                pairs = [
                    ('Camera', cam_settings.get('camera_model', 'Unknown')),
                    ('Lens', cam_settings.get('lens_name', 'Unknown')),
                    ('Aperture', f"f/{cam_settings.get('aperture', 'Unknown')}"),
                    ('Shutter Speed', shutter_display),
                    ('ISO', cam_settings.get('iso', 'Unknown')),
                    ('Focal Length', f"{cam_settings.get('focal_length', 'Unknown')} mm")
                ]

                for label, value in pairs:
                    with ui.row().classes('gap-2'):
                        ui.label(f"{label}:").classes('font-bold text-sm')
                        ui.label(str(value)).classes('text-sm')

    def handle_focus(self, direction: str):
        """Handle focus adjustment"""
        if not self.camera_manager.connected:
            ui.notify('Camera not connected', type='warning')
            return

        try:
            config = self.camera_manager.camera.get_config()
            OK, manualfocusdrive_cfg = gp.gp_widget_get_child_by_name(config, 'manualfocusdrive')
            if OK >= gp.GP_OK:
                manualfocusdrive_cfg.set_value(direction)
                print(direction)
                self.camera_manager.camera.set_config(config)
                ui.notify(f'Focus adjusted: {direction}', type='positive')
            else:
                ui.notify('Manual focus control not found on camera', type='warning')
        except Exception as e:
            ui.notify(f'Error adjusting focus: {str(e)}', type='negative')
            logging.error(f"Focus control error: {e}")

    def handle_autofocus(self, enable: bool):
        """Handle enabling/disabling auto focus"""
        if not self.camera_manager.connected:
            ui.notify('Camera not connected', type='warning')
            return

        try:
            config = self.camera_manager.camera.get_config()
            OK, autofocus_cfg = gp.gp_widget_get_child_by_name(config, 'cancelautofocus')
            if OK >= gp.GP_OK:
                autofocus_cfg.set_value(1 if enable else 0)
                self.camera_manager.camera.set_config(config)
                ui.notify(
                    f'Auto Focus {"enabled" if enable else "disabled"}',
                    type='positive'
                )
            else:
                ui.notify('Auto Focus control not found on camera', type='warning')
        except Exception as e:
            ui.notify(f'Error controlling Auto Focus: {str(e)}', type='negative')
            logging.error(f"Auto Focus control error: {e}")

    def handle_autofocus_drive(self, enable: bool):
        """Handle enabling/disabling auto focus drive"""
        if not self.camera_manager.connected:
            ui.notify('Camera not connected', type='warning')
            return

        try:
            config = self.camera_manager.camera.get_config()
            OK, autofocus_drive_cfg = gp.gp_widget_get_child_by_name(config, 'autofocusdrive')
            if OK >= gp.GP_OK:
                autofocus_drive_cfg.set_value(1 if enable else 0)
                self.camera_manager.camera.set_config(config)
                ui.notify(
                    f'Auto Focus Drive {"enabled" if enable else "disabled"}',
                    type='positive'
                )
            else:
                ui.notify('Auto Focus Drive control not found on camera', type='warning')
        except Exception as e:
            ui.notify(f'Error controlling Auto Focus Drive: {str(e)}', type='negative')
            logging.error(f"Auto Focus Drive control error: {e}")


    def handle_viewfinder(self, enable: bool):
        """Handle enabling/disabling viewfinder"""
        if not self.camera_manager.connected:
            ui.notify('Camera not connected', type='warning')
            return

        try:
            # Set lock to prevent other operations during viewfinder changes
            self.camera_manager.set_camera_connection_lock(True)

            config = self.camera_manager.camera.get_config()
            OK, viewfinder_cfg = gp.gp_widget_get_child_by_name(config, 'viewfinder')
            if OK >= gp.GP_OK:
                if enable:
                    # First extend mirror with preview
                    self.camera_manager.camera.capture_preview()
                    # Then enable viewfinder
                    viewfinder_cfg.set_value(1)
                else:
                    # Disable viewfinder
                    viewfinder_cfg.set_value(0)
                    # Re-initiate capture preview to restore camera state
                    self.camera_manager.camera.capture_preview()

                self.camera_manager.camera.set_config(config)

                ui.notify(
                    f'Viewfinder {"enabled" if enable else "disabled"}',
                    type='positive'
                )
            else:
                ui.notify('Viewfinder control not found on camera', type='warning')
        except Exception as e:
            ui.notify(f'Error controlling Viewfinder: {str(e)}', type='negative')
            logging.error(f"Viewfinder control error: {e}")
        finally:
            # Always release the lock
            self.camera_manager.set_camera_connection_lock(False)


    def create_dataset_dialog(self):
        """Show dialog for creating a new dataset"""
        dialog = ui.dialog()
        with dialog, ui.card():
            ui.label('Create New Dataset').classes('text-xl mb-4')

            dataset_name = ui.input(
                label='Dataset Name',
                placeholder='Enter dataset name'
            ).classes('w-full mb-4')

            with ui.row().classes('gap-2 justify-end'):
                ui.button(
                    'Cancel',
                    on_click=dialog.close,
                    color='red'
                )

                ui.button(
                    'Create',
                    on_click=lambda: self.handle_create_dataset(dataset_name.value, dialog),
                    color='green'
                )

        dialog.open()

    def handle_create_dataset(self, name: str, dialog):
        """Handle dataset creation"""
        if not name:
            ui.notify('Please enter a dataset name', type='warning')
            return

        try:
            dataset = self.dataset_manager.create_dataset(name)
            if dataset:
                dialog.close()
                ui.notify(f'Dataset "{name}" created successfully', type='positive')
                self.refresh_dataset_list()
            else:
                ui.notify('Failed to create dataset', type='negative')
        except Exception as e:
            ui.notify(f'Error creating dataset: {str(e)}', type='negative')

    def create_scenario_dialog(self):
        """Show dialog for creating a new scenario"""
        if not self.current_dataset:
            ui.notify('Please select a dataset first', type='warning')
            return

        dialog = ui.dialog()
        with dialog, ui.card():
            ui.label('Create New Scenario').classes('text-xl mb-4')

            scenario_type = ui.select(
                label='Scenario Type',
                options=[
                    ('vignette', 'Vignetting Test'),
                    ('distortion', 'Distortion Analysis'),
                    ('bokeh', 'Bokeh Analysis'),
                    ('sharpness', 'Sharpness Test'),
                    ('chromatic', 'Chromatic Aberration')
                ]
            ).classes('w-full mb-4')

            focal_length = ui.number('Focal Length (mm)', value=50, min=1).classes('w-full mb-4')

            notes = ui.textarea(
                label='Notes (optional)',
                placeholder='Enter any additional notes about this test...'
            ).classes('w-full mb-4')

            with ui.row().classes('gap-2 justify-end'):
                ui.button(
                    'Cancel',
                    on_click=dialog.close,
                    color='red'
                )

                ui.button(
                    'Create',
                    on_click=lambda: self.handle_create_scenario(
                        scenario_type.value,
                        focal_length.value,
                        notes.value,
                        dialog
                    ),
                    color='green'
                    )

        dialog.open()

    def handle_create_scenario(self, scenario_type: str, focal_length: float, notes: str, dialog):
        """Handle scenario creation"""
        if not scenario_type:
            ui.notify('Please select a scenario type', type='warning')
            return

        try:
            # Get just the first part of the tuple if it is one
            actual_type = scenario_type[0] if isinstance(scenario_type, tuple) else scenario_type

            metadata = {
                'focal_length': focal_length,
                'notes': notes if notes else '',
                'created': datetime.now().strftime("%Y%m%d_%H%M%S")
            }

            scenario = self.dataset_manager.create_scenario(
                self.current_dataset['id'],
                actual_type,  # Use the extracted type
                metadata
            )

            if scenario:
                dialog.close()
                ui.notify(f'Scenario created successfully', type='positive')
                self.select_scenario(scenario)
                self.refresh_dataset_list()
            else:
                ui.notify('Failed to create scenario', type='negative')
        except Exception as e:
            ui.notify(f'Error creating scenario: {str(e)}', type='negative')

    def show_camera_settings(self):
        """Show camera settings dialog with all current camera settings"""
        dialog = ui.dialog()
        with dialog, ui.card().classes('p-4 min-w-[600px]'):
            with ui.row().classes('w-full justify-between items-center mb-4'):
                ui.label('Camera Settings').classes('text-xl')
                ui.button(text='', on_click=dialog.close, color='red')

            try:
                if not self.camera_manager.connected or not self.camera_manager.camera:
                    ui.label('Camera not connected').classes('text-red-500')
                    return

                camera_config = self.camera_manager.camera.get_config()
                
                # First, try to set camera to Aperture Priority mode
                OK, expprogram = gp.gp_widget_get_child_by_name(camera_config, 'expprogram')
                if OK >= gp.GP_OK:
                    current_mode = expprogram.get_value()
                    logging.info(f"Current exposure mode: {current_mode}")
                    
                    # Try setting to Aperture Priority mode
                    try:
                        expprogram.set_value('A')
                        self.camera_manager.camera.set_config(camera_config)
                        logging.info("Set camera to Aperture Priority mode")
                    except Exception as e:
                        logging.error(f"Failed to set Aperture Priority mode: {e}")

                # Get fresh config after mode change
                camera_config = self.camera_manager.camera.get_config()
                
                # List of possible aperture setting names for Nikon
                aperture_settings = [
                    'aperture',
                    'f-number',
                    'fnumber',
                    'aperturevalue',
                    'd002',              # Some Nikons use this PTP code
                    'capturesettings/f-number',  # Some cameras nest settings
                    'capturesettings/aperture'
                ]

                # Try each possible aperture setting name
                for setting_name in aperture_settings:
                    try:
                        OK, widget = gp.gp_widget_get_child_by_name(camera_config, setting_name)
                        if OK >= gp.GP_OK:
                            logging.info(f"Found aperture setting under name: {setting_name}")
                            choices = [widget.get_choice(i) for i in range(widget.count_choices())]
                            current = widget.get_value()
                            logging.info(f"Current aperture: {current}")
                            logging.info(f"Available apertures: {choices}")
                            return choices
                    except Exception as e:
                        logging.debug(f"Setting {setting_name} not found: {e}")

                # Debug: Print all available settings
                logging.info("Available camera settings:")
                for i in range(camera_config.count_children()):
                    child = camera_config.get_child(i)
                    try:
                        name = child.get_name()
                        if child.get_type() in [gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU]:
                            choices = [child.get_choice(j) for j in range(child.count_choices())]
                            current = child.get_value()
                            logging.info(f"{name}: current={current}, available={choices}")
                        else:
                            value = child.get_value()
                            logging.info(f"{name}: {value}")
                    except Exception as e:
                        logging.debug(f"Error getting setting {i}: {e}")

                return None

            except Exception as e:
                logging.error(f"Error showing camera settings: {e}")
                ui.label(f'Error retrieving camera settings: {str(e)}').classes('text-red-500')

            dialog.open()

    def select_dataset(self, dataset):
        """Select a dataset and show its details"""
        self.current_dataset = dataset
        self.current_scenario = None
        self.scenario_details.clear()
        
        with self.scenario_details:
            with ui.row().classes('w-full justify-between items-center mb-4'):
                ui.label(f'Dataset: {dataset["name"]}').classes('text-xl')

            if 'scenarios' in dataset and dataset['scenarios']:
                for scenario in dataset['scenarios']:
                    with ui.card().classes('w-full p-4 border-2 border-gray-200 mb-2'):
                        with ui.column().classes('gap-2'):
                            # First row - main info and buttons
                            with ui.row().classes('w-full justify-between items-center'):
                                # Left side - scenario info
                                with ui.row().classes('gap-2 items-center flex-grow'):
                                    scenario_type = scenario['type']
                                    display_type = scenario_type[1] if isinstance(scenario_type, tuple) else scenario_type.title()
                                    ui.label(display_type).classes('text-lg')
                                    if scenario['metadata'].get('focal_length'):
                                        ui.label('|').classes('text-gray-400')
                                        ui.label(f"focal length: {scenario['metadata']['focal_length']}mm").classes('text-sm text-gray-600')
                                    if scenario['metadata'].get('notes'):
                                        ui.label('|').classes('text-gray-400')
                                        ui.label(f"📝 {scenario['metadata']['notes']}").classes('text-sm text-gray-600')
                                
                                # Right side - buttons
                                with ui.row().classes('gap-2 ml-auto'):
                                    is_selected = (self.current_scenario and 
                                                 self.current_scenario['id'] == scenario['id'])
                                    select_btn = ui.button(
                                        '✓ Selected' if is_selected else '📋 Select',
                                        on_click=lambda s=scenario: self.select_scenario(s),
                                        color='gray' if is_selected else 'blue'
                                    )
                                    
                                    if is_selected:
                                        select_btn.disable()
                            
                            # Second row - photo summary
                            with ui.row().classes('gap-2 text-sm text-gray-600'):
                                photos = scenario.get('photos', [])
                                if photos:
                                    ui.label(f"{len(photos)} photos").classes('px-2 py-1 rounded')
                                    analyzed_count = sum(1 for p in photos if 'analysis' in p)
                                    if analyzed_count > 0:
                                        ui.label(f"{analyzed_count} analyzed").classes('px-2 py-1 bg-blue-100 rounded')
                                else:
                                    ui.label('No photos').classes('italic')
            else:
                ui.label('No scenarios yet - create one to begin testing').classes('text-gray-500 italic mt-4')

        self.refresh_dataset_list()

    async def select_scenario(self, scenario):
        """Select and display a scenario"""
        self.current_scenario = scenario
        self.scenario_details.clear()

        with self.scenario_details:
            # Header with scenario info
            with ui.card().classes('w-full p-4 border-2 border-gray-200 mb-4'):
                with ui.row().classes('w-full justify-between items-center'):
                    with ui.column().classes('gap-1'):
                        ui.label(f"{scenario['type'].title()}").classes('text-xl font-bold')
                        with ui.row().classes('gap-2 text-sm text-gray-600'):
                            ui.label(f"focal length: {scenario['metadata']['focal_length']}mm")
                            if scenario['metadata'].get('notes'):
                                ui.label('|').classes('text-gray-400')
                                ui.label(f"📝 {scenario['metadata']['notes']}")

            # Capture controls
            with ui.card().classes('w-full p-4 border-2 border-gray-200 mb-4'):
                with ui.row().classes('justify-between items-center'):
                    ui.label('Capture').classes('text-lg font-bold')
                    with ui.row().classes('gap-2'):
                        if scenario['type'] in ['vignette', 'chromatic']:
                            self.capture_button = ui.button(
                                '📸 Capture with Aperture',
                                on_click=self.show_aperture_selection_dialog,
                                color='green'
                            )
                        else:
                            self.capture_button = ui.button(
                                '📸 Capture Photo',
                                on_click=self.handle_capture_photo,
                                color='green'
                            )
                        
                        ui.button(
                            '📥 Import RAW',
                            on_click=self.handle_import_raw,
                            color='blue'
                        )

            # Photos grid
            photos = scenario.get('photos', [])
            with ui.card().classes('w-full p-4 border-2 border-gray-200'):
                with ui.row().classes('justify-between items-center mb-4'):
                    ui.label('Photos').classes('text-lg font-bold')
                    ui.label(f'{len(photos)} total').classes('text-sm text-gray-600')

                if not photos:
                    ui.label('No photos taken yet').classes('text-gray-500 italic')
                else:
                    with ui.grid(columns=3).classes('w-full gap-4'):
                        for photo in photos:
                            with ui.card().classes('w-full p-4 border-2 border-gray-200'):
                                with ui.column().classes('gap-2'):
                                    # Top row - metadata and file info
                                    with ui.row().classes('w-full justify-between items-center'):
                                        with ui.column().classes('gap-1'):
                                            ui.label(photo.get('filename', 'Unnamed')).classes('font-bold text-sm')
                                            timestamp = photo.get('metadata', {}).get('capture_time', 'Unknown time')
                                            ui.label(timestamp).classes('text-xs text-gray-600')
                                    
                                    # Photo preview with analysis indicator overlay
                                    with ui.card().classes('relative w-full overflow-hidden'):
                                        # Handle RAW file preview
                                        preview_path = photo.get('preview_path', photo.get('path'))
                                        if preview_path:
                                            if preview_path.lower().endswith(('.cr2', '.nef', '.arw')):
                                                jpeg_path = preview_path.rsplit('.', 1)[0] + '_preview.jpg'
                                                if not os.path.exists(jpeg_path):
                                                    convert_raw_to_jpeg(preview_path, jpeg_path)
                                                preview_path = jpeg_path
                                            
                                            if os.path.exists(preview_path):
                                                ui.image(preview_path).classes('w-full h-full object-cover')
                                            else:
                                                ui.label('Preview not available').classes('text-red-500 italic')
                                        
                                        # Analysis indicator overlay
                                        if 'analysis' in photo:
                                            with ui.element('div').classes('absolute top-2 right-2 bg-green-500 rounded-full p-1'):
                                                ui.icon('check').classes('text-white text-sm')

                                    # Action buttons
                                    with ui.row().classes('gap-2 mt-2 justify-center'):
                                        if scenario['type'] != 'bokeh':
                                            ui.button(
                                                '🔍 Analyze',
                                                on_click=lambda p=photo: self.run_photo_analysis(scenario, p),
                                                color='blue'
                                            ).classes('text-sm px-3 py-1')

                                        if 'analysis' in photo:
                                            ui.button(
                                                '📊 Results',
                                                on_click=lambda p=photo: self.show_photo_analysis(scenario, p),
                                                color='blue'
                                            ).classes('text-sm px-3 py-1')

                                        ui.button(
                                            '🗑️ Delete',
                                            on_click=lambda p=photo: self.delete_photo(scenario, p),
                                            color='red'
                                        ).classes('text-sm px-3 py-1')

        return True

    def refresh_dataset_list(self):
        """Refresh the dataset list display"""
        self.dataset_list.clear()
        try:
            datasets = self.dataset_manager.list_datasets()
            
            for dataset in datasets:
                with self.dataset_list, ui.card().classes('w-full p-4 border-2 border-gray-200'):
                    with ui.column().classes('w-full gap-2'):  # Changed to column to allow two rows
                        # First row - main info and buttons
                        with ui.row().classes('w-full justify-between items-center'):
                            # Left side - dataset info in one line
                            with ui.row().classes('gap-2 items-center flex-grow'):
                                ui.label(dataset['name']).classes('text-lg')
                                if dataset.get('camera_model') or dataset.get('lens_name'):
                                    ui.label('|').classes('text-gray-400')
                                    if dataset.get('camera_model'):
                                        ui.label(f"📷 {dataset['camera_model']}").classes('text-sm text-gray-600')
                                    if dataset.get('lens_name'):
                                        ui.label('|').classes('text-gray-400')
                                        ui.label(f"🔍 {dataset['lens_name']}").classes('text-sm text-gray-600')
                            
                            # Right side - buttons
                            with ui.row().classes('gap-2 ml-auto'):  # Added ml-auto for right alignment

                                is_selected = self.current_dataset and self.current_dataset['id'] == dataset['id']
                                select_btn = ui.button(
                                    '✓ Selected' if is_selected else '📋 Select', 
                                    on_click=lambda d=dataset: self.select_dataset(d),
                                    color='gray' if is_selected else 'blue'
                                )
                                
                                if is_selected:
                                    select_btn.disable()
                                

                                has_photos = any(scenario.get('photos', []) for scenario in dataset.get('scenarios', []))
                                camera_info_btn = ui.button('📷 Assign details from frame', 
                                    on_click=lambda d=dataset: self.update_dataset_camera_info(d),
                                    color='blue'
                                )
                                if not has_photos:
                                    camera_info_btn.disable()
                                    camera_info_btn.tooltip('No photos available in dataset')
                                    
                                ui.button('🗑️ Delete', 
                                    on_click=lambda d=dataset: self.delete_dataset(d),
                                    color='red'
                                )
                        
                        # Second row - scenarios summary
                        with ui.row().classes('gap-2 text-sm'):
                            scenarios = dataset.get('scenarios', [])
                            if scenarios:
                                scenario_counts = {}
                                for scenario in scenarios:
                                    s_type = scenario['type']
                                    photo_count = len(scenario.get('photos', []))
                                    if photo_count > 0:
                                        scenario_counts[s_type] = scenario_counts.get(s_type, 0) + photo_count
                                
                                for s_type, count in scenario_counts.items():
                                    ui.label(f"{s_type.title()}: {count} photos").classes('px-2 py-1 rounded')
                            else:
                                ui.label('No scenarios').classes('italic')
                            
        except Exception as e:
            ui.notify(f'Error refreshing dataset list: {str(e)}', type='negative')
            logging.error(f"Error in refresh_dataset_list: {e}")

    def delete_dataset(self, dataset):
        """Delete a dataset"""
        try:
            # Show confirmation dialog
            dialog = ui.dialog()
            with dialog, ui.card().classes('p-4'):
                ui.label('Confirm Delete').classes('text-xl mb-4')
                ui.label(f'Are you sure you want to delete dataset "{dataset["name"]}"?').classes('mb-4')
                ui.label('This will permanently delete all scenarios and photos.').classes('text-red-500 mb-4')

                with ui.row().classes('gap-2 justify-end'):
                    ui.button('Cancel', on_click=dialog.close, color='red')
                    ui.button(
                        'Delete',
                        on_click=lambda: self.perform_dataset_delete(dialog, dataset),
                        color='red'
                    )

            dialog.open()

        except Exception as e:
            ui.notify(f'Error showing delete dialog: {str(e)}', type='negative')
            logging.error(f"Error in delete_dataset: {e}")

    def perform_dataset_delete(self, dialog, dataset):
        """Actually perform the dataset deletion after confirmation"""
        try:
            dialog.close()
            if self.dataset_manager.delete_dataset(dataset['id']):
                ui.notify(f'Dataset deleted successfully', type='positive')
                if self.current_dataset and self.current_dataset['id'] == dataset['id']:
                    self.current_dataset = None
                    self.scenario_details.clear()
                    with self.scenario_details:
                        ui.label('No dataset selected').classes('text-gray-500 italic')
                self.refresh_dataset_list()
            else:
                ui.notify('Failed to delete dataset', type='negative')
        except Exception as e:
            ui.notify(f'Error deleting dataset: {str(e)}', type='negative')

    def create_main_page(self):
        """Create the main application page"""

        @ui.page('/')
        def main_page():
            dark_mode_enabled = app.storage.user.get('dark_mode', False)
            self.toggle_dark_mode(dark_mode_enabled)

            with ui.column().classes('w-full max-w-5xl mx-auto p-4 gap-4'):
                # Title and header
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Lens Evaluation Application').classes('text-2xl')
                    with ui.row().classes('gap-2'):
                        ui.switch(
                            text='Dark mode',
                            on_change=lambda e: self.toggle_dark_mode(e.value),
                        ).classes('self-center').bind_value(app.storage.user, 'dark_mode')
                        ui.button(
                            'Help',
                            on_click=self.show_help_dialog,
                            color='blue',
                            icon='help'
                        )

                # Camera Controls
                with ui.card().classes('w-full p-4'):
                    self.create_camera_controls()

                # Dataset and Scenario Management
                with ui.row().classes('w-full gap-4'):
                    # Dataset List
                    with ui.card().classes('w-full p-4'): 
                        with ui.row().classes('w-full justify-between items-center mb-4'):
                            ui.label('Datasets').classes('text-xl')
                            with ui.row().classes('gap-2 justify-end'):
                                ui.button(
                                    'New Dataset',
                                    on_click=self.create_dataset_dialog,
                                    color='blue'
                                )

                        # Dataset list container - make it full width
                        self.dataset_list = ui.column().classes('w-full gap-2 max-h-96 overflow-y-auto')
                        self.refresh_dataset_list()

                    # Scenario Management
                    with ui.card().classes('w-full p-4'):  
                        with ui.row().classes('justify-between items-center mb-4 w-full'):
                            ui.label('Dataset management').classes('text-xl')
                            ui.button(
                                'Add scenario',
                                on_click=self.create_scenario_dialog,
                                color='blue'
                            )

                        # Scenario details container
                        self.scenario_details = ui.card().classes('w-full p-4')
                        with self.scenario_details:
                            ui.label('No dataset selected').classes('text-gray-500 italic')

                # Log Display
                with ui.card().classes('w-full p-4'):
                    with ui.row().classes('justify-between items-center mb-2 w-full'):
                        ui.label('System Logs').classes('text-xl')
                        ui.button(
                            'Clear',
                            on_click=lambda: setattr(self.log_display, 'value', ''),
                            color='red',
                        )
                    self.setup_ui_logging()


    async def do_batch_capture(self, dialog, apertures):
        """Perform batch capture for selected apertures"""
        logging.debug(f"Received apertures for batch capture: {apertures}")

        if not apertures:
            logging.error("No apertures selected for batch capture.")
            ui.notify("No apertures selected for batch capture.", type='negative')
            return

        try:
            # Close the dialog
            dialog.close()

            # Determine the correct setting name for aperture
            camera_config = self.camera_manager.camera.get_config()
            setting_name = None
            aperture_settings = [
                'aperture', 'f-number', 'fnumber', 'aperturevalue', 'd002',
                'capturesettings/f-number', 'capturesettings/aperture'
            ]
            for name in aperture_settings:
                OK, widget = gp.gp_widget_get_child_by_name(camera_config, name)
                if OK >= gp.GP_OK:
                    setting_name = name
                    break

            if not setting_name:
                logging.error("Aperture setting name not found.")
                ui.notify("Aperture setting name not found.", type='negative')
                return

            # Set dataset path
            dataset_path = self.current_dataset.get('path', '/default/path')  # Adjust as needed

            # Proceed with batch capture logic
            total_apertures = len(apertures)
            for idx, aperture in enumerate(apertures, 1):
                try:
                    self.update_progress(f'Setting aperture {idx}/{total_apertures}...', aperture)
                    
                    # Get fresh config for each iteration
                    camera_config = self.camera_manager.camera.get_config()
                    OK, current_widget = gp.gp_widget_get_child_by_name(camera_config, setting_name)
                    
                    if OK >= gp.GP_OK and self.set_aperture(camera_config, current_widget, aperture):
                        # Capture photo
                        self.update_progress(f'Capturing photo {idx}/{total_apertures}...', aperture)
                        self.do_capture(dataset_path, aperture)
                except Exception as e:
                    logging.error(f"Error during capture at {aperture}: {e}")
                    self.update_progress(f'Error during capture {idx}/{total_apertures}: {str(e)}', aperture)
                    continue

            self.update_progress('Batch capture complete!')
            await self.select_scenario(self.current_scenario)

        except Exception as e:
            logging.error(f"Batch capture failed: {e}")
    
    def set_aperture(self,camera_config, widget, aperture_value):
        """Set the aperture value on the camera."""
        try:
            gp.gp_widget_set_value(widget, aperture_value)
            self.camera_manager.camera.set_config(camera_config)
            logging.debug(f"Aperture set to {aperture_value}")
            return True
        except Exception as e:
            logging.error(f"Failed to set aperture: {e}")
            return False




    def show_chromatic_results(self, analysis):
        """Display chromatic aberration analysis results"""
        # Show score and timestamp
        with ui.row().classes('w-full justify-between mb-4'):
            ui.label(
                f"CA Score: {analysis.get('chromatic_aberration_score', 0):.1f}/100"
            ).classes('text-xl font-bold')
            ui.label(
                f"Analyzed: {analysis.get('analysis_time', 'Unknown')}"
            ).classes('text-gray-500')

        # Show original and analyzed images side by side
        with ui.row().classes('gap-4'):
            # Original Image
            with ui.card().classes('p-2'):
                ui.label('Original Image').classes('font-bold mb-2')
                preview_path = analysis.get('preview_path')
                if preview_path and os.path.exists(preview_path):
                    ui.image(preview_path).classes('max-w-xs')
                else:
                    ui.label('Preview not available').classes('text-red-500 italic')

            # Analysis Visualization
            with ui.card().classes('p-2'):
                ui.label('CA Detection').classes('font-bold mb-2')
                if 'visualization_path' in analysis:
                    ui.image(analysis['visualization_path']).classes('max-w-xs')
                else:
                    ui.label('Visualization not available').classes('text-red-500 italic')

        # Show channel differences
        with ui.card().classes('p-4 mt-4'):
            ui.label('Channel Differences').classes('font-bold mb-2')
            diffs = analysis.get('channel_differences', {})
            with ui.grid(columns=3).classes('gap-4'):
                for channel, value in diffs.items():
                    with ui.card().classes('p-2'):
                        ui.label(channel.replace('_', '-').title())
                        ui.label(f"{value:.2f}").classes(
                            'font-bold ' +
                            ('text-green-500' if value < 10 else
                             'text-yellow-500' if value < 20 else
                             'text-red-500')
                        )

        # Analysis interpretation
        with ui.card().classes('p-4 mt-4'):
            ui.label('Analysis Interpretation').classes('font-bold mb-2')
            score = analysis.get('chromatic_aberration_score', 0)
            if score >= 80:
                msg = "Excellent - Minimal chromatic aberration detected"
            elif score >= 60:
                msg = "Good - Some color fringing but within normal range"
            else:
                msg = "Significant chromatic aberration detected"
            ui.label(msg)

    def delete_photo(self, scenario, photo_info):
        """Delete a photo from a scenario"""
        try:
            # Show confirmation dialog
            dialog = ui.dialog()
            with dialog, ui.card().classes('p-4'):
                ui.label('Confirm Delete').classes('text-xl mb-4')
                ui.label('Are you sure you want to delete this photo?').classes('mb-4')

                with ui.row().classes('gap-2 justify-end'):
                    ui.button('Cancel', on_click=dialog.close, color='red')
                    ui.button(
                        'Delete',
                        on_click=lambda: self.perform_photo_delete(dialog, scenario, photo_info),
                        color='red'
                    )

            dialog.open()

        except Exception as e:
            ui.notify(f'Error showing delete dialog: {str(e)}', type='negative')
            logging.error(f"Error in delete_photo: {e}")

    def perform_photo_delete(self, dialog, scenario, photo_info):
        """Actually perform the photo deletion after confirmation"""
        try:
            # Close the dialog first
            dialog.close()

            # Remove physical file
            if 'path' in photo_info and os.path.exists(photo_info['path']):
                os.remove(photo_info['path'])

            # Remove visualization file if it exists
            if 'analysis' in photo_info:
                viz_path = photo_info['analysis'].get('visualization_path')
                if viz_path and os.path.exists(viz_path):
                    os.remove(viz_path)

            # Remove photo from scenario's photos list
            scenario['photos'] = filter(lambda p: p['filename'] != photo_info['filename'], scenario['photos'])


            # Update scenario in dataset
            if self.dataset_manager.update_scenario(
                    self.current_dataset['id'],
                    scenario):
                ui.notify('Photo deleted successfully', type='positive')
                # Force a full UI refresh by re-selecting the scenario
                async def refresh_ui():
                    await self.select_scenario(scenario)
                ui.timer(0.1, refresh_ui, once=True)
            else:
                ui.notify('Failed to update scenario after delete', type='negative')
        except Exception as e:
            ui.notify(f'Error deleting photo: {str(e)}', type='negative')
            logging.error(f"Error in perform_photo_delete: {e}")

    def run(self):
        """Start the UI"""
        print(f'NiceGUI version: {__version__}')
        self.create_main_page()

        ui.run(    
            tailwind=True,
            show=True,
            storage_secret="asuofygbsdufvihdiud",   #whatever its only for dark mode
        )


    def show_click_debug(self, img_path, click_x, click_y):
        """Debug helper to visualize click location"""
        try:
            import cv2
            import numpy as np
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return
                
                
            # Draw click point
            cv2.circle(img, (int(click_x), int(click_y)), 5, (0, 0, 255), -1)
            
                            
            # Draw click point
            cv2.circle(img, (int(click_x), int(click_y)), 5, (0, 0, 255), -1)
            
            # Save debug image
            debug_path = os.path.join(
                os.path.dirname(img_path),
                f"click_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")

            # Draw click point
            cv2.circle(img, (int(click_x), int(click_y)), 5, (0, 0, 255), -1)
            cv2.imwrite(debug_path, img)
            return debug_path
            
        except Exception as e:
            logging.error(f"Error creating click debug: {e}")
            return None

    def print_available_apertures(self):
        """Print all available camera settings and return aperture settings"""
        try:
            if not self.camera_manager.connected or not self.camera_manager.camera:
                logging.error("Cannot get settings: Camera not connected")
                return None

            camera_config = self.camera_manager.camera.get_config()
            
            # First, try to set camera to Aperture Priority mode
            OK, expprogram = gp.gp_widget_get_child_by_name(camera_config, 'expprogram')
            if OK >= gp.GP_OK:
                current_mode = expprogram.get_value()
                logging.info(f"Current exposure mode: {current_mode}")
                
                # Try setting to Aperture Priority mode
                try:
                    expprogram.set_value('A')
                    self.camera_manager.camera.set_config(camera_config)
                    logging.info("Set camera to Aperture Priority mode")
                except Exception as e:
                    logging.error(f"Failed to set Aperture Priority mode: {e}")

            # Get fresh config after mode change
            camera_config = self.camera_manager.camera.get_config()
            
            # List of possible aperture setting names for Nikon
            aperture_settings = [
                'aperture',
                'f-number',
                'fnumber',
                'aperturevalue',
                'd002',              # Some Nikons use this PTP code
                'capturesettings/f-number',  # Some cameras nest settings
                'capturesettings/aperture'
            ]

            # Try each possible aperture setting name
            for setting_name in aperture_settings:
                try:
                    OK, widget = gp.gp_widget_get_child_by_name(camera_config, setting_name)
                    if OK >= gp.GP_OK:
                        logging.info(f"Found aperture setting under name: {setting_name}")
                        choices = [widget.get_choice(i) for i in range(widget.count_choices())]
                        current = widget.get_value()
                        logging.info(f"Current aperture: {current}")
                        logging.info(f"Available apertures: {choices}")
                        return choices
                except Exception as e:
                    logging.debug(f"Setting {setting_name} not found: {e}")

            # Debug: Print all available settings
            logging.info("Available camera settings:")
            for i in range(camera_config.count_children()):
                child = camera_config.get_child(i)
                try:
                    name = child.get_name()
                    if child.get_type() in [gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU]:
                        choices = [child.get_choice(j) for j in range(child.count_choices())]
                        current = child.get_value()
                        logging.info(f"{name}: current={current}, available={choices}")
                    else:
                        value = child.get_value()
                        logging.info(f"{name}: {value}")
                except Exception as e:
                    logging.debug(f"Error getting setting {i}: {e}")

            return None

        except Exception as e:
            logging.error(f"Error getting camera settings: {e}")
            return None

    async def show_photo_analysis(self, scenario, photo_info):
        """Show analysis dialog for a photo"""
        dialog = ui.dialog()
        with dialog, ui.card().classes('p-4 min-w-[800px]'):
            with ui.row().classes('w-full justify-between items-center'):
                title = {
                    'distortion': 'Distortion Analysis Results',
                    'vignette': 'Vignetting Analysis Results',
                    'bokeh': 'Bokeh Analysis Results',
                    'chromatic': 'Chromatic Aberration Analysis Results',
                    'sharpness': 'Sharpness Analysis Results'
                }.get(scenario['type'], 'Analysis Results')

                ui.label(title).classes('text-xl')
                ui.button(text='✕', on_click=dialog.close, color='red')

            if 'analysis' not in photo_info:
                with ui.column().classes('gap-4 items-center'):
                    ui.label('No analysis results available').classes('text-gray-500 italic')
                    if scenario['type'] == 'bokeh':
                        ui.label('Click on the image to analyze bokeh').classes('text-sm')
                    else:
                        ui.button(
                            'Analyze Now',
                            on_click=lambda: self.run_photo_analysis(scenario, photo_info, dialog),
                            color='blue'
                        )
            else:
                if scenario['type'] == 'distortion':
                    self.show_distortion_results(photo_info['analysis'])
                elif scenario['type'] == 'vignette':
                    self.show_vignetting_results(photo_info['analysis'])
                elif scenario['type'] == 'bokeh':
                    self.show_bokeh_results(photo_info['analysis'])
                elif scenario['type'] == 'sharpness':
                    self.show_sharpness_results(photo_info['analysis'])
                elif scenario['type'] == 'chromatic':
                    self.show_chromatic_results(photo_info['analysis'])

        dialog.open()

    def show_vignetting_results(self, analysis):
        """Display vignetting analysis results"""
        results = analysis.get('vignetting_results', {})

        # Show metadata if available from either source
        metadata = analysis.get('metadata', {})
        if not metadata and 'camera_settings' in analysis:
            metadata = {'camera_settings': analysis['camera_settings']}

        if metadata:
            self.show_metadata_card(metadata)

        # Show score and timestamp
        with ui.row().classes('w-full justify-between mb-4'):
            ui.label(
                f"Vignetting Score: {results.get('vignetting_score', 0):.1f}/100"
            ).classes('text-xl font-bold')
            ui.label(
                f"Analyzed: {analysis.get('analyzed_at', 'Unknown')}"
            ).classes('text-gray-500')

        # Show original and analyzed images side by side
        with ui.row().classes('gap-4'):
            # Left side - Original Image
            with ui.card().classes('p-2'):
                ui.label('Original Image').classes('font-bold mb-2')
                preview_path = analysis.get('preview_path')

                # Convert RAW to JPEG for display if needed
                if preview_path and preview_path.lower().endswith(('.cr2', '.nef', '.arw')):
                    jpeg_path = preview_path.rsplit('.', 1)[0] + '_preview.jpg'
                    if not os.path.exists(jpeg_path):
                        from analysis import convert_raw_to_jpeg
                        convert_raw_to_jpeg(preview_path, jpeg_path)
                    preview_path = jpeg_path

                if preview_path and os.path.exists(preview_path):
                    ui.image(preview_path).classes('max-w-xs')
                else:
                    ui.label('Preview not available').classes('text-red-500 italic')

            # Right side - Intensity Map
            with ui.card().classes('p-2'):
                ui.label('Intensity Map').classes('font-bold mb-2')
                if 'visualization_path' in analysis:
                    ui.image(analysis['visualization_path']).classes('max-w-xs')
                else:
                    ui.label('Visualization not available').classes('text-red-500 italic')

        # Corner measurements with colored values based on ratio
        with ui.card().classes('p-4 mt-4'):
            ui.label('Corner Measurements').classes('font-bold mb-2')
            with ui.grid(columns=2).classes('gap-4'):
                for corner, ratio in results.get('corner_ratios', {}).items():
                    with ui.card().classes('p-2'):
                        ui.label(corner.replace('_', ' ').title())
                        ui.label(f"{ratio:.2f}").classes(
                            'font-bold text-lg ' +
                            ('text-green-500' if ratio > 0.8 else
                             'text-yellow-500' if ratio > 0.6 else
                             'text-red-500')
                        )

            ui.label(f"Average corner ratio: {results.get('average_corner_ratio', 0):.2f}").classes('mt-4')

        # Analysis interpretation
        with ui.card().classes('p-4 mt-4'):
            ui.label('Analysis Interpretation').classes('font-bold mb-2')
            score = results.get('vignetting_score', 0)
            if score >= 80:
                msg = "Excellent - Very minimal vignetting detected"
            elif score >= 60:
                msg = "Good - Some light falloff but within normal range"
            else:
                msg = "Significant vignetting detected"
            ui.label(msg)

    async def run_photo_analysis(self, scenario, photo_info, dialog=None):
        """Run analysis on a photo and save results"""
        try:
            from analysis import analyze_vignetting, analyze_distortion, analyze_chromatic_aberration, analyze_sharpness

            # Create loading notification
            loading_notification = ui.notification(
                'Analyzing image...',
                type='ongoing',
                position='center',
                timeout=None
            )

            # Get the metadata from photo_info
            metadata = photo_info.get('metadata', {})

            # Run appropriate analysis based on scenario type
            if scenario['type'] == 'distortion':
                results = analyze_distortion(photo_info['path'])
                photo_info['analysis'] = {
                    **results,
                    'preview_path': photo_info['path'],
                    'analyzed_at': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'type': 'distortion',
                    'metadata': metadata
                }
            elif scenario['type'] == 'vignette':
                results = analyze_vignetting(photo_info['path'])
                photo_info['analysis'] = {
                    'vignetting_results': results,
                    'visualization_path': results.get('visualization_path'),
                    'preview_path': photo_info['path'],
                    'analyzed_at': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'type': 'vignette',
                    'metadata': metadata
                }
            elif scenario['type'] == 'chromatic':
                results = analyze_chromatic_aberration(photo_info['path'])
                photo_info['analysis'] = {
                    **results,
                    'preview_path': photo_info['path'],
                    'analyzed_at': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'type': 'chromatic',
                    'metadata': metadata
                }
            elif scenario['type'] == 'sharpness':
                results = analyze_sharpness(photo_info['path'])
                photo_info['analysis'] = {
                    **results,
                    'preview_path': photo_info['path'],
                    'analyzed_at': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'type': 'sharpness',
                    'metadata': metadata
                }

            # Save updated scenario to dataset
            if self.dataset_manager.update_scenario(
                    self.current_dataset['id'],
                    self.current_scenario
            ):
                # Close loading notification
                loading_notification.delete()
                
                # Close dialog before showing results
                if dialog:
                    dialog.close()
                
                # Show results in new dialog
                await self.show_photo_analysis(scenario, photo_info)
                ui.notify('Analysis complete', type='positive', position='bottom')
                
                # Refresh the scenario view
                await self.select_scenario(scenario)
            else:
                loading_notification.delete()
                ui.notify('Failed to save analysis results', type='negative')

        except Exception as e:
            logging.error(f"Error in photo analysis: {str(e)}")
            ui.notify(f"Analysis failed: {str(e)}", type='negative')

    def show_bokeh_results(self, analysis):
        """Display bokeh analysis results"""
        if 'metadata' in analysis:
            self.show_metadata_card(analysis['metadata'])

        # Show scores and metadata
        with ui.row().classes('w-full justify-between mb-4'):
            ui.label(
                f"Overall Score: {analysis.get('overall_score', 0):.1f}/100"
            ).classes('text-xl font-bold')

        # Show original and analyzed images side by side
        with ui.row().classes('gap-4'):
            # Original Image
            with ui.card().classes('p-2'):
                ui.label('Original Image').classes('font-bold mb-2')
                preview_path = analysis.get('preview_path')
                if preview_path and os.path.exists(preview_path):
                    ui.image(preview_path).classes('max-w-xs')
                else:
                    ui.label('Preview not available').classes('text-red-500 italic')

            # Analysis Visualization
            with ui.card().classes('p-2'):
                ui.label('Analysis').classes('font-bold mb-2')
                if 'visualization_path' in analysis:
                    ui.image(analysis['visualization_path']).classes('max-w-xs')
                else:
                    ui.label('Visualization not available').classes('text-red-500 italic')

        # Show detailed metrics
        with ui.grid(columns=3).classes('gap-4 mt-4'):
            # Shape regularity metrics
            with ui.card().classes('p-4'):
                ui.label('Shape Regularity').classes('font-bold mb-2')
                shape_metrics = analysis.get('shape_regularity', {}).get('metrics', {})
                ui.label(f"Score: {analysis.get('shape_regularity', {}).get('score', 0):.1f}/100")
                ui.label(f"Circularity: {shape_metrics.get('circularity', 0):.3f}")

            # Color fringing metrics
            with ui.card().classes('p-4'):
                ui.label('Color Fringing').classes('font-bold mb-2')
                color_metrics = analysis.get('color_fringing', {}).get('metrics', {})
                ui.label(f"Score: {analysis.get('color_fringing', {}).get('score', 0):.1f}/100")
                ui.label(f"Avg Difference: {color_metrics.get('average_color_difference', 0):.1f}")

            # Intensity metrics
            with ui.card().classes('p-4'):
                ui.label('Intensity Distribution').classes('font-bold mb-2')
                intensity_metrics = analysis.get('intensity_distribution', {}).get('metrics', {})
                ui.label(f"Score: {analysis.get('intensity_distribution', {}).get('score', 0):.1f}/100")
                ui.label(f"Uniformity: {intensity_metrics.get('std_intensity', 0):.1f}")

        # Analysis interpretation
        with ui.card().classes('p-4 mt-4'):
            ui.label('Analysis Interpretation').classes('font-bold mb-2')
            score = analysis.get('overall_score', 0)
            if score >= 80:
                msg = "Excellent - Very high sharpness and detail retention"
            elif score >= 60:
                msg = "Good - Acceptable sharpness for most purposes"
            else:
                msg = "Below average sharpness - may indicate focus issues or lens limitations"
            ui.label(msg)

    def show_sharpness_results(self, analysis):
        """Display sharpness analysis results"""
        if 'metadata' in analysis:
            self.show_metadata_card(analysis['metadata'])

        # Show score and timestamp
        with ui.row().classes('w-full justify-between mb-4'):
            ui.label(
                f"Sharpness Score: {analysis.get('sharpness_score', 0):.1f}/100"
            ).classes('text-xl font-bold')
            ui.label(
                f"Analyzed: {analysis.get('analysis_time', 'Unknown')}"
            ).classes('text-gray-500')

        # Show original and analyzed images side by side
        with ui.row().classes('gap-4'):
            # Original Image
            with ui.card().classes('p-2'):
                ui.label('Original Image').classes('font-bold mb-2')
                preview_path = analysis.get('preview_path')
                if preview_path and preview_path.lower().endswith(('.cr2', '.nef', '.arw')):
                    jpeg_path = preview_path.rsplit('.', 1)[0] + '_preview.jpg'
                    if not os.path.exists(jpeg_path):
                        convert_raw_to_jpeg(preview_path, jpeg_path)
                    preview_path = jpeg_path
                if preview_path and os.path.exists(preview_path):
                    ui.image(preview_path).classes('max-w-xs')
                else:
                    ui.label('Preview not available').classes('text-red-500 italic')

            # Analysis Visualization
            with ui.card().classes('p-2'):
                ui.label('Analysis').classes('font-bold mb-2')
                if 'visualization_path' in analysis:
                    ui.image(analysis['visualization_path']).classes('max-w-xs')
                else:
                    ui.label('Visualization not available').classes('text-red-500 italic')

        # Detailed metrics
        with ui.card().classes('p-4 mt-4'):
            ui.label('Detailed Measurements').classes('font-bold mb-2')
            with ui.grid(columns=2).classes('gap-4'):
                ui.label(f"Edge Intensity: {analysis.get('edge_intensity', 0):.2f}")
                ui.label(f"Edge Density: {analysis.get('edge_density', 0):.2f}")
                ui.label(f"Local Variance: {analysis.get('local_variance', 0):.2f}")

        # Analysis interpretation
        with ui.card().classes('p-4 mt-4'):
            ui.label('Analysis Interpretation').classes('font-bold mb-2')
            score = analysis.get('sharpness_score', 0)
            if score >= 80:
                msg = "Excellent - Very high sharpness and detail retention"
            elif score >= 60:
                msg = "Good - Acceptable sharpness for most purposes"
            else:
                msg = "Below average sharpness - may indicate focus issues or lens limitations"
            ui.label(msg)

    async def refresh_camera_settings(self, dialog):
        """Refresh camera settings display"""
        try:
            dialog.close()
            await asyncio.sleep(0.5)  # Give time for the dialog to close
            self.show_camera_settings()
        except Exception as e:
            logging.error(f"Error refreshing camera settings: {e}")
            ui.notify('Failed to refresh camera settings', type='negative')

    def update_progress(self, message, aperture=None):
        """Update the progress of the batch capture process"""
        if aperture:
            logging.info(f"{message} for aperture {aperture}")
        else:
            logging.info(message)
        # Update the UI or log with the progress message
        if self.log_display:
            current = self.log_display.value
            self.log_display.value = current + message + "\n"
    
    def extract_camera_info_from_dataset(self, dataset):
        """Extract camera and lens info from first photo in dataset"""
        try:
            logging.info(f"Attempting to extract camera info from dataset: {dataset.get('name')}")
            
            if not dataset.get('scenarios'):
                logging.error(f"Dataset {dataset.get('name')} has no scenarios")
                return None
                
            logging.info(f"Found {len(dataset['scenarios'])} scenarios")
            
            for idx, scenario in enumerate(dataset['scenarios']):
                logging.info(f"Checking scenario {idx + 1}: {scenario.get('name')}")
                
                if not scenario.get('photos'):
                    logging.info(f"No photos in scenario {idx + 1}")
                    continue
                    
                logging.info(f"Found {len(scenario['photos'])} photos in scenario")
                first_photo = scenario['photos'][0]
                
                # Truncate large data structures in logging
                photo_info = {
                    'filename': first_photo.get('filename'),
                    'metadata': first_photo.get('metadata')
                }
                logging.info(f"First photo info: {photo_info}")
                
                metadata = first_photo.get('metadata', {})
                camera_model, lens_name = self.extract_camera_info(metadata)
                
                logging.info(f"Extracted camera_model: {camera_model}, lens_name: {lens_name}")
                
                if camera_model and lens_name:
                    return {
                        'camera_model': camera_model,
                        'lens_name': lens_name
                    }
                else:
                    logging.error("Found metadata but no camera or lens information")
                    
            logging.error("No usable camera information found in any scenario")
            return None
            
        except Exception as e:
            logging.error(f"Error extracting camera info: {e}")
            return None
    
    def update_dataset_camera_info(self, dataset):
        """Update dataset with camera info from first photo"""
        try:
            camera_info = self.extract_camera_info_from_dataset(dataset)
            if not camera_info:
                ui.notify('No camera information found in dataset photos', type='negative')
                return False
            
            # Create a copy of the dataset to modify
            updated_dataset = dataset.copy()
            updated_dataset['camera_model'] = camera_info['camera_model']
            updated_dataset['lens_name'] = camera_info['lens_name']
            
            if self.dataset_manager.update_dataset(updated_dataset):
                ui.notify(
                    f'Updated camera info: {camera_info["camera_model"]} with {camera_info["lens_name"]}', 
                    type='positive'
                )
                self.refresh_dataset_list()
                return True
            else:
                ui.notify('Failed to update dataset', type='negative')
                return False
            
        except Exception as e:
            ui.notify(f'Error updating camera info: {str(e)}', type='negative')
            logging.error(f"Error updating dataset camera info: {e}")
            return False
    
    def extract_camera_info(self, metadata):
        """
        Extract camera and lens information from photo metadata.
        
        Args:
            metadata (dict): Photo metadata dictionary
            
        Returns:
            tuple: (camera_model, lens_name) or (None, None) if not found
        """
        try:
            # Check if camera_settings exists in metadata
            if 'camera_settings' in metadata:
                settings = metadata['camera_settings']
                camera_model = settings.get('camera_model')
                lens_name = settings.get('lens_name')
                return camera_model, lens_name
                
            # Fallback to top-level metadata
            camera_model = metadata.get('camera_model')
            lens_name = metadata.get('lens_name')
            return camera_model, lens_name
            
        except Exception as e:
            logging.error(f"Error extracting camera info from metadata: {e}")
            return None, None

    def show_help_dialog(self):
        """Show help dialog with about section and links"""
        dialog = ui.dialog()
        with dialog, ui.card().classes('p-6 max-w-[800px]'):
            # Header
            with ui.row().classes('w-full justify-between items-center mb-6'):
                ui.label('Lens Evaluation Application').classes('text-2xl font-bold')
                ui.button(text='✕', on_click=dialog.close, color='red')
            
            # About section
            with ui.card().classes('mb-6 p-4'):
                ui.label('About').classes('text-xl font-bold mb-2')
                ui.label('''
                    This application is designed to evaluate various lens properties using at-home 
                    printable test charts. It provides comprehensive analysis of lens characteristics 
                    including sharpness, bokeh, distortions, chromatic aberration, and vignetting.
                ''').classes('text-gray-700 mb-4')
                
                # Features list
                ui.label('Key Features:').classes('font-bold mb-2')
                features = [
                    '- Image Acquisition in RAW format',
                    '- Camera control through USB interface',
                    '- Advanced Image Processing and Analysis',
                    '- Evaluation of Multiple Lens Properties',
                    '- User-friendly Web Interface',
                    '- Session Management (Save and Load)',
                ]
                for feature in features:
                    ui.label(feature).classes('ml-4 mb-1')

            # Links section
            with ui.card().classes('p-4'):
                ui.label('Links & Resources').classes('text-xl font-bold mb-4')
                with ui.row().classes('gap-4'):
                    ui.link(
                        'GitHub Repository', 
                        'https://github.com/N4M3L355/clona',
                        new_tab=True
                    ).classes('text-blue-500 hover:text-blue-700')
                    ui.link(
                        'Report Issues', 
                        'https://github.com/N4M3L355/clona/issues',
                        new_tab=True
                    ).classes('text-blue-500 hover:text-blue-700')
                    
            # Contact info
            with ui.card().classes('mt-6 p-4'):
                ui.label('Contact').classes('text-xl font-bold mb-2')
                ui.label('For questions or inquiries, please contact:')
                ui.link(
                    'lea.kralova00@gmail.com',
                    'mailto:lea.kralova00@gmail.com'
                ).classes('text-blue-500 hover:text-blue-700')

        dialog.open()

    def toggle_dark_mode(self, enabled):
        """Toggle dark mode and save preference"""
        
        if enabled:
            ui.dark_mode().enable()
            ui.colors(
                primary='#4B7AB8',    # Vibrant dark blue
                blue='#4B7AB8',       # Same blue for consistency
                green='#4BA87B',      # Vibrant dark green
                red='#B84B6F',        # Vibrant dark rose
                yellow='#B8A04B',     # Vibrant dark gold
                positive='#4BA87B',   # Same as green
                negative='#B84B6F',   # Same as red
                warning='#B8A04B',    # Same as yellow
            )
        else:
            ui.dark_mode().disable()
            ui.colors(
            )