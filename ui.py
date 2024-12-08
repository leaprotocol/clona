import os
from nicegui import ui
import logging
import time
from typing import Callable
from datetime import datetime
import gphoto2 as gp  # Add this import
import shutil
import cv2
from analysis import analyze_bokeh, analyze_sharpness, convert_raw_to_jpeg

class LensAnalysisUI:
    def __init__(self, camera_manager, dataset_manager):
        self.camera_manager = camera_manager
        self.dataset_manager = dataset_manager
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
            self.capture_button.disable()
            self.settings_button.disable()
            return

        # Check if camera is ready
        is_ready = self.camera_manager.wait_for_camera_ready(timeout=1)  # Short timeout for UI

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

    def handle_connect_camera(self):
        """Handle camera connection"""
        success = self.camera_manager.initialize_camera()
        ui.notify('Camera connected successfully' if success else 'Failed to connect camera',
                  type='positive' if success else 'negative')
        self.update_camera_status()

    def handle_disconnect_camera(self):
        """Handle camera disconnection"""
        self.camera_manager.release_camera()
        ui.notify('Camera disconnected')
        self.update_camera_status()

    def handle_capture_photo(self):
        """Handle photo capture"""
        if not self.camera_manager.connected:
            ui.notify('Camera not connected', type='negative')
            return

        if not self.current_scenario:
            ui.notify('Please select a scenario first', type='warning')
            return

        # Include 'distortion' in the list of scenarios that use aperture selection
        if self.current_scenario['type'] in ['vignette', 'bokeh', 'distortion', 'sharpness', 'chromatic']:
            self.show_aperture_selection_dialog()
        else:
            self.do_capture()

    def show_aperture_selection_dialog(self):
        """Show dialog for selecting multiple apertures for batch capture"""
        available_apertures = self.print_available_apertures()
        if not available_apertures:
            ui.notify(
                'Could not get aperture settings from camera - make sure camera is connected and supports aperture control',
                type='negative')
            return

        dialog = ui.dialog().classes('dialog-class')  # Add a class for styling

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
                    on_click=force_close_dialog
                ).classes('text-gray-500')

            aperture_switches = []
            for aperture in available_apertures:
                switch = ui.switch(f'{aperture}')
                aperture_switches.append((aperture, switch))

            with ui.row().classes('gap-2 justify-end mt-4'):
                ui.button(
                    'Cancel',
                    on_click=force_close_dialog
                ).classes('bg-gray-500 text-white')

                ui.button(
                    'Capture Series',
                    on_click=lambda: self.do_batch_capture(
                        dialog,
                        [ap for ap, sw in aperture_switches if sw.value]
                    )
                ).classes('bg-green-500 text-white')

        # Add escape key and outside click handlers with force close
        dialog.on_escape = force_close_dialog
        dialog.on_click_outside = force_close_dialog

        dialog.open()

    def do_capture(self):
        """Regular capture with full metadata"""
        try:
            dataset_path = os.path.join("datasets", self.current_dataset['id'], "photos")
            os.makedirs(dataset_path, exist_ok=True)

            # Capture now returns both path and metadata
            capture_result = self.camera_manager.capture_image("captures")
            if capture_result:
                temp_path = capture_result['path']
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_filename = f"{self.current_scenario['type']}_{timestamp}_{os.path.basename(temp_path)}"
                final_path = os.path.join(dataset_path, final_filename)

                shutil.move(temp_path, final_path)

                # Create photo info with all metadata
                photo_info = {
                    'filename': final_filename,
                    'path': final_path,
                    'timestamp': timestamp,
                    'metadata': {
                        'scenario_type': self.current_scenario['type'],
                        'scenario_id': self.current_scenario['id'],
                        'camera_settings': capture_result['metadata']['camera_settings']
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

    def create_camera_controls(self):
        """Create camera control section"""
        with ui.card().classes('w-full mb-4'):
            with ui.row().classes('w-full items-center'):
                ui.label('Camera Control').classes('text-xl mr-4')
                self.status_label = ui.label('○ Camera Disconnected').classes('text-red-500')

            with ui.row().classes('gap-2 mt-2'):
                self.connect_button = ui.button(
                    'Connect Camera',
                    on_click=self.handle_connect_camera
                ).classes('bg-blue-500 text-white')

                self.disconnect_button = ui.button(
                    'Disconnect Camera',
                    on_click=self.handle_disconnect_camera
                ).classes('bg-red-500 text-white')
                self.disconnect_button.disable()

            with ui.row().classes('gap-2 mt-2'):
                self.capture_button = ui.button(
                    '📸 Capture Photo',
                    on_click=self.handle_capture_photo
                ).classes('bg-green-500 text-white')
                self.capture_button.disable()

                self.settings_button = ui.button(
                    '⚙️ Camera Settings',
                    on_click=self.show_camera_settings
                ).classes('bg-purple-500 text-white')
                self.settings_button.disable()

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
                    on_click=dialog.close
                ).classes('bg-gray-500 text-white')

                ui.button(
                    'Create',
                    on_click=lambda: self.handle_create_dataset(dataset_name.value, dialog)
                ).classes('bg-green-500 text-white')

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
                    on_click=dialog.close
                ).classes('bg-gray-500 text-white')

                ui.button(
                    'Create',
                    on_click=lambda: self.handle_create_scenario(
                        scenario_type.value,
                        focal_length.value,
                        notes.value,
                        dialog
                    )
                ).classes('bg-green-500 text-white')

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
                ui.button(text='✕', on_click=dialog.close).classes('text-gray-500')

            try:
                if not self.camera_manager.connected or not self.camera_manager.camera:
                    ui.label('Camera not connected').classes('text-red-500')
                    return

                camera_config = self.camera_manager.camera.get_config()
                settings_to_check = [
                    'aperture', 'iso', 'shutterspeed', 'imageformat', 'capturetarget',
                    'capturesizeclass', 'capture', 'shootingmode', 'exposurecompensation',
                    'flashcompensation', 'datetimeutc', 'datetime',
                    # Focus distance settings
                    'focusmode', 'focusdistance', 'manualfocusdistance', 'focallength', 'd162',
                    # Exposure mode settings
                    'expprogram', 'exposureprogram', 'autoexposuremode', 'exposuremode',
                    'capturemode', 'shooting mode', 'shootingmode', 'd054'
                ]

                # Create sections for different types of settings
                sections = {
                    'Exposure': ['aperture', 'shutterspeed', 'iso', 'exposurecompensation',
                                 'expprogram', 'exposuremode', 'autoexposuremode'],
                    'Focus': ['focusmode', 'focusdistance', 'manualfocusdistance', 'focallength'],
                    'Capture': ['imageformat', 'capturetarget', 'capturesizeclass', 'capture'],
                    'Other': []
                }

                for section_name, setting_keys in sections.items():
                    with ui.card().classes('p-4 mb-4'):
                        ui.label(section_name).classes('font-bold mb-2')
                        settings_found = False

                        for setting in setting_keys:
                            try:
                                OK, widget = gp.gp_widget_get_child_by_name(camera_config, setting)
                                if OK >= gp.GP_OK:
                                    settings_found = True
                                    with ui.row().classes('w-full justify-between items-center mb-2'):
                                        ui.label(f"{setting}:").classes('font-mono')

                                        if widget.get_type() in [gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU]:
                                            choices = [widget.get_choice(i)
                                                       for i in range(widget.count_choices())]
                                            current = widget.get_value()
                                            choices_str = f"Current: {current} (Available: {', '.join(choices)})"
                                            ui.label(choices_str).classes('text-sm text-gray-600')
                                        else:
                                            value = widget.get_value()
                                            ui.label(str(value)).classes('text-sm text-gray-600')
                            except Exception as e:
                                logging.debug(f"Setting {setting} not available: {e}")

                        if not settings_found:
                            ui.label('No settings found in this category').classes('text-gray-500 italic')

                # Add refresh button at the bottom
                with ui.row().classes('justify-end mt-4'):
                    ui.button('Refresh Settings',
                              on_click=lambda: self.refresh_camera_settings(dialog)
                              ).classes('bg-blue-500 text-white')

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
            ui.label(f'Dataset: {dataset["name"]}').classes('text-xl')
            if 'metadata' in dataset and 'lens_details' in dataset['metadata']:
                ui.label(f'Lens: {dataset["metadata"]["lens_details"]}')

            if 'scenarios' in dataset and dataset['scenarios']:
                with ui.column().classes('mt-4 gap-2'):
                    ui.label('Scenarios:').classes('font-bold')
                    for scenario in dataset['scenarios']:
                        with ui.card().classes('w-full p-2'):
                            with ui.row().classes('justify-between items-center'):
                                # Handle both string and tuple types
                                scenario_type = scenario['type']
                                display_type = scenario_type[1] if isinstance(scenario_type,
                                                                              tuple) else scenario_type.title()
                                ui.label(display_type).classes('font-bold')
                                ui.button('Select',
                                          on_click=lambda s=scenario: self.select_scenario(s)
                                          ).classes('bg-blue-500 text-white p-1 text-sm')
            else:
                ui.label('No scenarios yet - create one to begin testing').classes('text-gray-500 italic mt-4')

        self.refresh_dataset_list()

    def select_scenario(self, scenario):
        """Select and display a scenario"""
        self.current_scenario = scenario
        self.scenario_details.clear()

        with self.scenario_details:
            ui.label(f"Scenario: {scenario['type'].title()}").classes('text-xl mb-2')
            ui.label(f"Focal Length: {scenario['metadata']['focal_length']}mm").classes('mb-1')

            if scenario['metadata'].get('notes'):
                with ui.card().classes('w-full mb-4 bg-gray-50'):
                    ui.label('Notes:').classes('font-bold')
                    ui.label(scenario['metadata']['notes'])

            # Photos and Analysis section
            photos = scenario.get('photos', [])
            with ui.card().classes('w-full mb-4'):
                ui.label('Photos').classes('font-bold mb-2')

                if not photos:
                    ui.label('No photos taken yet').classes('text-gray-500 italic')
                else:
                    with ui.grid(columns=2).classes('gap-4'):
                        for photo in photos:
                            with ui.card().classes('p-4'):
                                ui.label(f"Filename: {photo['filename']}").classes('font-bold')
                                ui.label(f"Taken: {photo['timestamp']}")

                                # Display photo with click handler for bokeh analysis
                                if scenario['type'] == 'bokeh':
                                    preview_path = photo['path']
                                    # Convert RAW to JPEG for viewing if needed
                                    if preview_path.lower().endswith(('.cr2', '.nef', '.arw')):
                                        jpg_path = preview_path.rsplit('.', 1)[0] + '_preview.jpg'
                                        if not os.path.exists(jpg_path):
                                            from analysis import convert_raw_to_jpeg
                                            convert_raw_to_jpeg(preview_path, jpg_path)
                                        preview_path = jpg_path

                                    # Create a container for the image
                                    with ui.card().classes('relative'):
                                        img = ui.image(preview_path).classes('cursor-pointer max-w-full')
                                        img.on("click", lambda e, s=scenario, p=photo: self.handle_bokeh_click(e, s, p))
                                        ui.label('Click on the bokeh circle to analyze').classes(
                                            'text-sm text-gray-500')
                                else:
                                    # Regular image display for non-bokeh scenarios
                                    preview_path = photo['path']
                                    if preview_path.lower().endswith(('.cr2', '.nef', '.arw')):
                                        jpg_path = preview_path.rsplit('.', 1)[0] + '_preview.jpg'
                                        if not os.path.exists(jpg_path):
                                            from analysis import convert_raw_to_jpeg
                                            convert_raw_to_jpeg(preview_path, jpg_path)
                                        preview_path = jpg_path
                                    ui.image(preview_path).classes('max-w-full')

                                with ui.row().classes('gap-2 mt-2'):
                                    if scenario['type'] == 'distortion':
                                        ui.button(
                                            'Analyze Distortion',
                                            on_click=lambda p=photo: self.run_photo_analysis(scenario, p)
                                        ).classes('bg-blue-500 text-white p-2')
                                    elif scenario['type'] == 'vignette':
                                        ui.button(
                                            'Analyze Vignette',
                                            on_click=lambda p=photo: self.run_photo_analysis(scenario, p)
                                        ).classes('bg-blue-500 text-white p-2')
                                    elif scenario['type'] == 'chromatic':
                                        ui.button(
                                            'Analyze Chromatic Aberration',
                                            on_click=lambda p=photo: self.run_photo_analysis(scenario, p)
                                        ).classes('bg-blue-500 text-white p-2')
                                    elif scenario['type'] == 'bokeh':
                                        if 'analysis' not in photo:
                                            ui.label('Click image to analyze bokeh').classes('text-sm text-gray-500')

                                    if 'analysis' in photo:
                                        ui.button(
                                            'View Results',
                                            on_click=lambda p=photo: self.show_photo_analysis(scenario, p)
                                        ).classes('bg-green-500 text-white p-2')

                                    if 'metadata' in photo and 'aperture' in photo['metadata']:
                                        ui.label(f"f/{photo['metadata']['aperture']}").classes('text-gray-500')

                                    ui.button(
                                        'Delete Photo',
                                        on_click=lambda p=photo: self.delete_photo(scenario, p)
                                    ).classes('bg-red-500 text-white p-2')

            # Capture controls
            with ui.card().classes('w-full p-4'):
                ui.label('Capture').classes('font-bold mb-2')
                with ui.row().classes('gap-2'):
                    if scenario['type'] in ['vignette', 'chromatic']:
                        ui.button(
                            '📸 Capture with Aperture',
                            on_click=self.show_aperture_selection_dialog
                        ).classes('bg-green-500 text-white')
                    else:
                        ui.button(
                            '📸 Capture Photo',
                            on_click=self.handle_capture_photo
                        ).classes('bg-green-500 text-white')

                    if scenario['type'] == 'bokeh':
                        with ui.card().classes('mt-4 p-2 bg-gray-50'):
                            ui.label('Bokeh Analysis Instructions:').classes('font-bold')
                            ui.label('1. Take a photo of a defocused point light source')
                            ui.label('2. Click on the bokeh circle in the image to analyze')
                            ui.label('3. Analysis will measure shape, color fringing, and intensity distribution')

            # For debugging - show current scenario state
            logging.debug(f"Current scenario state: {scenario}")


    def refresh_dataset_list(self):
        """Refresh the dataset list display"""
        self.dataset_list.clear()

        try:
            datasets = self.dataset_manager.list_datasets()
            if not datasets:
                with self.dataset_list:
                    ui.label('No datasets available').classes('text-gray-500 italic')
            else:
                for dataset in datasets:
                    with self.dataset_list:
                        with ui.card().classes('w-full p-2'):
                            with ui.row().classes('justify-between items-center'):
                                ui.label(dataset['name']).classes(
                                    'font-bold text-blue-500'
                                    if dataset == self.current_dataset
                                    else ''
                                )
                                with ui.row().classes('gap-1'):
                                    ui.button(
                                        'Select',
                                        on_click=lambda d=dataset: self.select_dataset(d)
                                    ).classes('bg-blue-500 text-white p-1 text-sm')
                                    ui.button(
                                        'Delete',
                                        on_click=lambda d=dataset: self.delete_dataset(d)
                                    ).classes('bg-red-500 text-white p-1 text-sm')
        except Exception as e:
            with self.dataset_list:
                ui.label(f'Error loading datasets: {str(e)}').classes('text-red-500')

    def delete_dataset(self, dataset):
        """Delete a dataset"""
        try:
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
            with ui.column().classes('w-full max-w-5xl mx-auto p-4 gap-4'):
                # Title and header
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Lens Analysis Application').classes('text-2xl')
                    with ui.row().classes('gap-2'):
                        ui.button('Help',
                                  on_click=lambda: ui.notify('Help functionality coming soon', type='info')
                                  ).classes('bg-gray-500 text-white')

                # Camera Controls
                with ui.card().classes('w-full p-4'):
                    self.create_camera_controls()

                # Dataset and Scenario Management
                with ui.row().classes('w-full gap-4'):
                    # Left panel - Dataset List
                    with ui.card().classes('w-1/3 p-4'):
                        with ui.row().classes('justify-between items-center mb-4'):
                            ui.label('Datasets').classes('text-xl')
                            ui.button(
                                'New Dataset',
                                on_click=self.create_dataset_dialog
                            ).classes('bg-green-500 text-white')

                        # Dataset list container
                        self.dataset_list = ui.column().classes('gap-2 max-h-96 overflow-y-auto')
                        self.refresh_dataset_list()

                    # Right panel - Scenario Management
                    with ui.card().classes('w-2/3 p-4'):
                        with ui.row().classes('justify-between items-center mb-4'):
                            ui.label('Scenario').classes('text-xl')
                            ui.button(
                                'New Scenario',
                                on_click=self.create_scenario_dialog
                            ).classes('bg-blue-500 text-white')

                        # Scenario details container
                        self.scenario_details = ui.card().classes('w-full p-4')
                        with self.scenario_details:
                            ui.label('No dataset selected').classes('text-gray-500 italic')

                # Log Display
                with ui.card().classes('w-full p-4'):
                    with ui.row().classes('justify-between items-center mb-2'):
                        ui.label('System Logs').classes('text-xl')
                        ui.button(
                            'Clear',
                            on_click=lambda: setattr(self.log_display, 'value', '')
                        ).classes('bg-gray-500 text-white text-sm')
                    self.setup_ui_logging()

                # Start periodic status updates
                ui.timer(2.0, self.update_camera_status)

    # ui.py - fix the do_batch_capture method

    def do_batch_capture(self, dialog, apertures):
        """Capture multiple photos with different aperture values"""
        if not self.current_scenario:
            ui.notify('Please select a scenario first', type='warning')
            return

        if not apertures:
            ui.notify('Please select at least one aperture value', type='warning')
            return

        logging.info(f"Starting batch capture for apertures: {apertures}")
        dialog.close()

        # Create dataset directory
        dataset_path = os.path.join("datasets", self.current_dataset['id'], "photos")
        os.makedirs(dataset_path, exist_ok=True)

        # Set exposure mode first, with retry
        retries = 3
        while retries > 0:
            try:
                self.camera_manager.set_exposure_mode('AV')
                break
            except Exception as e:
                retries -= 1
                logging.warning(f"Failed to set exposure mode, retrying... ({e})")
                time.sleep(2)  # Wait before retry

        for idx, aperture in enumerate(apertures, 1):
            try:
                # Set aperture
                camera_config = self.camera_manager.camera.get_config()
                OK, aperture_widget = gp.gp_widget_get_child_by_name(camera_config, 'aperture')
                if OK >= gp.GP_OK:
                    aperture_widget.set_value(aperture)
                    self.camera_manager.camera.set_config(camera_config)
                    time.sleep(1)  # Wait for setting to take effect

                # Capture photo
                capture_result = self.camera_manager.capture_image("captures")
                if capture_result and 'path' in capture_result:  # Check for path in result
                    temp_path = capture_result['path']  # Get actual path from result

                    # Move to dataset location with scenario type in filename
                    final_filename = f"{self.current_scenario['type']}_f{aperture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.CR2"
                    final_path = os.path.join(dataset_path, final_filename)
                    shutil.move(temp_path, final_path)

                    # Add to scenario with metadata
                    photo_info = {
                        'filename': final_filename,
                        'path': final_path,
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                        'metadata': {
                            'aperture': aperture,
                            'camera_settings': capture_result.get('metadata', {}).get('camera_settings', {})
                        }
                    }

                    if 'photos' not in self.current_scenario:
                        self.current_scenario['photos'] = []
                    self.current_scenario['photos'].append(photo_info)

                    # Persist changes
                    self.dataset_manager.update_scenario(
                        self.current_dataset['id'],
                        self.current_scenario
                    )

                    ui.notify(f'Photo captured at f/{aperture}', type='positive')

            except Exception as e:
                logging.error(f"Error during capture at {aperture}: {e}")
                ui.notify(f'Error during capture at f/{aperture}: {str(e)}', type='negative')
                continue

        # Final UI update
        self.select_scenario(self.current_scenario)
        ui.notify('Batch capture complete', type='positive')

    def print_available_apertures(self):
        """Print all available camera settings and return aperture settings"""
        try:
            if not self.camera_manager.connected or not self.camera_manager.camera:
                logging.error("Cannot get settings: Camera not connected")
                return None

            camera_config = self.camera_manager.camera.get_config()
            settings_to_check = [
                'aperture', 'iso', 'shutterspeed', 'imageformat', 'capturetarget',
                'capturesizeclass', 'capture', 'shootingmode', 'exposurecompensation',
                'flashcompensation', 'datetimeutc', 'datetime',
                # Focus distance settings - different cameras use different names
                'focusmode',  # Some cameras
                'focusdistance',  # Some cameras
                'manualfocusdistance',  # Some cameras
                'focallength',  # Some cameras use this for focus distance
                'd162',  # Some PTP cameras use this for focus distance
                # Different cameras might use different names for exposure mode
                'expprogram',  # Some cameras
                'exposureprogram',  # Some cameras
                'autoexposuremode',  # Some cameras
                'exposuremode',  # Some cameras
                'capturemode',  # Some cameras
                'shooting mode',  # Some cameras
                'shootingmode',  # Some cameras
                'd054'  # Some PTP cameras use this for exposure mode
            ]

            available_settings = {}
            aperture_choices = None

            for setting in settings_to_check:
                try:
                    OK, widget = gp.gp_widget_get_child_by_name(camera_config, setting)
                    if OK >= gp.GP_OK:
                        # For settings that have choices
                        if widget.get_type() in [gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU]:
                            choices = [widget.get_choice(i) for i in range(widget.count_choices())]
                            current_value = widget.get_value()
                            available_settings[setting] = {
                                'choices': choices,
                                'current': current_value
                            }
                            if setting == 'aperture':
                                aperture_choices = choices
                        else:
                            # For other types of settings, just get current value
                            available_settings[setting] = {
                                'current': widget.get_value(),
                                'choices': None
                            }
                except Exception as e:
                    logging.debug(f"Setting {setting} not available: {e}")

            # Log all available settings
            logging.info("Available camera settings:")
            for setting, values in available_settings.items():
                if values['choices']:
                    logging.info(f"{setting}: current={values['current']}, available={values['choices']}")
                else:
                    logging.info(f"{setting}: current={values['current']}")

            return aperture_choices

        except Exception as e:
            logging.error(f"Error getting camera settings: {e}")
            return None
    def show_photo_analysis(self, scenario, photo_info):
        """Show analysis dialog for a photo"""
        dialog = ui.dialog()
        with dialog, ui.card().classes('p-4 min-w-[800px]'):
            with ui.row().classes('w-full justify-between items-center'):
                title = {
                    'distortion': 'Distortion Analysis Results',
                    'vignette': 'Vignetting Analysis Results',
                    'bokeh': 'Bokeh Analysis Results',
                    'chromatic': 'Chromatic Aberration Analysis Results'
                }.get(scenario['type'], 'Analysis Results')

                ui.label(title).classes('text-xl')
                ui.button(text='✕', on_click=dialog.close).classes('text-gray-500')

            if 'analysis' not in photo_info:
                with ui.column().classes('gap-4 items-center'):
                    ui.label('No analysis results available').classes('text-gray-500 italic')
                    if scenario['type'] == 'bokeh':
                        ui.label('Click on the image to analyze bokeh').classes('text-sm')
                    else:
                        ui.button(
                            'Analyze Now',
                            on_click=lambda: self.run_photo_analysis(scenario, photo_info, dialog)
                        ).classes('bg-blue-500 text-white')
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

    def show_sharpness_results(self, analysis):
        """Display sharpness analysis results"""
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
        with ui.card().classes('p-4 mt-4 bg-gray-50'):
            ui.label('Analysis Interpretation').classes('font-bold mb-2')
            score = analysis.get('sharpness_score', 0)
            if score >= 80:
                msg = "Excellent - Very high sharpness and detail retention"
            elif score >= 60:
                msg = "Good - Acceptable sharpness for most purposes"
            else:
                msg = "Below average sharpness - may indicate focus issues or lens limitations"
            ui.label(msg)

    def show_bokeh_results(self, analysis):
        """Display bokeh analysis results"""
        logging.info(f"Showing bokeh results with metadata: {analysis.get('metadata', {})}")

        # Show scores and metadata
        with ui.row().classes('w-full justify-between mb-4'):
            ui.label(
                f"Overall Score: {analysis.get('overall_score', 0):.1f}/100"
            ).classes('text-xl font-bold')

        # Show camera settings
        with ui.card().classes('w-full p-4 mb-4 bg-gray-50'):
            with ui.row().classes('gap-4 justify-start'):
                metadata = analysis.get('metadata', {})
                logging.info(f"Displaying camera settings from metadata: {metadata}")

                if 'aperture' in metadata:
                    ui.label(f"f/{metadata['aperture']}").classes('font-mono')
                if 'shutter_speed' in metadata:
                    ui.label(f"1/{metadata['shutter_speed']}s").classes('font-mono')
                if 'iso' in metadata:
                    ui.label(f"ISO {metadata['iso']}").classes('font-mono')

        # Show original and analyzed images side by side
        with ui.row().classes('gap-4'):
            # Original Image
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

    def handle_bokeh_click(self, event, scenario, photo_info):
        """Handle click on photo for bokeh analysis"""
        try:
            click_x = event.args.get("offsetX", 0)
            click_y = event.args.get("offsetY", 0)

            logging.info(f"Processing bokeh click at coordinates: ({click_x}, {click_y})")
            logging.info(f"Photo info received: {photo_info}")
            logging.info(f"Photo metadata: {photo_info.get('metadata', {})}")

            ui.notify('Starting bokeh analysis...', type='info')

            metadata = photo_info.get('metadata', {})
            results = analyze_bokeh(photo_info['path'], click_x, click_y, metadata)

            logging.info(f"Analysis results metadata: {results.get('metadata', {})}")

            # Store results
            photo_info['analysis'] = {
                **results,
                'preview_path': photo_info['path'],
                'analyzed_at': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'type': 'bokeh'
            }

            logging.info(f"Final analysis stored: {photo_info['analysis'].get('metadata', {})}")

            # Save updated scenario
            if self.dataset_manager.update_scenario(
                    self.current_dataset['id'],
                    scenario
            ):
                ui.notify('Bokeh analysis complete', type='positive')
                self.select_scenario(scenario)
                self.show_photo_analysis(scenario, photo_info)
            else:
                ui.notify('Failed to save analysis results', type='negative')

        except Exception as e:
            ui.notify(f'Error during bokeh analysis: {str(e)}', type='negative')
            logging.error(f"Bokeh analysis error: {e}")

    def show_vignetting_results(self, analysis):
        """Display vignetting analysis results"""
        results = analysis.get('vignetting_results', {})
        logging.info(f"Showing vignetting results with preview path: {analysis.get('preview_path')}")

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
        with ui.card().classes('p-4 mt-4 bg-gray-50'):
            ui.label('Analysis Interpretation').classes('font-bold mb-2')
            score = results.get('vignetting_score', 0)
            if score >= 80:
                msg = "Excellent - Very minimal vignetting detected"
            elif score >= 60:
                msg = "Good - Some light falloff but within normal range"
            else:
                msg = "Significant vignetting detected"
            ui.label(msg)

    def show_distortion_results(self, analysis):
        """Display distortion analysis results"""
        # Add logging
        logging.info(f"Showing distortion results with preview path: {analysis.get('preview_path')}")

        # Show score and timestamp
        with ui.row().classes('w-full justify-between mb-4'):
            ui.label(
                f"Distortion Score: {analysis.get('distortion_score', 0):.1f}/100"
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

            # Right side - Detection Visualization
            with ui.card().classes('p-2'):
                ui.label('Grid Detection').classes('font-bold mb-2')
                if 'visualization_path' in analysis and os.path.exists(analysis['visualization_path']):
                    ui.image(analysis['visualization_path']).classes('max-w-xs')
                else:
                    ui.label('Visualization not available').classes('text-red-500 italic')

        # Detailed metrics
        with ui.card().classes('p-4 mt-4'):
            ui.label('Detailed Measurements').classes('font-bold mb-2')
            with ui.grid(columns=2).classes('gap-4'):
                h_dev = analysis.get('horizontal_deviations', [])
                v_dev = analysis.get('vertical_deviations', [])
                ui.label(f"Number of Lines: {len(h_dev) + len(v_dev)}").classes('font-bold')
                ui.label(f"Average Deviation: {analysis.get('average_deviation', 0):.2f} pixels")
                ui.label(f"Horizontal Lines: {len(h_dev)}").classes('text-blue-500')
                ui.label(f"Vertical Lines: {len(v_dev)}").classes('text-blue-500')

        # Analysis interpretation
        with ui.card().classes('p-4 mt-4 bg-gray-50'):
            ui.label('Analysis Interpretation').classes('font-bold mb-2')
            score = analysis.get('distortion_score', 0)
            distortion_type = analysis.get('type', 'unknown')
            if score >= 80:
                msg = "Excellent - Minimal distortion detected"
            elif score >= 60:
                msg = f"Good - Some {distortion_type} distortion but within normal range"
            else:
                msg = f"Significant {distortion_type} distortion detected"
            ui.label(msg)

            if distortion_type == 'barrel':
                ui.label("Barrel distortion: lines curve outward from center").classes('mt-2 text-sm text-gray-600')
            elif distortion_type == 'pincushion':
                ui.label("Pincushion distortion: lines curve inward toward center").classes(
                    'mt-2 text-sm text-gray-600')


    def run_photo_analysis(self, scenario, photo_info, dialog=None):
        """Run analysis on a photo and save results"""
        try:
            from analysis import analyze_vignetting, analyze_distortion, analyze_chromatic_aberration  # Added import

            if scenario['type'] == 'distortion':
                results = analyze_distortion(photo_info['path'])
                photo_info['analysis'] = {
                    **results,
                    'preview_path': photo_info['path'],
                    'analyzed_at': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'type': 'distortion'
                }
            elif scenario['type'] == 'vignette':
                results = analyze_vignetting(photo_info['path'])
                photo_info['analysis'] = {
                    'vignetting_results': results,
                    'visualization_path': results.get('visualization_path'),
                    'preview_path': photo_info['path'],
                    'analyzed_at': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'type': 'vignette'
                }
            elif scenario['type'] == 'chromatic':  # Add this section
                results = analyze_chromatic_aberration(photo_info['path'])
                photo_info['analysis'] = {
                    **results,
                    'preview_path': photo_info['path'],
                    'analyzed_at': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'type': 'chromatic'
                }

            # Save updated scenario to dataset
            if self.dataset_manager.update_scenario(
                    self.current_dataset['id'],
                    self.current_scenario
            ):
                ui.notification('Analysis complete', type='positive', position='bottom')
                if dialog:
                    dialog.close()
                # Refresh the UI
                self.select_scenario(scenario)
                # Show results
                self.show_photo_analysis(scenario, photo_info)
            else:
                ui.notification('Analysis failed to save', type='negative', position='bottom')

        except Exception as e:
            ui.notification(f'Error during analysis: {str(e)}', type='negative', position='bottom')
            logging.error(f"Analysis error: {e}")
            if dialog:
                dialog.close()
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
        with ui.card().classes('p-4 mt-4 bg-gray-50'):
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
                    ui.button('Cancel', on_click=dialog.close).classes('bg-gray-500 text-white')
                    ui.button(
                        'Delete',
                        on_click=lambda: self.perform_photo_delete(dialog, scenario, photo_info)
                    ).classes('bg-red-500 text-white')

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

            # Remove from scenario's photos list
            scenario['photos'] = [p for p in scenario['photos'] if p['filename'] != photo_info['filename']]

            # Update scenario in dataset
            if self.dataset_manager.update_scenario(
                    self.current_dataset['id'],
                    scenario
            ):
                ui.notify('Photo deleted successfully', type='positive')
                # Refresh the scenario view
                self.select_scenario(scenario)
            else:
                ui.notify('Failed to update scenario after delete', type='negative')

        except Exception as e:
            ui.notify(f'Error deleting photo: {str(e)}', type='negative')
            logging.error(f"Error in perform_photo_delete: {e}")

    def run(self):
        """Start the UI"""
        self.create_main_page()
        ui.run()
