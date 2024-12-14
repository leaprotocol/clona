from nicegui import ui
import os
from datetime import datetime
import shutil
import exifread
import logging
from nicegui.events import UploadEventArguments
from analysis import convert_raw_to_jpeg, analyze_chromatic_aberration, analyze_sharpness




class DatasetControls:
    def __init__(self, dataset_manager, on_dataset_change=None):
        self.dataset_manager = dataset_manager
        self.on_dataset_change = on_dataset_change
        self.dataset_list = None
        self.current_dataset = None
        self.current_scenario = None
        self.scenario_details = None
        

    async def setup_controls(self):
        """Setup the dataset control UI elements"""
        with ui.column().classes('w-full gap-4'):
            # Dataset controls
            with ui.card().classes('w-full p-4'):
                ui.label('Datasets').classes('text-xl mb-4')
                self.dataset_list = ui.column().classes('w-full gap-2')
                await self.refresh_dataset_list()
            
            # Scenario controls
            with ui.card().classes('w-full p-4'):
                with ui.row().classes('justify-between items-center mb-4'):
                    ui.label('Scenarios').classes('text-xl')
                    ui.button(
                        'New Scenario',
                        on_click=self.create_scenario_dialog
                    ).classes('bg-blue-500 text-white')
                self.scenario_list = ui.column().classes('w-full gap-2')
                await self.refresh_scenario_list()

    async def select_dataset(self, dataset):
        """Handle dataset selection"""
        try:
            logging.debug(f"Dataset selected: {dataset}")
            self.current_dataset = dataset
            self.current_scenario = None
            await self.refresh_scenario_list()
            if self.on_dataset_change:
                await self.on_dataset_change(dataset)
        except Exception as e:
            logging.error(f"Error in select_dataset: {str(e)}")
            ui.notify('Error selecting dataset', type='negative')

    async def delete_dataset(self, dataset):
        """Handle dataset deletion"""
        try:
            if await self.dataset_manager.delete_dataset(dataset['id']):
                ui.notify('Dataset deleted successfully', type='positive')
                await self.refresh_dataset_list()
            else:
                ui.notify('Failed to delete dataset', type='negative')
        except Exception as e:
            ui.notify(f'Error deleting dataset: {str(e)}', type='negative')

    async def refresh_dataset_list(self):
        """Refresh the dataset selection dropdown"""
        if not self.dataset_list:
            return False

        self.dataset_list.clear()
        try:
            datasets = await self.dataset_manager.list_datasets()
            
            if not datasets:
                with self.dataset_list:
                    ui.label('No datasets available').classes('text-gray-500 italic')
                return True
            
            with self.dataset_list:
                with ui.row().classes('w-full justify-between items-center'):
                    select = ui.select(
                        options=[{'text': d['name'], 'value': d} for d in datasets],
                        value=self.current_dataset,
                        on_change=lambda e: self.select_dataset(e.value)
                    ).classes('w-full')
                    select.props('filled with-input')
                    
                    ui.button(
                        'Delete',
                        on_click=lambda: self.delete_dataset(self.current_dataset) if self.current_dataset else None
                    ).classes('bg-red-500 text-white p-1 text-sm').props('disabled={!self.current_dataset}')

            return True
        except Exception as e:
            with self.dataset_list:
                ui.label(f'Error loading datasets: {str(e)}').classes('text-red-500')
            return False

    async def select_scenario(self, scenario):
        """Select a scenario and update UI"""
        try:
            self.current_scenario = scenario
            if self.scenario_details:
                self.scenario_details.clear()
                with self.scenario_details:
                    await self.refresh_scenario_details()
            return True
        except Exception as e:
            logging.error(f"Error selecting scenario: {e}")
            ui.notify('Error selecting scenario', type='negative')
            return False

    async def refresh_scenario_details(self):
        """Refresh the scenario details display"""
        if not self.current_scenario:
            return False

        try:
            for photo in self.current_scenario.get('photos', []):
                with ui.card().classes('w-full mb-4'):
                    with ui.column().classes('w-full'):
                        preview_path = photo.get('preview_path', '')
                        if preview_path:
                            if preview_path.lower().endswith(('.cr2', '.nef', '.arw')):
                                jpg_path = preview_path.rsplit('.', 1)[0] + '_preview.jpg'
                                if not os.path.exists(jpg_path):
                                    convert_raw_to_jpeg(preview_path, jpg_path)
                                preview_path = jpg_path
                            ui.image(preview_path).classes('max-w-full')

                        with ui.row().classes('gap-2 mt-2'):
                            if self.current_scenario['type'] == 'distortion':
                                ui.button(
                                    'Analyze Distortion',
                                    on_click=lambda p=photo: self.run_photo_analysis(self.current_scenario, p)
                                ).classes('bg-blue-500 text-white p-2')
                            elif self.current_scenario['type'] == 'vignette':
                                ui.button(
                                    'Analyze Vignette',
                                    on_click=lambda p=photo: self.run_photo_analysis(self.current_scenario, p)
                                ).classes('bg-blue-500 text-white p-2')

            return True
        except Exception as e:
            logging.error(f"Error refreshing scenario details: {e}")
            ui.notify('Error refreshing scenario details', type='negative')
            return False

    async def run_photo_analysis(self, scenario, photo_info, dialog=None):
        """Run analysis on a photo based on scenario type"""
        try:
            loading_notification = ui.notify('Running analysis...', type='ongoing')
            metadata = photo_info.get('metadata', {})

            if scenario['type'] == 'chromatic':
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

            if await self.dataset_manager.update_scenario(
                self.current_dataset['id'],
                scenario
            ):
                loading_notification.delete()
                if dialog:
                    dialog.close()
                await self.show_photo_analysis(scenario, photo_info)
                ui.notify('Analysis complete', type='positive', position='bottom')
                await self.select_scenario(scenario)
            else:
                loading_notification.delete()
                ui.notify('Failed to save analysis results', type='negative')

        except Exception as e:
            ui.notify(f'Error running analysis: {str(e)}', type='negative')
            logging.error(f"Error running analysis: {e}")

    async def handle_import_raw(self):
        """Handle RAW file import"""
        if not self.current_scenario:
            ui.notify('Please select a scenario first', type='warning')
            return

        async def handle_upload(e: UploadEventArguments):
            try:
                if not e.name.lower().endswith(('.cr2', '.nef', '.arw')):
                    ui.notify('Please select a RAW file (.CR2, .NEF, or .ARW)', type='warning')
                    return

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_filename = f"{self.current_scenario['type']}_{timestamp}_{e.name}"
                dataset_path = os.path.join("datasets", self.current_dataset['id'], "photos")
                final_path = os.path.join(dataset_path, final_filename)

                # Save uploaded file
                os.makedirs(dataset_path, exist_ok=True)
                with open(final_path, 'wb') as f:
                    e.content.seek(0)
                    shutil.copyfileobj(e.content, f)

                # Generate preview
                preview_filename = f"{os.path.splitext(final_filename)[0]}_preview.jpg"
                preview_path = os.path.join(dataset_path, preview_filename)
                convert_raw_to_jpeg(final_path, preview_path)

                # Extract metadata
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

                if await self.dataset_manager.update_scenario(
                    self.current_dataset['id'],
                    self.current_scenario
                ):
                    ui.notify('RAW file imported successfully', type='positive')
                    await self.select_scenario(self.current_scenario)
                else:
                    ui.notify('File imported but failed to update dataset', type='warning')

            except Exception as e:
                ui.notify(f'Error importing RAW file: {str(e)}', type='negative')
                logging.error(f"Error importing RAW file: {e}")

        ui.upload(
            label='Import RAW file',
            on_upload=handle_upload
        ).classes('max-w-full')

    async def handle_dataset_selection(self, dataset_id):
        """Handle dataset selection and update UI accordingly"""
        try:
            dataset = await self.dataset_manager.load_dataset(dataset_id)
            if dataset is None:
                ui.notify('Failed to load dataset', type='negative')
                return False

            self.current_dataset = dataset
            await self.refresh_dataset_list()
            ui.notify(f'Dataset {dataset["name"]} selected', type='positive')
            return True
        except Exception as e:
            logging.error(f"Error in handle_dataset_selection: {e}")
            ui.notify(f'Error selecting dataset: {str(e)}', type='negative')
            return False

    async def handle_capture_photo(self):
        """Handle photo capture"""
        try:
            if not self.current_scenario:
                ui.notify('Please select a scenario first', type='warning')
                return

            if self.current_scenario['type'] in ['vignette', 'bokeh', 'distortion', 'sharpness', 'chromatic']:
                await self.show_aperture_selection_dialog()
            else:
                dataset_path = os.path.join("datasets", self.current_dataset['id'], "photos")
                aperture = self.current_scenario.get('aperture', 'default_aperture')
                await self.photo_capture.do_capture(dataset_path, aperture)

        except Exception as e:
            logging.error(f"Error in handle_capture_photo: {e}")
            ui.notify('Error capturing photo', type='negative')

    def create_scenario_dialog(self):
        """Show dialog for creating a new scenario"""
        dialog = ui.dialog()
        with dialog, ui.card():
            ui.label('Create New Scenario').classes('text-xl mb-4')
            
            scenario_type = ui.select(
                label='Scenario Type',
                options=['distortion', 'vignette', 'bokeh', 'chromatic', 'sharpness']
            ).classes('w-full mb-4')
            
            focal_length = ui.number(
                label='Focal Length (mm)',
                min=1,
                max=1000
            ).classes('w-full mb-4')
            
            notes = ui.textarea(
                label='Notes',
                placeholder='Enter any additional notes'
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

    async def setup_scenario_list(self):
        """Setup the scenario list UI elements"""
        self.scenario_list = ui.column().classes('w-full gap-2')
        await self.refresh_scenario_list()

    async def refresh_scenario_list(self):
        """Refresh the scenario list display"""
        if not self.scenario_list or not self.current_dataset:
            return False

        self.scenario_list.clear()
        try:
            scenarios = self.current_dataset.get('scenarios', [])
            
            if not scenarios:
                with self.scenario_list:
                    ui.label('No scenarios available').classes('text-gray-500 italic')
                return True
            
            with self.scenario_list:
                for scenario in scenarios:
                    with ui.card().classes('w-full p-2 mb-2'):
                        with ui.row().classes('justify-between items-center'):
                            ui.label(f"{scenario['type']} - {scenario['metadata'].get('focal_length', 'N/A')}mm").classes(
                                'font-bold text-blue-500' 
                                if scenario == self.current_scenario 
                                else ''
                            )
                            with ui.row().classes('gap-1'):
                                ui.button(
                                    'Select',
                                    on_click=lambda s=scenario: self.select_scenario(s)
                                ).classes('bg-blue-500 text-white p-1 text-sm')
                                ui.button(
                                    'Delete',
                                    on_click=lambda s=scenario: self.delete_scenario(s)
                                ).classes('bg-red-500 text-white p-1 text-sm')
            return True
        except Exception as e:
            with self.scenario_list:
                ui.label(f'Error loading scenarios: {str(e)}').classes('text-red-500')
            return False

    async def delete_scenario(self, scenario):
        """Handle scenario deletion"""
        try:
            if await self.dataset_manager.delete_scenario(self.current_dataset['id'], scenario['id']):
                ui.notify('Scenario deleted successfully', type='positive')
                await self.refresh_scenario_list()
            else:
                ui.notify('Failed to delete scenario', type='negative')
        except Exception as e:
            ui.notify(f'Error deleting scenario: {str(e)}', type='negative')
            logging.error(f"Error in delete_scenario: {e}")