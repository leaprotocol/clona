import logging
import asyncio
from nicegui import ui
from ui import LensAnalysisUI
from camera_manager import CameraManager
from dataset_manager import DatasetManager

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    logging.debug("Starting Clona Lens Analysis Application")
    
    try:
        # Initialize managers
        logging.debug("Initializing Camera Manager...")
        camera_manager = CameraManager()
        
        logging.debug("Initializing Dataset Manager...")
        dataset_manager = DatasetManager()

        # Create UI instance
        logging.debug("Starting UI...")
        ui_instance = LensAnalysisUI(camera_manager, dataset_manager)
        
        # Setup the UI within the NiceGUI context
        @ui.page('/')
        async def main_page():
            await ui_instance.setup_ui()
            
        ui_instance.run()

    except Exception as e:
        logging.error(f"Application error: {e}")
        raise

if __name__ in {"__main__", "__mp_main__"}:
    asyncio.run(main())