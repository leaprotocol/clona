import logging
from ui import LensAnalysisUI
from camera_manager import CameraManager
from dataset_manager import DatasetManager

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    logging.debug("Starting Clona Lens Evaluation Application")
    
    try:
        # Initialize managers
        logging.debug("Initializing Camera Manager...")
        camera_manager = CameraManager()
        
        logging.debug("Initializing Dataset Manager...")
        dataset_manager = DatasetManager()

        # Create and run UI
        logging.debug("Starting UI...")
        ui = LensAnalysisUI(camera_manager, dataset_manager)
        ui.run()

    except Exception as e:
        logging.error(f"Application error: {e}")
        raise

if __name__ in {"__main__", "__mp_main__"}:
    main()