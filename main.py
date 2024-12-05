import logging
from camera_manager import CameraManager
from dataset_manager import DatasetManager
from ui import LensAnalysisUI


def setup_logging():
    """Set up application logging"""
    # Remove existing handlers first
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add basic console handler
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    ))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

def main():
    """Main application entry point"""
    # Initialize logging
    setup_logging()
    logging.info("Starting Lens Analysis Application")

    # Initialize managers
    camera_manager = CameraManager()
    dataset_manager = DatasetManager()

    # Initialize and run UI
    app_ui = LensAnalysisUI(camera_manager, dataset_manager)
    app_ui.run()

if __name__ in {"__main__", "__mp_main__"}:
    main()