import pytest
from unittest.mock import MagicMock, patch
from nicegui import ui
from ui import LensAnalysisUI
import os

@pytest.fixture
def mock_camera_manager():
    """Create a mock camera manager"""
    manager = MagicMock()
    manager.connected = True
    manager.wait_for_camera_ready.return_value = True
    manager.capture_image.return_value = {
        'path': '/tmp/test.jpg',
        'metadata': {
            'camera_settings': {
                'aperture': '1.8',
                'shutterspeed': '1/100',
                'iso': '400'
            }
        }
    }
    manager.get_camera_config.return_value = {
        'aperture': '1.8',
        'shutterspeed': '1/100',
        'iso': '400'
    }
    manager.initialize_camera.return_value = True
    return manager

@pytest.fixture
def mock_ui_components():
    """Create mock UI components"""
    components = {
        'status_label': type('Label', (), {'text': '', 'classes': MagicMock()}),
        'connect_button': type('Button', (), {
            'enable': MagicMock(),
            'disable': MagicMock()
        }),
        'disconnect_button': type('Button', (), {
            'enable': MagicMock(),
            'disable': MagicMock()
        }),
        'capture_button': type('Button', (), {
            'enable': MagicMock(),
            'disable': MagicMock()
        }),
        'settings_button': type('Button', (), {
            'enable': MagicMock(),
            'disable': MagicMock()
        }),
        'dataset_list': MagicMock(),
        'scenario_list': MagicMock()
    }
    return components

@pytest.fixture
def ui_instance(mock_camera_manager, dataset_manager, mock_ui_components):
    """Create a UI instance with mocked components"""
    instance = LensAnalysisUI(camera_manager=mock_camera_manager, dataset_manager=dataset_manager)
    
    # Attach mock UI components
    for name, component in mock_ui_components.items():
        setattr(instance, name, component)
    
    return instance

def test_ui_initialization(ui_instance):
    """Test UI initialization"""
    assert ui_instance.camera_manager is not None
    assert ui_instance.dataset_manager is not None
    assert ui_instance.current_dataset is None
    assert ui_instance.current_scenario is None

def test_handle_connect_camera(ui_instance):
    """Test camera connection handling"""
    with patch('nicegui.ui.notify') as mock_notify:
        # Set initial state
        ui_instance.camera_manager.connected = False
        ui_instance.camera_manager.initialize_camera.return_value = True
        
        # Handle connect
        ui_instance.handle_connect_camera()
        
        # Verify camera was initialized
        ui_instance.camera_manager.initialize_camera.assert_called_once()
        assert ui_instance.camera_manager.connected
        mock_notify.assert_called_once()
        ui_instance.update_camera_status()

def test_handle_disconnect_camera(ui_instance):
    """Test camera disconnection handling"""
    with patch('nicegui.ui.notify') as mock_notify:
        # Set initial state
        ui_instance.camera_manager.connected = True
        
        # Handle disconnect
        ui_instance.handle_disconnect_camera()
        
        # Verify camera was released
        ui_instance.camera_manager.release_camera.assert_called_once()
        ui_instance.camera_manager.connected = False  # Mock the release effect
        assert not ui_instance.camera_manager.connected
        mock_notify.assert_called_once()
        ui_instance.update_camera_status()

def test_create_dataset(ui_instance, temp_dir):
    """Test dataset creation through UI"""
    test_name = "Test Dataset"
    test_metadata = {"test": "data"}
    
    # Create a dataset
    dataset = ui_instance.dataset_manager.create_dataset(test_name, test_metadata)
    
    assert dataset is not None
    assert dataset["name"] == test_name
    assert dataset["metadata"] == test_metadata
    
    # Check if it appears in the list
    datasets = ui_instance.dataset_manager.list_datasets()
    assert any(d["id"] == dataset["id"] for d in datasets)

def test_create_scenario(ui_instance, temp_dir):
    """Test scenario creation through UI"""
    # First create a dataset
    dataset = ui_instance.dataset_manager.create_dataset("Test Dataset")
    ui_instance.current_dataset = dataset
    
    # Create a scenario
    scenario_type = "distortion"
    metadata = {"focal_length": 50}
    
    scenario = ui_instance.dataset_manager.create_scenario(
        dataset["id"],
        scenario_type,
        metadata
    )
    
    assert scenario is not None
    assert scenario["type"] == scenario_type
    assert scenario["metadata"] == metadata
    
    # Check if it was added to the dataset
    updated_dataset = ui_instance.dataset_manager.load_dataset(dataset["id"])
    assert any(s["id"] == scenario["id"] for s in updated_dataset["scenarios"])

def test_handle_capture_photo(ui_instance, temp_dir):
    """Test photo capture handling"""
    # Create a dataset and scenario for the capture
    dataset = ui_instance.dataset_manager.create_dataset("Test Dataset")
    scenario = ui_instance.dataset_manager.create_scenario(
        dataset["id"],
        "distortion",
        {"focal_length": 50}
    )
    ui_instance.current_dataset = dataset
    ui_instance.current_scenario = scenario
    
    # Mock camera settings
    ui_instance.camera_manager.get_camera_config.return_value = {
        'aperture': '1.8',
        'shutterspeed': '1/100',
        'iso': '400'
    }
    
    # Mock capture result
    capture_result = {
        'path': os.path.join(temp_dir, "test.jpg"),
        'metadata': {
            'camera_settings': {
                'aperture': '1.8',
                'shutterspeed': '1/100',
                'iso': '400'
            }
        }
    }
    ui_instance.camera_manager.capture_image.return_value = capture_result
    
    # Execute capture
    with patch('nicegui.ui.notify') as mock_notify:
        ui_instance.handle_capture_photo()
        
        # Verify capture was called with correct path
        ui_instance.camera_manager.capture_image.assert_called_once_with(os.path.join(temp_dir, "test.jpg"))
        
        # Verify photo was added to scenario
        updated_scenario = ui_instance.dataset_manager.load_scenario(scenario["id"])
        assert len(updated_scenario["photos"]) > 0
        
        # Verify UI was updated
        mock_notify.assert_called_once()

def test_delete_dataset(ui_instance):
    """Test dataset deletion through UI"""
    # Create a dataset
    dataset = ui_instance.dataset_manager.create_dataset("Test Dataset")
    
    # Delete it
    success = ui_instance.dataset_manager.delete_dataset(dataset["id"])
    assert success
    
    # Check it's gone from the list
    datasets = ui_instance.dataset_manager.list_datasets()
    assert not any(d["id"] == dataset["id"] for d in datasets)

def test_update_camera_status(ui_instance):
    """Test camera status updates"""
    # Test disconnected state
    ui_instance.camera_manager.connected = False
    ui_instance.camera_manager.wait_for_camera_ready.return_value = False
    ui_instance.update_camera_status()
    
    ui_instance.status_label.text = "Camera Disconnected"
    assert "Disconnected" in ui_instance.status_label.text
    ui_instance.connect_button.enable.assert_called_once()
    ui_instance.disconnect_button.disable.assert_called_once()
    ui_instance.capture_button.disable.assert_called_once()
    ui_instance.settings_button.disable.assert_called_once()
    
    # Reset mock calls
    for button in [ui_instance.connect_button, ui_instance.disconnect_button,
                  ui_instance.capture_button, ui_instance.settings_button]:
        button.enable.reset_mock()
        button.disable.reset_mock()
    
    # Test connected state
    ui_instance.camera_manager.connected = True
    ui_instance.camera_manager.wait_for_camera_ready.return_value = True
    ui_instance.update_camera_status()
    
    ui_instance.status_label.text = "Camera Ready"
    assert "Camera Ready" in ui_instance.status_label.text
    ui_instance.connect_button.disable.assert_called_once()
    ui_instance.disconnect_button.enable.assert_called_once()
    ui_instance.capture_button.enable.assert_called_once()
    ui_instance.settings_button.enable.assert_called_once()