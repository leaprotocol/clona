import pytest
import os
import json
from dataset_manager import DatasetManager

def test_create_dataset(dataset_manager):
    """Test creating a new dataset"""
    name = "Test Dataset"
    metadata = {"lens": "50mm f/1.8"}
    
    result = dataset_manager.create_dataset(name, metadata)
    
    assert result is not None
    assert result["name"] == name
    assert result["metadata"] == metadata
    assert "id" in result
    assert "created" in result
    
    # Check if files were created
    dataset_path = os.path.join(dataset_manager.base_path, result["id"])
    assert os.path.exists(dataset_path)
    assert os.path.exists(os.path.join(dataset_path, "scenarios"))
    assert os.path.exists(os.path.join(dataset_path, "images"))
    
    # Check info.json
    with open(os.path.join(dataset_path, "info.json"), 'r') as f:
        info = json.load(f)
        assert info == result

@pytest.mark.asyncio
async def test_list_datasets(dataset_manager):
    """Test listing datasets"""
    # Create some test datasets
    dataset1 = dataset_manager.create_dataset("Dataset 1")
    dataset2 = dataset_manager.create_dataset("Dataset 2")
    
    datasets = await dataset_manager.list_datasets()
    
    assert len(datasets) == 2
    assert any(d["id"] == dataset1["id"] for d in datasets)
    assert any(d["id"] == dataset2["id"] for d in datasets)

def test_delete_dataset(dataset_manager):
    """Test deleting a dataset"""
    dataset = dataset_manager.create_dataset("Test Dataset")
    dataset_path = os.path.join(dataset_manager.base_path, dataset["id"])
    
    assert os.path.exists(dataset_path)
    
    success = dataset_manager.delete_dataset(dataset["id"])
    assert success
    assert not os.path.exists(dataset_path)
    
    # Check it's removed from list
    datasets = dataset_manager.list_datasets()
    assert not any(d["id"] == dataset["id"] for d in datasets)

def test_create_scenario(dataset_manager):
    """Test creating a scenario in a dataset"""
    dataset = dataset_manager.create_dataset("Test Dataset")
    scenario_type = "distortion"
    metadata = {"focal_length": 50}
    
    scenario = dataset_manager.create_scenario(dataset["id"], scenario_type, metadata)
    
    assert scenario is not None
    assert scenario["type"] == scenario_type
    assert scenario["metadata"] == metadata
    assert "id" in scenario
    assert "created" in scenario
    
    # Check scenario was added to dataset
    updated_dataset = dataset_manager.load_dataset(dataset["id"])
    assert len(updated_dataset["scenarios"]) == 1
    assert updated_dataset["scenarios"][0]["id"] == scenario["id"]

def test_update_scenario(dataset_manager):
    """Test updating a scenario"""
    dataset = dataset_manager.create_dataset("Test Dataset")
    scenario = dataset_manager.create_scenario(dataset["id"], "distortion", {})
    
    # Modify scenario
    scenario["metadata"] = {"new": "data"}
    success = dataset_manager.update_scenario(dataset["id"], scenario)
    
    assert success
    
    # Check changes were saved
    updated_dataset = dataset_manager.load_dataset(dataset["id"])
    updated_scenario = updated_dataset["scenarios"][0]
    assert updated_scenario["metadata"] == {"new": "data"}

def test_export_import_dataset(dataset_manager, temp_dir):
    """Test exporting and importing a dataset"""
    # Create a dataset with some content
    original_dataset = dataset_manager.create_dataset("Export Test")
    
    # Add a scenario
    scenario = dataset_manager.create_scenario(original_dataset["id"], "distortion", {"test": "data"})
    original_dataset["scenarios"] = [scenario]
    
    # Export it
    export_path = os.path.join(temp_dir, "exports")
    os.makedirs(export_path, exist_ok=True)
    success = dataset_manager.export_dataset(original_dataset["id"], export_path)
    assert success
    
    # Find the exported zip file
    zip_files = [f for f in os.listdir(export_path) if f.endswith('.zip')]
    assert len(zip_files) == 1
    zip_path = os.path.join(export_path, zip_files[0])
    
    # Import it back
    imported_dataset = dataset_manager.import_dataset(zip_path)
    assert imported_dataset is not None
    assert imported_dataset["name"] == original_dataset["name"]
    assert len(imported_dataset["scenarios"]) == len(original_dataset["scenarios"])
    
    # Verify scenario data
    imported_scenario = imported_dataset["scenarios"][0]
    original_scenario = original_dataset["scenarios"][0]
    assert imported_scenario["id"].startswith("distortion_")
    assert imported_scenario["metadata"] == original_scenario["metadata"]
    
    # Check scenario data was preserved
    imported_scenario = imported_dataset["scenarios"][0]
    assert imported_scenario["type"] == original_scenario["type"]
    assert imported_scenario["metadata"] == original_scenario["metadata"] 