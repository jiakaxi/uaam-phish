"""
Test scenario label functionality in multimodal datamodule.
"""

from __future__ import annotations


import pandas as pd
import pytest
from transformers import BertTokenizer
from torchvision import transforms

from src.data.multimodal_datamodule import MultimodalDataset, multimodal_collate_fn


@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    return pd.DataFrame(
        {
            "id": ["sample_1", "sample_2", "sample_3", "sample_4"],
            "label": [1, 0, 1, 0],
            "url_text": [
                "http://phish.com",
                "http://legit.com",
                "http://test.com",
                "http://ood.com",
            ],
            "html_text": [
                "<html>phish</html>",
                "<html>legit</html>",
                "<html>test</html>",
                "<html>ood</html>",
            ],
            "img_path": ["img1.jpg", "img2.jpg", "corrupt/light/img3.jpg", "img4.jpg"],
            "brand": ["PayPal", "Amazon", "PayPal", "NewBrand"],
            "corruption_level": [None, None, "light", None],
        }
    )


@pytest.fixture
def tokenizer():
    """Create BERT tokenizer."""
    return BertTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def transform():
    """Create image transform."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def test_scenario_clean_iid(sample_df, tokenizer, transform, tmp_path):
    """Test that clean IID samples get 'clean' scenario."""
    dataset = MultimodalDataset(
        df=sample_df,
        url_max_len=200,
        url_vocab_size=128,
        html_tokenizer=tokenizer,
        html_max_len=256,
        visual_transform=transform,
        image_dir=tmp_path,
        protocol="iid",
        scenario=None,
    )

    # Sample 0 should be clean
    sample = dataset[0]
    assert "meta" in sample, "Sample should contain 'meta' field"
    assert sample["meta"]["scenario"] == "clean"
    assert sample["meta"]["corruption_level"] == "clean"
    assert sample["meta"]["protocol"] == "iid"


def test_scenario_corruption_light(sample_df, tokenizer, transform, tmp_path):
    """Test that light corruption samples get 'light' scenario."""
    dataset = MultimodalDataset(
        df=sample_df,
        url_max_len=200,
        url_vocab_size=128,
        html_tokenizer=tokenizer,
        html_max_len=256,
        visual_transform=transform,
        image_dir=tmp_path,
        protocol="iid",
        scenario=None,
    )

    # Sample 2 has explicit corruption_level='light'
    sample = dataset[2]
    assert sample["meta"]["scenario"] == "light"
    assert sample["meta"]["corruption_level"] == "light"


def test_scenario_brandood(sample_df, tokenizer, transform, tmp_path):
    """Test that brand-OOD protocol gets 'brandood' scenario."""
    dataset = MultimodalDataset(
        df=sample_df,
        url_max_len=200,
        url_vocab_size=128,
        html_tokenizer=tokenizer,
        html_max_len=256,
        visual_transform=transform,
        image_dir=tmp_path,
        protocol="brandood",
        scenario=None,
    )

    # All samples in brandood protocol should have 'brandood' scenario
    sample = dataset[3]
    assert sample["meta"]["scenario"] == "brandood"
    assert sample["meta"]["corruption_level"] == "clean"
    assert sample["meta"]["protocol"] == "brandood"


def test_scenario_override(sample_df, tokenizer, transform, tmp_path):
    """Test explicit scenario override."""
    dataset = MultimodalDataset(
        df=sample_df,
        url_max_len=200,
        url_vocab_size=128,
        html_tokenizer=tokenizer,
        html_max_len=256,
        visual_transform=transform,
        image_dir=tmp_path,
        protocol="iid",
        scenario="heavy",  # Explicit override
    )

    # All samples should use the override
    sample = dataset[0]
    assert sample["meta"]["scenario"] == "heavy"
    assert sample["meta"]["corruption_level"] == "heavy"


def test_collate_fn_meta(sample_df, tokenizer, transform, tmp_path):
    """Test that collate function properly handles meta fields."""
    dataset = MultimodalDataset(
        df=sample_df,
        url_max_len=200,
        url_vocab_size=128,
        html_tokenizer=tokenizer,
        html_max_len=256,
        visual_transform=transform,
        image_dir=tmp_path,
        protocol="iid",
        scenario=None,
    )

    # Create a batch
    batch = [dataset[i] for i in range(2)]
    collated = multimodal_collate_fn(batch)

    # Check that meta is properly collated
    assert "meta" in collated
    assert "scenario" in collated["meta"]
    assert "corruption_level" in collated["meta"]
    assert "protocol" in collated["meta"]

    # Meta fields should be lists of strings
    assert isinstance(collated["meta"]["scenario"], list)
    assert isinstance(collated["meta"]["corruption_level"], list)
    assert isinstance(collated["meta"]["protocol"], list)
    assert len(collated["meta"]["scenario"]) == 2

    # Check values
    assert collated["meta"]["scenario"][0] == "clean"
    assert collated["meta"]["protocol"][0] == "iid"


def test_scenario_inference_from_path(tokenizer, transform, tmp_path):
    """Test scenario inference from image paths."""
    # Create dataframe with corruption in paths
    df = pd.DataFrame(
        {
            "id": ["s1", "s2", "s3", "s4"],
            "label": [1, 1, 1, 1],
            "url_text": ["http://test.com"] * 4,
            "html_text": ["<html></html>"] * 4,
            "img_path": [
                "clean/img1.jpg",
                "corrupt/light/img2.jpg",
                "corrupt/medium/img3.jpg",
                "corrupt/heavy/img4.jpg",
            ],
        }
    )

    dataset = MultimodalDataset(
        df=df,
        url_max_len=200,
        url_vocab_size=128,
        html_tokenizer=tokenizer,
        html_max_len=256,
        visual_transform=transform,
        image_dir=tmp_path,
        protocol="iid",
        scenario=None,
    )

    # Test clean
    assert dataset[0]["meta"]["scenario"] == "clean"

    # Test light corruption (inferred from path)
    assert dataset[1]["meta"]["scenario"] == "light"

    # Test medium corruption (inferred from path)
    assert dataset[2]["meta"]["scenario"] == "medium"

    # Test heavy corruption (inferred from path)
    assert dataset[3]["meta"]["scenario"] == "heavy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
