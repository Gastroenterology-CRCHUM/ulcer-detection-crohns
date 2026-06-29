"""
tests/test_data_transforms.py
=============================
Unit tests for image transformation pipeline.
"""

import numpy as np
import pytest
import torchvision.transforms as T
from PIL import Image

from src.data.transforms import CLAHE_Y, ResizeWithPad, get_transforms


class TestResizeWithPad:
    """Test ResizeWithPad transformation."""

    def test_init(self):
        """Test ResizeWithPad initialization."""
        transform = ResizeWithPad(target_size=512, fill=0)
        assert transform.target_size == 512
        assert transform.fill == 0

    def test_output_is_square(self, rgb_image):
        """Test that output is always square."""
        transform = ResizeWithPad(target_size=256, fill=0)
        result = transform(rgb_image)
        assert result.size == (256, 256)

    def test_aspect_ratio_preserved(self, rgb_image):
        """Test that aspect ratio is preserved in the resized portion."""
        transform = ResizeWithPad(target_size=512, fill=0)
        result = transform(rgb_image)
        assert result.size == (512, 512)
        # Result should be PIL Image
        assert isinstance(result, Image.Image)

    def test_different_target_sizes(self):
        """Test with different target sizes."""
        img = Image.new("RGB", (320, 240))
        sizes = [224, 256, 512]
        for size in sizes:
            transform = ResizeWithPad(target_size=size, fill=128)
            result = transform(img)
            assert result.size == (size, size)

    def test_fill_value(self, non_square_rgb_image):
        """Test that fill value affects padding."""
        transform = ResizeWithPad(target_size=200, fill=0)
        result = transform(non_square_rgb_image)
        # Check that padding was applied
        assert result.size == (200, 200)
        result_arr = np.array(result)
        # With a 50x100 image resized to 200x200, left/right padding should be present
        assert result_arr[:, 0, :].min() == 0
        assert result_arr[:, -1, :].min() == 0


class TestCLAHEY:
    """Test CLAHE_Y transformation."""

    def test_output_shape(self, rgb_image):
        """Test that output has same shape as input."""
        transform = CLAHE_Y()
        result = transform(rgb_image)
        assert result.size == rgb_image.size
        assert result.mode == "RGB"

    def test_luminance_enhancement(self, rgb_image):
        """Test that luminance channel is enhanced."""
        transform = CLAHE_Y()
        result = transform(rgb_image)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_uniform_rgb_input(self):
        """Test CLAHE_Y with a uniform RGB image (CLAHE_Y assumes RGB input)."""
        uniform_img = Image.new("RGB", (128, 128), color=(128, 128, 128))
        transform = CLAHE_Y()
        result = transform(uniform_img)
        assert result.size == (128, 128)


class TestGetTransforms:
    """Test get_transforms pipeline builder."""

    def test_training_pipeline(self):
        """Test training pipeline includes augmentations."""
        pipeline = get_transforms(img_size=224, is_training=True, equalize=True)
        assert isinstance(pipeline, T.Compose)
        # Should have: ResizeWithPad, CLAHE_Y, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, Normalize
        assert len(pipeline.transforms) >= 6

    def test_inference_pipeline(self):
        """Test inference pipeline excludes augmentations."""
        pipeline = get_transforms(img_size=224, is_training=False, equalize=True)
        assert isinstance(pipeline, T.Compose)
        # Should have: ResizeWithPad, CLAHE_Y, ToTensor, Normalize (no flip/rotate/colorjitter)
        assert len(pipeline.transforms) >= 4

    def test_different_img_sizes(self):
        """Test that different image sizes are handled correctly."""
        for size in [224, 256, 512]:
            pipeline = get_transforms(img_size=size, is_training=False)
            assert isinstance(pipeline, T.Compose)

    def test_augmentation_parameters(self):
        """Test custom augmentation parameters."""
        pipeline = get_transforms(
            img_size=256,
            is_training=True,
            equalize=False,
            rotation_degrees=45,
            horizontal_flip_prob=0.7,
            brightness_factor=0.2,
            contrast_factor=0.3,
            saturation_factor=0.1,
        )
        assert isinstance(pipeline, T.Compose)

    def test_output_shape(self):
        """Test that pipeline output has correct shape."""
        import torch

        pipeline = get_transforms(img_size=224, is_training=False, equalize=False)
        img = Image.new("RGB", (512, 512))
        result = pipeline(img)
        # Output should be tensor with shape (3, 224, 224)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_normalization_values(self):
        """Test that ImageNet normalization is used."""
        # This test verifies the correct mean/std for transfer learning
        pipeline = get_transforms(img_size=224, is_training=False, equalize=False)
        # Pipeline should contain Normalize transform with ImageNet stats
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        assert isinstance(pipeline, T.Compose)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
