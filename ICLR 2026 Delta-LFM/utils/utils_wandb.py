import wandb
import torch
import numpy as np
from typing import List, Optional, Dict, Union



"""
    config = {
        "epochs": 3,
        "batch_size": 64,
        "lr": 0.001,
        "model": "simple-cnn"
    }
    
    # Initialize logger
    logger = WandBLogger(
        project="wandb-demo",
        run_name="cnn-mnist-v1",
        config=config,
        tags=["mnist", "demo"]
    )
"""

class WandBLogger:
    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        save_code: bool = True,
    ):
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            entity=entity,
            tags=tags,
            group=group,
            save_code=save_code,
        )
        self.config = self.run.config

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log scalar metrics to Weights & Biases.

        Args:
            metrics (dict): Metric name and value pairs.
            step (int): Optional step or epoch number.
        """
        wandb.log(metrics, step=step)

    def log_images(
        self,
        images: Union[
            torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]
        ],
        captions: Optional[List[str]] = None,
        label: str = "images",
        step: Optional[int] = None,
    ):
        """
        Log one or more images to Weights & Biases.

        Args:
            images: A single image or list of images (torch.Tensor or np.ndarray).
            captions: Optional list of captions for each image.
            label: Key name under which images are logged.
            step: Optional step or epoch number.
        """
        if not isinstance(images, list):
            images = [images]

        wandb_images = []

        for i, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()

            img = self._process_image_array(img)

            caption = captions[i] if captions and i < len(captions) else None
            wandb_images.append(wandb.Image(img, caption=caption))

        wandb.log({label: wandb_images}, step=step)

    # Standard image processing
    def _process_image_array(self, img: np.ndarray) -> np.ndarray:
        """
        Convert image to HWC uint8 format for logging.

        Args:
            img: A numpy array in HWC or CHW format, any dtype.

        Returns:
            np.ndarray in HWC format, dtype=uint8, RGB or grayscale.
        """
        img = img.astype(np.float32)

        # Convert CHW to HWC if needed
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        # Handle grayscale (H, W)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        # Scale if necessary
        if img.max() <= 1.0:
            img = img * 255.0

        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def log_artifact(self, file_path: str, artifact_name: str, artifact_type: str = "model"):
        """
        Log a file (e.g., model checkpoint) as an artifact.

        Args:
            file_path: Path to the file.
            artifact_name: Name to track it in wandb.
            artifact_type: Type of artifact, e.g., "model", "dataset".
        """
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)

    def finish(self):
        """
        Gracefully close the Weights & Biases run.
        """
        wandb.finish()
