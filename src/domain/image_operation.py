from abc import ABC, abstractmethod
import numpy as np

class ImageOperation(ABC):
    """
    Abstract Base Class for all image processing operations.
    Follows the Strategy Pattern.
    """
    @abstractmethod
    def execute(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Executes the image processing operation.
        
        Args:
            image (np.ndarray): The input image (BGR or Grayscale).
            **kwargs: Additional parameters specific to the operation.
            
        Returns:
            np.ndarray: The processed image.
        """
        pass
