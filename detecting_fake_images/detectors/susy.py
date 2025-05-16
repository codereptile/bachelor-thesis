import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from torchvision import transforms


class SuSy:
    """
    Wraps the SuSy TorchScript model and mirrors your original code:
    - loads the model on init
    - extracts top‐contrast patches
    - returns the average raw logit for the 'authentic' class
    """

    def __init__(self, top_k_patches=5, patch_size=224, plot_worst_patches: int = 0, device=None):
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = torch.jit.load("SuSy.pt", map_location=self.device)
        self.model.eval()
        self.top_k_patches = top_k_patches
        self.patch_size = patch_size
        self._to_gray = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Grayscale()
        ])
        self.classes = ['authentic', 'dalle-3-images', 'diffusiondb', 'midjourney-images', 'midjourney_tti', 'realisticSDXL']
        self.auth_idx = self.classes.index("authentic")
        self.plot_worst_patches = plot_worst_patches

    def is_real(self, image: Image.Image) -> float:
        """
        Returns the average *raw* logit score for the 'authentic' class
        over the top_k patches (no softmax applied).
        """

        # Get the image dimensions
        width, height = image.size

        # Calculate the number of patches
        num_patches_x = width // self.patch_size
        num_patches_y = height // self.patch_size

        # Divide the image in patches
        patches = np.zeros((num_patches_x * num_patches_y, self.patch_size, self.patch_size, 3), dtype=np.uint8)
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                x = i * self.patch_size
                y = j * self.patch_size
                patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
                patches[i * num_patches_y + j] = np.array(patch)

        # Compute the most relevant patches (optional)
        dissimilarity_scores = []
        for patch in patches:
            transform_patch = transforms.Compose([transforms.PILToTensor(), transforms.Grayscale()])
            grayscale_patch = transform_patch(Image.fromarray(patch)).squeeze(0)
            glcm = graycomatrix(grayscale_patch, [5], [0], 256, symmetric=True, normed=True)
            dissimilarity_scores.append(graycoprops(glcm, "contrast")[0, 0])

        # Sort patch indices by their dissimilarity score
        sorted_indices = np.argsort(dissimilarity_scores)[::-1]

        # Extract top k patches and convert them to tensor
        top_patches = patches[sorted_indices[:self.top_k_patches]]
        top_patches = torch.from_numpy(np.transpose(top_patches, (0, 3, 1, 2))) / 255.0
        top_patches = top_patches.to(self.device)

        # Predict patches
        with torch.no_grad():
            preds = self.model(top_patches).cpu()

        if self.plot_worst_patches:
            print(pd.DataFrame(preds.numpy(), columns=self.classes))

            auth_logits = preds[:, self.auth_idx].numpy()

            worst_indices = np.argsort(auth_logits)[:self.plot_worst_patches]

            n = self.plot_worst_patches
            ncols = int(np.ceil(np.sqrt(n)))
            nrows = int(np.ceil(n / ncols))

            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
            axes = np.array(axes).reshape(-1)

            for plot_idx, patch_idx in enumerate(worst_indices):
                patch_img = top_patches[patch_idx].cpu().numpy().transpose(1, 2, 0)
                patch_img = np.clip(patch_img * 255, 0, 255).astype(np.uint8)
                axes[plot_idx].imshow(patch_img)
                axes[plot_idx].set_title(f"Worst {plot_idx + 1}\nLogit: {auth_logits[patch_idx]:.2f}")
                axes[plot_idx].axis('off')

            for i in range(n, nrows * ncols):
                axes[i].axis('off')

            plt.tight_layout()
            plt.show()

        return preds[:, self.auth_idx].mean().item()

    def score_to_bool(self, score: float, threshold: float = 0.7) -> bool:
        return score > threshold