from typing import Any
import torch
from tqdm import tqdm
from cupbearer.detectors.anomaly_detector import AnomalyDetector
from collections import defaultdict
import umap
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import pdb

class FeatureVisualizer(AnomalyDetector):
    def _compute_layerwise_scores(self, inputs: Any, features: Any) -> dict[str, torch.Tensor]:
        pass

    def _train(self, data: Any) -> None:
        pass

    def train(
        self,
        trusted_data=None,
        untrusted_data=None,
        test_data=None,
        *,
        batch_size: int = 32,
        max_steps: int | None = None,
        shuffle: bool = False,
        **kwargs,
    ):
        dataloaders = {}
        for k, data in {"trusted": trusted_data, "untrusted": untrusted_data, "test": test_data}.items():
            if data is None:
                dataloaders[k] = None
            else:
                dataloaders[k] = torch.utils.data.DataLoader(
                    data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                )

        self.features = defaultdict(list)
        self.class_labels = []
        
        with torch.no_grad():
            if 'trusted' in dataloaders:
                for i, batch in tqdm(enumerate(dataloaders['trusted']), desc=f"Collecting features on {k} data"):
                    inputs, new_labels = batch
                    features = self.feature_extractor(inputs)

                    for name, feature in features.items():
                        self.features[name].append(feature.cpu())
                    labels = torch.zeros_like(new_labels)
                    self.class_labels.append(labels)

            if 'test' in dataloaders:
                for i, batch in tqdm(enumerate(dataloaders['test']), desc=f"Collecting features on {k} data"):
                    if max_steps is not None and i >= max_steps:
                        break

                    (inputs, new_labels), (new_anomalies, new_agreements) = batch
                    features = self.feature_extractor(inputs)

                    for name, feature in features.items():
                        self.features[name].append(feature.cpu())
                    class_label = 1 + new_labels.cpu() * 4 + new_anomalies.cpu() * 2 + new_agreements.cpu()
                    self.class_labels.append(class_label)

            self.features = {k: torch.cat(v, dim=0) for k, v in self.features.items()}
            self.class_labels = torch.cat(self.class_labels, dim=0)

    def visualize_all_features(self, data_types=['trusted', 'untrusted', 'test'], use_densmap=False, save_dir='feature_visualizations'):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Define a fixed colormap with 8 colors
        cmap = plt.cm.get_cmap('tab10', 9)
        norm = plt.Normalize(vmin=0, vmax=8)

        for feature_name in self.features.keys():
            plt.figure(figsize=(10, 8))
            embedding = self.get_embedding(data_types, feature_name, use_densmap)
            combined_class = self.class_labels
            
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=combined_class, cmap=cmap, norm=norm, s=5)
            cb = plt.colorbar(scatter, ticks=range(9))
            cb.set_ticklabels(self.get_all_class_names())
            plt.title(f"UMAP visualization of {feature_name}")
            
            img_path = save_dir / f"{feature_name}_umap.png"
            plt.savefig(img_path)
            plt.close()

            # PCA visualization
            pca = PCA(n_components=5)
            pca_result = pca.fit_transform(self.features[feature_name].to(torch.float32).numpy())

            fig, axs = plt.subplots(2, 2, figsize=(20, 16))
            for i in range(4):
                row = i // 2
                col = i % 2
                scatter = axs[row, col].scatter(pca_result[:, 0], pca_result[:, i+1], c=combined_class, cmap=cmap, norm=norm, s=5)
                axs[row, col].set_title(f"PCA: PC1 vs PC{i+1}")
                axs[row, col].set_xlabel(f"PC1")
                axs[row, col].set_ylabel(f"PC{i+2}")

            plt.suptitle(f"PCA visualization of {feature_name}")
            cb = fig.colorbar(scatter, ax=axs.ravel().tolist(), label='Class', ticks=range(9))
            cb.set_ticklabels(self.get_all_class_names())
            
            pca_img_path = save_dir / f"{feature_name}_pca.png"
            plt.savefig(pca_img_path)
            plt.close()


    def get_embedding(self, data_types, feature_name, use_densmap):
        reducer = umap.UMAP(n_components=2, densmap=use_densmap)
        return reducer.fit_transform(self.features[feature_name].to(torch.float32).numpy())

    def get_all_class_names(self):
        return [
            'Trusted',
            'Disagree normal false',
            'Agree normal false',
            'Disagree anomalous false',
            'Agree anomalous false',
            'Disagree normal true',
            'Agree normal true',
            'Disagree anomalous true',
            'Agree anomalous true',
        ]

    def train_and_visualize(self, task, data_types=['trusted', 'untrusted', 'test'], use_densmap=False, save_dir='feature_visualizations', batch_size=32):
        trusted_data = None
        untrusted_data = None
        test_data = None
        if 'trusted' in data_types:
            trusted_data = task.trusted_data
        if 'untrusted' in data_types:
            untrusted_data = task.untrusted_train_data
        if 'test' in data_types:
            test_data = task.test_data
        self.train(trusted_data, untrusted_data, test_data, batch_size=batch_size)
        self.visualize_all_features(data_types, use_densmap, save_dir)