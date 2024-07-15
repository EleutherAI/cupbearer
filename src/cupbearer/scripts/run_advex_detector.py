import torch
import argparse
from cupbearer import detectors, tasks, utils, scripts
from cupbearer.detectors.statistical import atp_detector
from cupbearer.detectors.statistical.atp_detector import ImpactfulDeviationDetector
from pathlib import Path
from cupbearer.detectors.statistical.probe_detector import probe_error
from cupbearer.detectors.statistical.helpers import mahalanobis_from_data, local_outlier_factor
from cupbearer.scripts.measure_accuracy import measure_accuracy
import gc

datasets = ["cifar10"]  # Add more datasets if supported in the future

def get_batchnorm_layers(model, max_n_layers):
    batchnorm_layers = [name + '.input' for name, _ in model.named_modules() if 'batchnorm' in name.lower() or 'bnorm' in name.lower() or 'bn' in name.lower()]
    mnames = [name for name, _ in model.named_modules()]
    if not batchnorm_layers:
        raise ValueError("No batch normalization layers found in the model.")
    
    if len(batchnorm_layers) <= max_n_layers:
        return batchnorm_layers
    
    # Select evenly spaced layers
    indices = torch.linspace(0, len(batchnorm_layers) - 1, max_n_layers).long()
    return [batchnorm_layers[i] for i in indices]


def main(
        dataset, 
        detector_type, 
        max_n_layers, 
        model_name, 
        features, 
        ablation, 
        k=20, 
        layerwise=True, 
        alpha=8,
        impact_threshold=1.e-4,
        epsilon=8/255,
        n_train_examples=500,
        n_test_examples=50,
        run_id=''):
    
    task = tasks.adversarial_image_task(
        dataset=dataset,
        model_name=model_name,
        epsilon=epsilon,
        n_train_examples=n_train_examples,
        n_test_examples=n_test_examples
    )

    # Get the batch normalization layers
    layer_list = get_batchnorm_layers(task.model, max_n_layers)

    # Absolute path so everyone on Eleuther nodes can reuse the same cache. You might need to mess with permissions.
    cache_path = f"/mnt/ssd-1/david/cupbearer/cache/{dataset}-{model_name}-{max_n_layers}.pt"
    activation_cache = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if Path(cache_path).exists():
        activation_cache = detectors.activation_based.ActivationCache.load(cache_path, device)

    activation_processing_function = lambda x, *args: x

    if detector_type == "accuracy":
        measure_accuracy(
            task, 
            batch_size=32, 
            pbar=False, 
            save_path=f"logs/adv_image/{dataset}-accuracy", 
            histogram_percentile=95)
        return

    elif features == "activations":
        batch_size = 20
        eval_batch_size = 20

        if detector_type == "mahalanobis":
            detector = detectors.MahalanobisDetector(
                activation_names=layer_list,
                activation_processing_func=activation_processing_function,
                cache=activation_cache
            )
        elif detector_type == "lof":
            detector = detectors.statistical.lof_detector.LOFDetector(
                activation_names=layer_list,
                activation_processing_func=activation_processing_function,
                cache=activation_cache
            )
        elif detector_type == 'que':
            detector = detectors.statistical.que_detector.QuantumEntropyDetector(
                activation_names=layer_list,
                activation_processing_func=activation_processing_function,
                alpha=alpha,
                cache=activation_cache
            )
        elif detector_type == 'isoforest':
            detector = detectors.statistical.isoforest_detector.IsoForestDetector(
                activation_names=layer_list,
                activation_processing_func=activation_processing_function,
                cache=activation_cache
            )
        # Add more detectors as needed

    elif features == "attribution":

        def effect_prob_func(logits):
            top_logits, _ = logits.max(dim=-1)
            rest_logits = (logits.sum(dim=-1) - top_logits) / (logits.size(-1) - 1)
            return (top_logits - rest_logits).sum()

        batch_size = 4
        eval_batch_size = 4

        layer_dict = { layer.replace('.input', '') : (4096,) for layer in layer_list}

        effect_capture_args = {}
        if ablation == 'raw':
            effect_capture_method = 'raw'
        else:
            effect_capture_method = 'atp'
            effect_capture_args['ablation'] = 'pcs'
            effect_capture_args['n_pcs'] = 32


        if detector_type == "mahalanobis":
            detector = atp_detector.MahaAttributionDetector(
                layer_dict, 
                effect_prob_func, 
                effect_capture_method=effect_capture_method,
                effect_capture_args=effect_capture_args,
                activation_processing_func=activation_processing_function,
                cache=activation_cache,
                append_activations=False,
                head_dim=0
            )

        elif detector_type == 'que':
            detector = atp_detector.QueAttributionDetector(
                layer_dict, 
                effect_prob_func, 
                effect_capture_method=effect_capture_method,
                effect_capture_args=effect_capture_args,
                activation_processing_func=activation_processing_function,
                cache=activation_cache,
                append_activations=False,
                head_dim=0
            )

    # Add more feature types if needed

    save_path = f"logs/adv_image/{dataset}-{detector_type}-{features}-{model_name}-{max_n_layers}-{run_id}"

    if features == "attribution":
        save_path += f"-{ablation}"
    if detector_type == "lof":
        save_path += f"-{k}"
    if detector_type == "que":
        save_path += f"-{alpha}"

    if Path(save_path).exists():
        detector.load_weights(Path(save_path) / "detector")
        scripts.eval_detector(task, detector, save_path, pbar=True, batch_size=eval_batch_size, train_from_test=False, layerwise=layerwise)
    else:
        scripts.train_detector(task, detector, 
                        batch_size=batch_size, 
                        save_path=save_path, 
                        eval_batch_size=eval_batch_size,
                        pbar=True,
                        train_from_test=False,
                        layerwise=layerwise)
    
    del task, detector
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Adversarial Image Detector")
    parser.add_argument('--detector_type', type=str, required=True, help='Type of detector to use')
    parser.add_argument('--model_name', type=str, default='Standard', help='Name of the model to use')
    parser.add_argument('--run_id', type=str, default='', help='Run ID to use')
    parser.add_argument('--max_n_layers', type=int, default=8, help='Maximum number of layers to use')
    parser.add_argument('--features', type=str, required=True, help='Features to use (activations, etc.)')
    parser.add_argument('--ablation', type=str, default='pcs', choices=['pcs', 'raw'], help='Ablation to use')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use')
    parser.add_argument('--k', type=int, default=20, help='k to use for LOF')
    parser.add_argument('--alpha', type=float, default=8, help='Alpha to use for QUE')
    parser.add_argument('--layerwise', action='store_true', default=False, help='Evaluate layerwise instead of aggregated')
    parser.add_argument('--impact_threshold', type=float, default=1e-4, help='Impact threshold for ImpactfulDeviationDetector')
    parser.add_argument('--epsilon', type=float, default=8/255, help='Epsilon for adversarial examples')
    parser.add_argument('--n_train_examples', type=int, default=500, help='Number of training examples')
    parser.add_argument('--n_test_examples', type=int, default=50, help='Number of test examples')

    args = parser.parse_args()

    def run_main_with_args(dataset, alpha):
        main(
            dataset, 
            args.detector_type, 
            args.max_n_layers,
            args.model_name, 
            args.features, 
            args.ablation, 
            k=args.k, 
            alpha=alpha, 
            layerwise=args.layerwise,
            impact_threshold=args.impact_threshold,
            epsilon=args.epsilon,
            n_train_examples=args.n_train_examples,
            n_test_examples=args.n_test_examples,
            run_id=args.run_id
        )

    if args.dataset == "all":
        for dataset in datasets:
            run_main_with_args(dataset, args.alpha)
    else:
        run_main_with_args(args.dataset, args.alpha)