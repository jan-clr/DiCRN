import graph_tool as gt

import hydra
import torch
from omegaconf import DictConfig
from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    print(cfg)
    from datasets.SWITCHESGraph import SWITCHESGraphDataModule, SWITCHESDatasetInfos
    from analysis.spectre_utils import SWITCHESSamplingMetrics
    from analysis.visualization import NonMolecularVisualization, CRNVisualization


    dataset_config = cfg["dataset"]
    datamodule = SWITCHESGraphDataModule(cfg)
    sampling_metrics = SWITCHESSamplingMetrics(datamodule)

    dataset_infos = SWITCHESDatasetInfos(datamodule, dataset_config)
    train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
    visualization_tools = CRNVisualization()

    extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}
    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    print("Using GPU:", use_gpu)
    utils.create_folders(cfg)

    model = DiscreteDenoisingDiffusion.load_from_checkpoint(cfg.checkpoint, dataset_infos=dataset_infos, train_metrics=train_metrics, sampling_metrics=sampling_metrics, visualization_tools=visualization_tools)

    model.generate_samples()


if __name__ == '__main__':
    main()
