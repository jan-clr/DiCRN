import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import hydra
import os
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings


import src.utils as utils
from src.guidance.guidance_diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.datasets import SWITCHESGraph
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from src.analysis.spectre_utils import SWITCHESSamplingMetrics
from src.analysis.visualization import CRNVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.utils import update_config_with_new_keys
from src.guidance.switches_regressor_discrete import SwitchesRegressorDiscrete


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    saved_cfg = cfg.copy()

    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    final_samples_to_generate = cfg.general.final_model_samples_to_generate
    final_chains_to_save = cfg.general.final_model_chains_to_save
    batch_size = cfg.train.batch_size
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg.general.final_model_samples_to_generate = final_samples_to_generate
    cfg.general.final_model_chains_to_save = final_chains_to_save
    cfg.train.batch_size = batch_size
    cfg = update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


@hydra.main(config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    datamodule = SWITCHESGraph.SWITCHESGraphDataModule(cfg, regressor=True)
    dataset_infos = SWITCHESGraph.SWITCHESDatasetInfos(datamodule=datamodule, dataset_config=dataset_config)

    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)

    train_metrics = TrainAbstractMetricsDiscrete()
    sampling_metrics = SWITCHESSamplingMetrics(datamodule)
    visualization_tools = CRNVisualization()

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features, 'load_model': True}

    # When testing, previous configuration is fully loaded
    cfg_pretrained, guidance_sampling_model = get_resume(cfg, model_kwargs)

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.model = cfg_pretrained.model
    model_kwargs['load_model'] = False

    utils.create_folders(cfg)

    # load pretrained regressor
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    guidance_model = SwitchesRegressorDiscrete.load_from_checkpoint(os.path.join(cfg.general.trained_regressor_path))

    model_kwargs['guidance_model'] = guidance_model

    if cfg.general.name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      gpus=cfg.general.gpus if torch.cuda.is_available() else 0,
                      limit_test_batches=100,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,        # TODO CHANGE with ray
                      enable_progress_bar=False,
                      logger=[],
                      )

    # add for conditional sampling
    model = guidance_sampling_model
    model.args = cfg
    model.guidance_model = guidance_model
    trainer.test(model, datamodule=datamodule, ckpt_path=None)


if __name__ == '__main__':
    main()
