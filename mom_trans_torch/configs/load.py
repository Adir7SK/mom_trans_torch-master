import os

# import json
import yaml
import itertools
import pandas as pd

CORRELATION_SPAN_DEFAULT = 252

ARCHITECTURES = [
    "LSTM",
    "LSTM_SIMPLE",
    "MOM_TRANS",
    "MOM_TRANS_SIMPLE",
    # "X_MOM_TRANS",
    "X_TREND",
    "X_TREND_CS",
    "CSMOM",
    "MLP",
    "MLP_SIMPLE",
    "TemporalFusionTransformer",
    "PatchTST",
    "PatchTST2",
    "PatchCopy",
    "PatchCopy2",
    "PxLSTM",
    "xLSTM",
    "Informer",
    "FEDFormer",
    "DLinear",
    "NLinear",
    "TimeMixer",
    "iTransformer",
    "iTransformer2",
    "Mamba",
    "Mamba2",
]


# def load_data_settings(file_name: str) -> dict:
#     """Load the data settings

#     :param str file_name: the file name
#     :return: the data settings
#     :rtype: dict
#     """

#     with open(
#         os.path.join("mom_trans_torch", "configs", "data_settings", f"{file_name}.yaml"),
#         "r",
#         encoding="UTF-8",
#     ) as f:
#         configs = yaml.safe_load(f)
#     configs["description"] = file_name
#     return configs


def load_train_settings(file_name: str) -> dict:
    """Load the train settings

    :param str file_name: the file name
    :return: the train settings
    :rtype: dict
    """
    with open(
        # os.path.join("mom_trans_torch", "configs", "train_settings", f"{file_name}.yaml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_settings", f"{file_name}.yaml"),
        "r",
        encoding="UTF-8",
    ) as f:
        configs = yaml.safe_load(f)
    configs["description"] = file_name
    return configs


def load_settings_for_architecture(file_name: str, architecture: str) -> dict:
    """Load the settings for the architecture

    :param str file_name: the file name
    :param str architecture: the architecture
    :return: the settings
    :rtype: dict
    """

    assert architecture in ARCHITECTURES
    settings = load_train_settings(file_name)

    # if settings["all_hidden_attention"]:
    #     assert settings["context_seq_len"] == settings["default_seq_len"]

    RUN_NAME = architecture
    if settings["test_run"]:
        RUN_NAME = f"TEST/{RUN_NAME}"

    settings["run_name"] = RUN_NAME

    settings["use_contexts"] = architecture in [
        # "NP_LSTM_FULL",
        # "NP_LSTM_ATTENTION",
        "X_TREND",
        "X_TREND_CS",
        # "NP_ATT_FULL",
        # "NP_MOM_TRANS_FULL",
    ]

    settings["cross_section"] = architecture in [
        "X_TREND_CS",
        "CSMOM",
    ]

    settings["signal_combine"] = architecture in ["SIGCOM"]

    # settings["MOM_TRANS"] = architecture in [
    #     "MOM_TRANS",
    #     # "MOM_TRANS_FULL",
    #     # "NP_MOM_TRANS_FULL",
    # ]

    if architecture not in [
        # "NP_LSTM_FULL",
        # "NP_LSTM_ATTENTION",
        "X_TREND",
        # "NP_ATT_FULL",
        # "NP_MOM_TRANS_FULL",
    ]:
        settings["num_context"] = 0

    if "correlation_span" not in settings:
        settings["correlation_span"] = CORRELATION_SPAN_DEFAULT

    if "local_time_embedding" not in settings:
        settings["local_time_embedding"] = False
    
    if "train_target_override" not in settings:
        settings["train_target_override"] = None

    if "valid_target_override" not in settings:
        settings["valid_target_override"] = None

    if not "extra_data_pre_steps" in settings:
        settings["extra_data_pre_steps"] = 0
    
    if not "replace_sharpe_loss" in settings:
        settings["replace_sharpe_loss"] = None

    if not "specify_weight_features" in settings:
        settings["specify_weight_features"] = []

    if not "tcost_inputs" in settings:
        settings["tcost_inputs"] = False
    
    if not "sigcom_sgn" in settings:
        settings["sigcom_sgn"] = False

    return settings

def load_settings_for_finetune(file_name: str) -> dict:
    with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_settings", f"{file_name}.yaml"),
        # os.path.join("mom_trans_torch", "configs", "finetune_settings", f"{file_name}.yaml"),
        "r",
        encoding="UTF-8",
    ) as f:
        configs = yaml.safe_load(f)
    configs["description"] = file_name
    return configs

# def load_sweep_settings(file_name: str) -> dict:
#     """Load the sweep settings
#
#     :param str file_name: the file name
#     :return: the sweep settings
#     :rtype: dict
#     """
#     with open(
#             os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_settings", f"{file_name}.yaml"),
#         # os.path.join("mom_trans_torch", "configs", "sweep_settings", f"{file_name}.yaml"),
#         "r",
#         encoding="UTF-8",
#     ) as f:
#         configs = yaml.safe_load(f)
#     # configs["name"] = file_name
#
#     # keys, values = zip(*configs['parameters'].items())
#     # all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
#     #
#     # # Filter: stride <= patch_len
#     # configs['parameters'] = [c for c in all_combos if c["stride"] <= c["patch_len"]]
#
#     # Add constraint for patch_len and stride
#     if 'parameters' in configs and 'patch_len' in configs['parameters'] and 'stride' in configs['parameters']:
#         configs['parameters']['stride']['conditions'] = {
#             'value': "${patch_len}",  # wandb syntax for referencing another parameter
#             'operator': "<="  # stride must be <= patch_len
#         }
#
#     return configs

def load_sweep_settings(file_name: str) -> dict:
    """Load the sweep settings

    :param str file_name: the file name
    :return: the sweep settings
    :rtype: dict
    """
    with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_settings", f"{file_name}.yaml"),
        # os.path.join("mom_trans_torch", "configs", "sweep_settings", f"{file_name}.yaml"),
        "r",
        encoding="UTF-8",
    ) as f:
        configs = yaml.safe_load(f)
    # configs["name"] = file_name

    return configs

# def load_sweep_settings(file_name: str) -> dict:
#     with open(
#         os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_settings", f"{file_name}.yaml"),
#         "r",
#         encoding="UTF-8",
#     ) as f:
#         configs = yaml.safe_load(f)
#
#     # Enforce stride <= patch_len constraint
#     if 'parameters' in configs and 'patch_len' in configs['parameters'] and 'stride' in configs['parameters']:
#         patch_lens = configs['parameters']['patch_len'].get('values', [])
#         strides = configs['parameters']['stride'].get('values', [])
#
#         # Create grid entries for valid combinations only
#         grid = []
#         for p in patch_lens:
#             for s in strides:
#                 if s <= p:  # Ensure stride <= patch_len
#                     grid.append({'patch_len': p, 'stride': s})
#
#         # Replace with a new grid parameter
#         if grid:
#             configs['parameters']['pair'] = {"values": grid}
#             # Remove the original patch_len/stride entries so sweep only uses the pair
#             del configs['parameters']['patch_len']
#             del configs['parameters']['stride']
#
#     return configs
