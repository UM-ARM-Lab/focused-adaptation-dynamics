import torch


def get_pour_type_and_error(batch, outputs, error, sample_i):
    sample_target_volume = batch["target_volume"][sample_i]
    sample_control_volume = batch["control_volume"][sample_i]
    init_target_vol = sample_target_volume[0]
    init_control_vol = sample_control_volume[0]
    if "time_mask" in batch:
        final_idx = int(batch["time_mask"][sample_i].sum() - 1)
    else:
        final_idx = int(batch["target_volume"][sample_i].shape[0] - 1)
    final_target_vol = sample_target_volume[final_idx]
    final_control_vol = sample_control_volume[final_idx]
    if outputs is not None:
        pour_error = torch.abs(
            final_target_vol - outputs["target_volume"][sample_i][final_idx]).cpu().detach().numpy().item()
    else:
        pour_error = None
    if len(error.shape) == 1:
        sample_error = error[sample_i].cpu().detach().numpy().item()
    else:
        sample_error = error[sample_i][final_idx].cpu().detach().numpy().item()
    if final_target_vol - init_target_vol > 0.95:  # easy pour
        pour_type = "easy_pour"
    elif (
            final_target_vol - init_target_vol > 0 and final_target_vol - init_target_vol < 0.8) or final_control_vol - init_control_vol < -0.05:
        pour_type = "hard_pour"
    else:
        pour_type = "not_pour"
    return pour_error, pour_type, sample_error
