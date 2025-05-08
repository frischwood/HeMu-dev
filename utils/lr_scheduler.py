
from torch.optim.lr_scheduler import ReduceLROnPlateau

def build_scheduler(config, optimizer):
    if config['SOLVER']['lr_scheduler'] == 'ReduceLROnPlateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=config["SOLVER"]["patience"], factor=0.1, threshold=1e-4, threshold_mode="rel", cooldown=0, min_lr=0, eps=1e-8)
    return lr_scheduler
