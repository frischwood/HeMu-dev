

from models.TSViT.TSViTreg import TSViTreg
from models.ResNet.resnet import ConvResNet
def get_model(config, device):
    model_config = config['MODEL']


    if model_config['architecture'] == "TSViT_reg":
        return TSViTreg(model_config).to(device)
    if model_config['architecture'] == "ConvResNet":
        return ConvResNet(model_config).to(device)

    else:
        raise NameError("Model architecture %s not found, choose from:'TSViT', 'ResNetConv'" % model_config['architecture'])
