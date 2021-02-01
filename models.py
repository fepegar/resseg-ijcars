from unet import UNet


def get_unet():
    model = UNet(
        in_channels=1,
        out_classes=2,
        dimensions=3,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        normalization='batch',
        pooling_type='max',
        padding=True,
        padding_mode='replicate',
        residual=False,
        initial_dilation=1,
        activation='PReLU',
        upsampling_type='linear',
        dropout=0,
        monte_carlo_dropout=0.5,
    )
    return model


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module):
    for param in module.parameters():
        param.requires_grad = True


def freeze_layers(model, num_layers):
    # num_layers should be None, 1, 2 or 3
    if num_layers is None:
        return
    if hasattr(model, 'module'):  # DataParallel
        model = model.module
    if num_layers >= 1:
        layer = model.classifier.conv_layer
        freeze(layer)
    if num_layers >= 2:
        first = -(num_layers - 1)
        for module in model.decoder.decoding_blocks[first:]:
            freeze(module)


def freeze_except(model, num_layers):
    # num_layers should be None, 1, 2 or 3
    # If None, nothing happens
    # If 1, all layers are frozen but the classifier
    # If 2 or 3, conv layers from last decoder block will also not be frozen
    if num_layers is None:
        return
    if hasattr(model, 'module'):  # DataParallel
        model = model.module
    freeze(model)
    if num_layers > 0:
        unfreeze(model.classifier)
    if num_layers > 1:
        unfreeze(model.decoder.decoding_blocks[-1])
    if num_layers == 3:
        unfreeze(model.decoder.decoding_blocks[-2])
