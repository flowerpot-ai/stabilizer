def get_optimizer_parameters_with_llrd(model, peak_lr, multiplicative_factor):
    num_encoder_layers = len(model.transformer.encoder.layer)
    # Task specific layer gets the peak_lr
    tsl_parameters = [
        {
            "params": [
                param for name, param in model.task_specific_layer.named_parameters()
            ],
            "param_names": [
                name for name, param in model.task_specific_layer.named_parameters()
            ],
            "lr": peak_lr,
            "name": "tsl_param_group",
        }
    ]

    # Starting from the last encoder layer each encoder layers get a lr defined by
    # current_layer_lr = prev_layer_lr * multiplicative_factor
    # the last encoder layer lr = peak_lr * multiplicative_factor
    encoder_parameters = [
        {
            "params": [param for name, param in layer.named_parameters()],
            "param_names": [name for name, param in layer.named_parameters()],
            "lr": peak_lr * (multiplicative_factor ** (num_encoder_layers - layer_num)),
            "name": f"transformer.encoder.layer.{layer_num}",
        }
        for layer_num, layer in enumerate(model.transformer.encoder.layer)
    ]

    # Embedding layer gets embedding layer lr = first encoder layer lr * multiplicative_factor
    embedding_parameters = [
        {
            "params": [
                param for name, param in model.transformer.embeddings.named_parameters()
            ],
            "param_names": [
                name for name, param in model.transformer.embeddings.named_parameters()
            ],
            "lr": peak_lr * (multiplicative_factor ** (num_encoder_layers + 1)),
            "name": "embeddings_param_group",
        }
    ]
    return tsl_parameters + encoder_parameters + embedding_parameters
