import torch
from omegaconf import OmegaConf
from ldm.utils import instantiate_from_config


# init the model
def load_model_from_config(config):
    model = instantiate_from_config(config.model)
    model.eval()
    return model


if __name__ == "__main__":
    config_path = "/Users/gunneo/AI/Codes/Generation/LDM/settings/VAEs/kl-f8/config.yaml"

    config = OmegaConf.load(config_path)
    model = load_model_from_config(config)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    print("Model loaded successfully!")

    # with torch.no_grad():
    #     x = torch.randn(1, 3, 256, 256)
    #     out = model(x)
    #     print(out.shape)
