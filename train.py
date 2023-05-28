import torch
from torch.utils.data import DataLoader
import timm
from torch_geometric.data import Data
from utils import LOGGER
from models.custom_graphnet import CustomGraphNet
from engine.diffusion.ddpm import DenoiseDiffusion, LatentDiffusion
from data_tools.bop_dataset import BOPDataset, Flag
from data_tools.dataset import LMDataset


def train():
    dataset = BOPDataset(
        "/Users/sebastian/Documents/Projects/pose_project/data/datasets/lm",
        Flag.TEST,
        use_cache=True,
        single_object=True,
    )
    # dataset.generate_graphs(simplify_factor=10, img_width=640, img_height=480)
    lm_dataset = LMDataset(bop_dataset=dataset)
    train_loader = DataLoader(lm_dataset, batch_size=1, shuffle=False)

    channels = 3
    out_channels = 3
    graph_net = CustomGraphNet(
        in_channels=channels,
        out_channels=out_channels,
        channels=32,  # min 32
        n_res_blocks=2,
        attention_levels=[1, 2],  # 4, 6, 7, 8],
        channel_multipliers=[1, 1, 2, 2],  # 4, 4, 8, 8],
        n_heads=4,
        tf_layers=2,
        d_cond=1024,
    )
    print(
        "number of parameters: ",
        sum(p.numel() for p in graph_net.parameters() if p.requires_grad),
    )
    backbone = timm.create_model(
        model_name="maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k",
        pretrained=True,
        
        #out_indices=(3,),
    )
    out = backbone.forward_features(torch.rand(2, 3, 224, 224))
    model = LatentDiffusion(
        unet_model=graph_net,
        backbone=backbone,
        latent_scaling_factor=1.0,
        n_steps=1000,
        linear_start=0,
        linear_end=1,
    )
    diffusion = DenoiseDiffusion(
        graph_net, cond_model=backbone, n_steps=100, device="cpu"
    )
    for data in lm_dataset: #train_loader:
        # data = data.to("cpu")
        loss = diffusion.loss(img=data["img"].unsqueeze(0), x0=data["x"], edge_index=data["edge_index"])
        break


def main() -> int:
    train()


if __name__ == "__main__":
    main()
