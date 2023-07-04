import enum
import math
import numpy as np
import torch
import torch.nn as nn
import timm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from data_tools.direct_dataset import DirectDataset
from utils.flags import Mode


def test_dataset():
    from utils import LOGGER
    from data_tools.bop_dataset import BOPDataset
    from data_tools.dataset import LMDataset
    from utils.cfg_utils import get_cfg
    cfg = get_cfg("/home/bmw/Documents/Sebastian/pose_project/configs/direct_method.yaml")

    dataset = BOPDataset(
        "/home/bmw/Documents/limemod/lm",#home/bmw/Documents/limemod/lm",
        Mode.TEST,
        use_cache=True,
        single_object=True,
        num_points=642,
    )
    a = dataset.models_info
    # dataset.generate_graphs(simplify_factor=10, img_width=640, img_height=480)
    #lm_dataset = LMDataset(bop_dataset=dataset)
    lm_dataset = DirectDataset(bop_dataset=dataset, cfg=cfg)
    LOGGER.info(f"dataset size: {len(lm_dataset)}")
    LOGGER.info(f"dataset[0]: {lm_dataset[0]}")
    return 0


def main() -> int:
    test_dataset()
    # graph = Graph.load(
    #     "/Users/sebastian/Documents/Projects/pose_project/data/datasets/lm/test/000005/graphs/000007_000000.npz"
    # )
    # graph.visualize()
    #test_dataset()

    ################################################
    bs = 1
    example_img = torch.rand(bs, 3, 224, 224)
    example_pose = torch.rand(bs, 3, 4)
    example_tranlation = example_pose[:, :, 3]
    example_rotation = example_pose[:, :, :3]
    cond = torch.randn(1, 7 * 7, 1024)

    channels = 3
    out_channels = 3
    graph = Graph.create_random_graph(20, channels)
    graph.set_edge_index()
    # graph = Graph.to_torch_geometric(graph)
    time_steps = torch.arange(0, 1, 1)

    ################################################
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
    graph_net(torch.from_numpy(graph.feature_matrix), torch.from_numpy(graph.edge_index), time_steps, cond)
    print(
        "number of parameters: ",
        sum(p.numel() for p in graph_net.parameters() if p.requires_grad),
    )
    # out = graph_net(graph.x, graph.edge_index, time_steps, cond)
    backbone = timm.create_model(
        model_name="maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k",
        pretrained=True,
    )
    from engine.diffusion.ddpm import LatentDiffusion, DDPMSampler, DenoiseDiffusion

    model = LatentDiffusion(
        unet_model=graph_net,
        backbone=backbone,
        latent_scaling_factor=1.0,
        n_steps=1000,
        linear_start=0,
        linear_end=1,
    )
    ddpmsampler = DDPMSampler(model)
    ddpm = DenoiseDiffusion(model, n_steps=100, device="cpu")
    cond = model.get_image_conditioning(example_img)  # 1, 1024, 7, 7
    x = torch.from_numpy(graph.feature_matrix)
    edge_index = torch.from_numpy(graph.edge_index).T

    noised_feat = ddpm.q_sample(x0=x, index=3)
    denoised_feat = ddpm.p_sample(
        x=noised_feat, edge_index=edge_index, c=cond, t=time_steps, step=3
    )
    ############################################
    graph_resnet = GraphResNetBlock(
        channels=channels, d_t_emb=256, out_channels=out_channels
    )
    gat = GATv2Conv(
        in_channels=channels, out_channels=out_channels, heads=4, concat=False
    )
    tran = TransformerConv(
        in_channels=channels, out_channels=out_channels, heads=4, concat=False
    )
    inf = InfusionTransformer(
        in_channels=channels,
        out_channels=out_channels,
        n_heads=4,
        n_layers=2,
        d_cond=256,
    )

    print(graph_resnet(graph.x, graph.edge_index).shape)
    print(inf(graph.x, graph.edge_index, cond).shape)
    print(tran(graph.x, graph.edge_index).shape)
    print(gat(graph.x, graph.edge_index).shape)

    g = torch.from_numpy(graph.adjacency_matrix).unsqueeze(0)
    h = torch.from_numpy(graph.feature_matrix).unsqueeze(0)
    t_emb = graph_net.time_step_embedding(time_steps)
    # t_emb = graph_net.time_step_embedding(time_steps)

    output = graph_net(adj_mat=g, feat_mat=h, time_steps=time_steps, cond=cond)
    print(0)

    # print(timm.models.list_models("vit*"))
    # print(timm.version.__version__)
    # model = timm.create_model(, pretrained=True)
    model = timm.create_model(
        model_name="maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k",
        pretrained=True,
    )
    model.eval()
    result = model(example_img)
    feats = model.forward_features(example_img)  # unpooled 1x1024x7x7
    feats_pooled = model.forward_head(feats, pre_logits=True)  # 1x1024
    print(result.shape)
    print(feats.shape)
    print(feats_pooled.shape)

    betas = get_named_beta_schedule("linear", 1000)
    diffuser = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.LEARNED,
        loss_type=LossType.RESCALED_KL,
    )
    pose = torch.ones(bs, 3, 3)
    timesteps = torch.tensor([1]).unsqueeze(0)
    diffused_pose = diffuser.q_sample(pose, timesteps)

    print(pose, diffused_pose)
    schedule = get_named_beta_schedule("linear", 20)

    return 0


if __name__ == "__main__":
    main()
