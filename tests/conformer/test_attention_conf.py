# from thunder.conformer.attention import MultiHeadAttention
# import torch.nn.functional as F
# from thunder.conformer.blocks import ConformerEncoder
# import torch
# from omegaconf import OmegaConf


# def test_encoder():
#     config = OmegaConf.load("conformer/model_config.yaml")
#     conformer_conf = OmegaConf.to_container(config["encoder"])
#     conformer_conf.pop("_target_")
#     # breakpoint()

#     enc = ConformerEncoder(**conformer_conf)

#     weights = torch.load("conformer/model_weights.ckpt")
#     encoder_weights = {
#         k.replace("encoder.", ""): v for k, v in weights.items() if "encoder" in k
#     }
#     enc.load_state_dict(encoder_weights, strict=True)

#     x = torch.randn(16, enc._feat_in, 256)
#     x_lens = torch.randint(0, 256, (16,))
#     # breakpoint()
#     enc(x, x_lens)


# # def compare_attention():
# #     a1 = MultiHeadAttention(n_head=10, n_feat=128, dropout_rate=0.1)
# #     # a2 = nn.MultiheadAttention(embed_dim=128, num_heads=10, dropout=0.1)
# #     a1.eval()
