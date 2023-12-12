from model import Uformer


def get_uformer_model(image_size: int):
    depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
    model_restoration = Uformer(img_size=image_size, embed_dim=16,depths=depths,
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
    return model_restoration
