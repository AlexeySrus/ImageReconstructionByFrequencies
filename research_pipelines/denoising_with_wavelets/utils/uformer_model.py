from model import Uformer


def get_uformer_model(image_size: int):
    depths=[1, 2, 8, 8, 2, 2, 2, 2, 2]
    model_restoration = Uformer(img_size=image_size, embed_dim=32,depths=depths,
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
    return model_restoration


if __name__ == '__main__':
    model = get_uformer_model(256)

    pamars_count = sum(p.numel() for p in model.parameters())
    print('Parameters count: {:.2f} M'.format(pamars_count / 1E+6))
