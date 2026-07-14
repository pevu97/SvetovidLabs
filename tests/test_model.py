import torch

from svetovid import ConvAutoencoder


def test_forward_preserves_shape():
    model = ConvAutoencoder()
    model.eval()
    x = torch.rand(2, 1, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape


def test_output_in_unit_range():
    """Decoder ends with Sigmoid, so reconstructions must stay in [0, 1]."""
    model = ConvAutoencoder()
    model.eval()
    x = torch.rand(2, 1, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_latent_is_compressed():
    """Encoder should downsample 256x256 by a factor of 2^5 = 32."""
    model = ConvAutoencoder()
    model.eval()
    x = torch.rand(1, 1, 256, 256)
    with torch.no_grad():
        z = model.encoder(x)
    assert z.shape == (1, 512, 8, 8)


def test_checkpoint_roundtrip(tmp_path):
    """state_dict saved in the training format loads back correctly."""
    model = ConvAutoencoder()
    path = tmp_path / "ckpt.pth"
    torch.save({"model_state_dict": model.state_dict()}, path)

    restored = ConvAutoencoder()
    checkpoint = torch.load(path, map_location="cpu")
    restored.load_state_dict(checkpoint["model_state_dict"])
