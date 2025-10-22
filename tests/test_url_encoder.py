import torch

from src.models.url_encoder import URLEncoder


def test_url_encoder_output_shape():
    encoder = URLEncoder(
        vocab_size=128,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=2,
        bidirectional=True,
        dropout=0.1,
        pad_id=0,
        proj_dim=256,
    )

    dummy_input = torch.randint(0, 128, (3, 50), dtype=torch.long)
    output = encoder(dummy_input)

    assert output.shape == (3, 256)
    assert torch.isfinite(output).all()
