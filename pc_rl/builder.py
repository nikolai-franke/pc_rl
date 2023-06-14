from pc_rl.models.embedder import Embedder
from pc_rl.models.transformer import MaskTransformer


def build_embedder(embedder_conf):
    conf = embedder_conf
    return Embedder(
        hidden_layers=conf["hidden_layers"],
        embedding_size=conf["embedding_size"],
        neighborhood_size=conf["neighborhood_size"],
        sampling_ratio=conf["sampling_ratio"],
        random_start=conf["random_start"],
    )


def build_mask_transformer(transformer_conf, embedder_conf):
    conf = transformer_conf
    embedder = build_embedder(embedder_conf)
    return MaskTransformer(
        mask_ratio=conf["mask_ratio"],
        depth=conf["depth"],
        num_heads=conf["num_heads"],
        embedder=embedder,
        mask_type=conf["mask_type"],
    )
