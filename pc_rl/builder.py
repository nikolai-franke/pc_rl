from pc_rl.models.embedder import Embedder


def build_embedder(embedder_conf):
    return Embedder(
        hidden_layers=embedder_conf["hidden_layers"],
        embedding_size=embedder_conf["embedding_size"],
        neighborhood_size=embedder_conf["neighborhood_size"],
        sampling_ratio=embedder_conf["sampling_ratio"],
        random_start=embedder_conf["random_start"],
    )
