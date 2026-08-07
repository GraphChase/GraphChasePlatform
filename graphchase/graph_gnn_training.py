from __future__ import annotations

from graphchase.configs import build_gnn_pretrain_parser
from graphchase.graph.gnn_graph import GNNGraphPretrainer, build_training_settings


def main() -> None:
    parser = build_gnn_pretrain_parser()
    args = parser.parse_args()
    print("Options")
    print("=" * 30)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 30)
    settings_list = build_training_settings(args)
    trainer = GNNGraphPretrainer(args)
    trainer.train(settings_list)


if __name__ == "__main__":
    main()
