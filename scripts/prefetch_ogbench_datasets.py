import argparse

import ogbench


def main() -> None:
    parser = argparse.ArgumentParser(description='Prefetch OGBench datasets into a local directory.')
    parser.add_argument(
        '--dataset-dir',
        default=None,
        help='Directory to store datasets (defaults to OGBench default: ~/.ogbench/data).',
    )
    parser.add_argument(
        'dataset_names',
        nargs='*',
        default=[
            'antmaze-medium-stitch-v0',
            'antmaze-large-stitch-v0',
            'antmaze-giant-stitch-v0',
            'antmaze-teleport-stitch-v0',
        ],
        help='Dataset names to download.',
    )
    args = parser.parse_args()

    if args.dataset_dir is None:
        ogbench.download_datasets(args.dataset_names)
    else:
        ogbench.download_datasets(args.dataset_names, dataset_dir=args.dataset_dir)


if __name__ == '__main__':
    main()

