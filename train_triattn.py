#!/usr/bin/env python3
import argparse

from cogtran2.training import add_common_arguments, train


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
