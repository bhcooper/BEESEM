#!/usr/bin/env python

# coding: utf-8
import argparse
from universal import *


def driver():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input file path')
    parser.add_argument('-o', '--output', help='output file path (optional)')
    args = parser.parse_args()

    Formatter(args.input).CSV(args.output)


if __name__ == '__main__':
    driver()
