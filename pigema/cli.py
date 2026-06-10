import argparse

from pigema.workflows.Execute_PINNs import main as pinns_main
from pigema.workflows.Transfer_Training import main as transfer_main
from pigema.workflows.Test_Hybrid_Processing import main as hybrid_main

def main():
    parser = argparse.ArgumentParer(
        description="PI-GEMA Scientific Workflow Engine"
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train-pinns")
    sub.add_parser("calibrate")
    sub.add_parser("hybrid")

    args = parser.parse_args()

    if args.command == "train-pinns":
        pinns_main()
    elif args.command == "calibrate":
        transfer_main()
    elif args.command == "hybrid":
        hybrid_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()





