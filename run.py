import argparse
from doculens import process


def main():
    parser = argparse.ArgumentParser(description="DocuLens document image processing")
    parser.add_argument("--image", required=True, help="Path to a document photograph")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--no-warp", action="store_true", help="Skip perspective correction")
    args = parser.parse_args()
    result = process(args.image, args.output, warp=not args.no_warp)
    print("Completed:", result)


if __name__ == "__main__":
    main()
