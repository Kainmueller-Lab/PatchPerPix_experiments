import argparse
import os
import random
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest="in_dir", required=True)
    parser.add_argument('-o', dest="out_dir", required=True)
    parser.add_argument('-f', dest="out_format", default="hdf")
    args = parser.parse_args()

    trainD = os.path.join(args.out_dir, "train")
    testD = os.path.join(args.out_dir, "test")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(trainD, exist_ok=True)
    os.makedirs(testD, exist_ok=True)

    trainFls = [
        "A01",
        "A02",
        "A03",
        "A04",
        "A05",
        "A06",
        "A07",
        "A08",
        "A09",
        "A10",
        "A11",
        "A12",
        "A13",
        "A14",
        "A15",
        "A16",
        "A17",
        "A18",
        "A19",
        "A20",
        "A21",
        "A22",
        "A23",
        "A24",
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B10",
        "B11",
        "B12",
        "B13",
        "B14",
        "B15",
        "B16",
        "B17",
        "B18",
        "B19",
        "B20",
        "B21",
        "B22",
        "B23",
        "B24",
        "C01",
        "C02"
    ]

    testFls = [
        "C03",
        "C04",
        "C05",
        "C06",
        "C07",
        "C08",
        "C09",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "D01",
        "D02",
        "D03",
        "D04",
        "D05",
        "D06",
        "D07",
        "D08",
        "D09",
        "D10",
        "D11",
        "D12",
        "D13",
        "D14",
        "D15",
        "D16",
        "D17",
        "D18",
        "D19",
        "D20",
        "D21",
        "D22",
        "D23",
        "D24",
        "E01",
        "E02",
        "E03",
        "E04"
    ]

    random.shuffle(trainFls)
    fmt = "." + args.out_format
    if args.out_format == "hdf":
        copy_func = shutil.copy2
    elif args.out_format == "zarr":
        copy_func = shutil.copytree

    for fl in trainFls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(trainD, fl + fmt))

    for fl in testFls:
        copy_func(os.path.join(args.in_dir, fl + fmt),
                  os.path.join(testD, fl + fmt))


if __name__ == "__main__":
    main()
