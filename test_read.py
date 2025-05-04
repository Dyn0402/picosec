

import uproot


def main():
    path = '/home/dylan/Desktop/picosec/data/2022_October_h4/processedTrees/ParameterTrees/Run224-Pool2_treeParam.root'

    with uproot.open(path) as file:
        print(file)
        print(file.keys())


if __name__ == '__main__':
    main()
