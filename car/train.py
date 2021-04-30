"""

"""

from data import Data
from random import randint

def main():
    print("Loading data...")
    data = Data()
    data.load_files()
    data.process_data()
    episodes = data.get_episodes()
    num_episodes = len(episodes)
    features, labels = data.get_collapsed_data(episodes[:5])
    m = features.shape[0]
    print(features.shape, labels.shape)
    for i in range(10):
        num = i + 990
        print(labels[num])
        data.disp_img(features[num])
    print("Data loaded!")

if __name__ == "__main__":
    main()