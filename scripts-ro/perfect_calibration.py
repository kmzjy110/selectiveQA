"""
Get Histogram and Calibration graphs for MaxProb.

"""

import matplotlib.pyplot as plt

def main():
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.plot([0,1],[0,1], color='k', linestyle='dashed')
    plt.title('Perfect calibration', fontsize=18)
    plt.xlabel('Model output probability', fontsize=17)
    plt.ylabel('Probability of correctness', fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('perfect.png', dpi=400)


if __name__ == '__main__':
    main()


