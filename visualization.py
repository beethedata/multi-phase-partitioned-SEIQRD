from typing import Dict, List

from matplotlib import pyplot as plt


def show_results(result_list: List[Dict[str, float]]):
    x = [i for i in range(len(result_list))]
    plt.clf()
    plt.plot(x, [data_dict['infected'] for data_dict in result_list], 'r', color='blue', label='infected')
    plt.plot(x, [data_dict['recovered'] for data_dict in result_list], 'r', color='green', label='recovered')
    plt.plot(x, [data_dict['deceased'] for data_dict in result_list], 'r', color='red', label='deceased')
    plt.title('Epidemic evolution')
    plt.show()
    pass
