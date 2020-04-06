from typing import Dict, List, Optional, Tuple

from matplotlib import pyplot as plt


def show_results(result_list: List[Dict[str, float]],
                 gt_data: Optional[List[Dict[str, float]]] = None,
                 title: str = 'Epidemic evolution'):
    x = [i for i in range(len(result_list))]
    plt.clf()

    plt.plot(x, [data_dict['quarantined'] for data_dict in result_list], 'r', color='blue', label='hospitalized')
    plt.plot(x, [data_dict['recovered'] for data_dict in result_list], 'r', color='green', label='recovered')
    plt.plot(x, [data_dict['deceased'] for data_dict in result_list], 'r', color='red', label='deceased')
    if gt_data is not None:
        gt_x = [i for i in range(len(gt_data))]
        plt.plot(gt_x, [data_dict['quarantined'] for data_dict in gt_data], '-.', color='blue', label='gt hospitalized')
        plt.plot(gt_x, [data_dict['recovered'] for data_dict in gt_data], '-.', color='green', label='gt recovered')
        plt.plot(gt_x, [data_dict['deceased'] for data_dict in gt_data], '-.', color='red', label='gt deceased')
    plt.legend()
    plt.title(title)
    plt.show()


def show_multiple_results(result_list: List[Tuple[List[Dict[str, float]], str, int]],
                          offset: int = 0,
                          title: str = 'Infected depending on quarantine end',
                          start_days: int = 20):
    plt.clf()
    max_y = 0
    max_lim = 300000
    for i, (results, result_name, end_day) in enumerate(result_list):
        x = [i for i in range(len(results))][offset:]
        y = []
        disappeared = False
        for day_number, data_dict in enumerate(results[offset:]):
            day = day_number + offset
            no_infected = data_dict['infected'] < 1 and data_dict['exposed'] < 1
            if no_infected and day > start_days and not disappeared:
                print(f'\nquarantine-end:\t{result_name}\ndeceased:{data_dict["deceased"]}')
                disappeared = True

            if disappeared:
                val = 0
            else:
                val = data_dict['infected']

            max_y = max(max_y, val)
            y.append(val)

        plt.plot(x, y, 'r', color=f'C{i}', label=f'{result_name} (day {end_day})')

    plt.legend()
    plt.title(title)
    plt.ylim(0, min(max_y, max_lim))
    plt.show()
