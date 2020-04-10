from typing import Dict, List, Optional, Tuple, Union

from matplotlib import pyplot as plt


def show_results(result_list: List[Dict[str, float]],
                 gt_data: Optional[List[Dict[str, float]]] = None,
                 title: str = 'Epidemic evolution fit',
                 infection_start_date: str = '02-20-2020',
                 predict_len: int = 0):
    x = [i for i in range(len(result_list))]
    plt.clf()
    if predict_len > 0:
        plt.plot(x[:-predict_len], [data_dict['quarantined'] for data_dict in result_list[:-predict_len]],
                 'r', color='blue', label='hospitalized')
        plt.plot(x[:-predict_len], [data_dict['deceased'] for data_dict in result_list[:-predict_len]],
                 'r', color='red', label='deceased')
        plt.plot(x[-predict_len:], [data_dict['quarantined'] for data_dict in result_list[-predict_len:]],
                 '.', color='blue', label='predicted hospitalized')
        plt.plot(x[-predict_len:], [data_dict['deceased'] for data_dict in result_list[-predict_len:]],
                 '.', color='red', label='predicted deceased')
    else:
        plt.plot(x, [data_dict['quarantined'] for data_dict in result_list], 'r', color='blue', label='hospitalized')
        plt.plot(x, [data_dict['deceased'] for data_dict in result_list], 'r', color='red', label='deceased')
    # plt.plot(x, [data_dict['infected'] for data_dict in result_list], 'r', color='green', label='recovered')
    # plt.plot(x, [data_dict['recovered'] for data_dict in result_list], 'r', color='green', label='recovered')

    draw_gt_data(gt_data=gt_data)
    plt.legend()
    plt.title(title)
    plt.ylabel('Number of individuals')
    plt.xlabel(f'Days since infection start ({infection_start_date})')
    plt.show()


def show_multiple_results(result_list: List[Tuple[List[Dict[str, float]], str, int]],
                          offset: int = 0,
                          title: str = 'Infected depending on quarantine end',
                          # title: str = 'Effects of quarantine on evolution of infected individuals',
                          # title: str = 'Predicted total number of infections',
                          # title: str = 'Predicted number of deaths',
                          infection_start_date: str = '02-20-2020',
                          start_days: int = 20, top_lim: Optional[int] = None, logarithmic: bool = False,
                          gt_data: Optional[List[Dict[str, Union[int, float]]]] = None):
    plt.clf()
    max_y = 0
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

        if logarithmic:
            plt.semilogy(x, y, 'r', color=f'C{i}', label=f'{result_name} (day {end_day})')
        else:
            plt.plot(x, y, 'r', color=f'C{i}', label=f'{result_name} (day {end_day})')

        # plt.plot(x, [data_dict['infected'] for data_dict in results], 'r', color='orange', label='infected')
        # plt.plot(x, [data_dict['exposed'] for data_dict in results], 'r', color='olive', label='exposed')
        # plt.plot(x, [data_dict['quarantined'] for data_dict in results], 'r', color='blue', label='quarantined')

        # total_infected = [data_dict['exposed'] + data_dict['infected'] + data_dict['quarantined'] +
        #                   data_dict['recovered'] + data_dict['deceased'] for data_dict in results]
        # plt.plot(x, total_infected, 'r', color='blue', label='total infected')

        # plt.plot(x, [data_dict['deceased'] for data_dict in results], 'r', color='red', label='deceased')

    draw_gt_data(gt_data=gt_data)

    plt.legend()
    plt.title(title)
    if top_lim is not None:
        if logarithmic:
            plt.ylim(1, min(max_y, top_lim))
        else:
            plt.ylim(0, min(max_y, top_lim))
    plt.ylabel('Number of individuals')
    plt.xlabel(f'Days since infection start ({infection_start_date})')
    plt.show()


def draw_gt_data(gt_data: Optional[List[Dict[str, Union[int, float]]]] = None):
    if gt_data is not None:
        gt_x = [i for i in range(len(gt_data))]
        plt.plot(gt_x, [data_dict['quarantined'] for data_dict in gt_data], '-.', color='blue', label='gt hospitalized')
        # plt.plot(gt_x, [data_dict['recovered'] for data_dict in gt_data], '-.', color='green', label='gt recovered')
        plt.plot(gt_x, [data_dict['deceased'] for data_dict in gt_data], '-.', color='red', label='gt deceased')
