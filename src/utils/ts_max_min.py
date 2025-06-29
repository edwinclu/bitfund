

def local_max_min(bars: list[float], local_length: int) -> dict[str, list[int]]:
    """
    :param bars: The list of bars or timeseries
    :param local_length: The number of neighboring bars on left/right to be considered a local max/min
    :return: The list of local max/min index
    """
    ret = {"max": [], "min": [], "all": []}
    # not long enough
    if len(bars) < 2 * local_length - 1:
        return ret
    for i in range(local_length, len(bars)-local_length):
        print(i, bars[i])
        if bars[i] == max(bars[i-local_length: i+local_length+1]):
            ret["max"].append(i)
            ret["all"].append(i)
        elif bars[i] == min(bars[i-local_length: i+local_length+1]):
            ret["min"].append(i)
            ret["all"].append(i)
    return ret


def find_mean_line(bars: list[float], local_max_min: dict[str, list[int]], max_outliers: int=1) -> list[float]:
    """
    :param bars: The list of bars or timeseries
    :param local_max_min: The list of local max/min index
    :param max_outliers: Max/Min should appear above/below the line. max_outliers controls the number of outliers
    allowed to appear on the other side of the line before considering the line ended and a new line should be started.
    :return: the mean lines of the bars in a given period
    """
    pass


def _segment_bars(bars: list[float], local_max_min: dict[str, list[int]], max_outliers: int=1) -> list[tuple[int, int]]:
    """
    :param bars: The list of bars or timeseries
    :param local_max_min: The list of local max/min index
    :param max_outliers: Max/Min should appear above/below the line. max_outliers controls the number of outliers
    allowed to appear on the other side of the line before considering the line ended and a new line should be started.
    :return: The list of segmented bars, each of which can be used to find a single Mean-line
    """
    def _is_max(i):
        return i in local_max_min["max"]
    def _is_min(i):
        return i in local_max_min["min"]
    ret = []
    start, end = 0, len(bars)



if __name__ == '__main__':
    local_max_min=local_max_min([1, 3, 5, 7, 2, 4, 6, 5, -1, 1, 2, 3, 4, 5], 3)
    print(local_max_min)
    # {'max': [3], 'min': [4, 8], 'all': [3, 4, 8]}
