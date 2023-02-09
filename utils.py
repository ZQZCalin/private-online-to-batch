import json
import matplotlib.pyplot as plt


def save_json(data, fname):
    '''
    Save data as JSON.
    '''
    with open(fname, 'w') as f:
        json.dump(data, f)


def load_json(fname):
    '''
    Load JSON file.
    '''
    with open(fname, 'r') as f:
        stats = json.load(f)
    return stats 


def dict_append(dict, **kwargs):
    '''
    Append values to dict[key] (which is assumed to be a list).
    '''
    for key, value in kwargs.items():
        assert type(dict[key]) == list, f'dict["{key}"] should be a list'
        dict[key].append(value)


def compare_stats(tuple1, *args):
    '''
    Compare training stats of 2 models.
    '''
    fig, axes = plt.subplots(4, 3)
    fig.suptitle('Training progress w.r.t. different time measures')

    x_measures = ['epochs', 'total_time', 'effective_time']
    y_measures = ['train_loss', 'train_acc', 'test_loss', 'test_acc']

    stats1, name1 = tuple1

    for i, x_measure in enumerate(x_measures):
        for j, y_measure in enumerate(y_measures):
            x1 = stats1[x_measure]
            y1 = stats1[y_measure]
            ax = axes[j][i]

            ax.plot(x1, y1, label=name1)
            for stats, name in args:
                x2 = stats[x_measure]
                y2 = stats[y_measure]
                ax.plot(x2, y2, label=name)

            if i == 0:
                ax.set_ylabel(y_measure)
            if j == len(y_measures)-1:
                ax.set_xlabel(x_measure)
            if y_measure in ['train_loss', 'test_loss']:
                ax.set_ylim(0, 10)

    for ax in fig.get_axes():
        ax.label_outer()
        if args:
            ax.legend()

    plt.show()


# testing code
if __name__ == "__main__":
    test_dict = {
        'attr1': [1,2,3],
        'attr2': 'hello world'
    }
    # save_json(test_dict, 'test.json')
    # print(load_json('test.json'))

    dict_append(test_dict, attr1=5)
    print(test_dict)