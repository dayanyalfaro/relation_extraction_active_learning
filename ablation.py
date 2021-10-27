strategies = ['random', 'uncertainty', 'bald', 'kmeans']
filters = ['knn', 'prob']
balanced = ['True', 'False']
types = ['least_confident','margin_sampling','entropy_sampling']

if __name__ == '__main__':
    for s in strategies:
        line = 'python main.py '
        line += f'\'balanced=False\' '
        line += f'\'strategy={s}\' '
        if s == 'uncertainty':
            for t in types:
                new_line = line + f'\'strategy.type={t}\' '
                print(new_line + '\n')
        else:
            print(line + '\n')

    for s in strategies:
        for f in filters:
            line = 'python main.py '
            line += f'\'balanced=True\' '
            line += f'\'strategy={s}\' '
            line += f'\'class_strategy={f}\' '
            if s == 'uncertainty':
                for t in types:
                    new_line = line + f'\'strategy.type={t}\' '
                    print(line + '\n')
            else:
                print(line + '\n')


