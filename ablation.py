strategies = ['uncertainty', 'bald']
filters = ['knn']
balanced = ['True', 'False']
types = ['_least_confident','_margin_sampling','_entropy_sampling']
batch_size = [25,50,75,100,125,150]
ratio = [2,3,4,5]

def experiment_1():
    prep = True
    rndom = 'python main.py \'strategy=random\' '
    uncert = 'python main.py \'strategy=uncertainty\' \'strategy.type=_entropy_sampling\' '

    for command in [rndom,uncert]:
        for size in batch_size:
            line =  command + f'\'balance=False\' '
            line += f'\'select_batch_size={size}\' '
            if prep:
                line += f'\'preprocess=True\' '
                line += f'\'split=True\' '
                prep = False
            print(line)

    for r in ratio:
        for size in batch_size:
            line =  uncert + f'\'balance=True\' '
            line += f'\'select_batch_size={size}\' '
            line += f'\'pre_batch_size={int(size * r)}\' '
            print(line)

def experiment_2():
    # rndom = 'python main.py \'balance=False\' \'strategy=random\' \'preprocess=True\' \'split=True\' '
    # print(rndom)
    for s in strategies:
        line = 'python main.py '
        line += f'\'balance=False\' '
        line += f'\'strategy={s}\' '
        if s == 'uncertainty':
            for t in types:
                new_line = line + f'\'strategy.type={t}\' '
                print(new_line)
        else:
            print(line)

    for s in strategies:
        line = 'python main.py '
        line += f'\'balance=True\' '
        line += f'\'strategy={s}\' '

        if s == 'uncertainty':
            for t in types:
                new_line = line + f'\'strategy.type={t}\' '
                print(new_line)
        else:
            print(line)

if __name__ == '__main__':
    # experiment_1()
    experiment_2()


