strategies = ['random', 'uncertainty', 'bald']
filters = ['knn']
balanced = ['True', 'False']
types = ['_least_confident','_margin_sampling','_entropy_sampling']
batch_size = [50,100,150,200]
ratio = [1.5,2,2.5,3,3.5,4,4.5,5]

def experiment_1():
    prep = True
    rndom = 'python main.py \'strategy=random\''
    uncert = 'python main.py \'strategy=uncertainty\' \'strategy.type=_entropy_sampling\' '

    for command in [rndom,uncert]:
        for size in batch_size:
            line =  command + f'\'balance=False\' '
            line += f'\'select_batch_size={size}\' '
            if prep:
                line += f'\'preprocess=True\' '
                prep = False
            print(line)

    for r in ratio:
        for size in batch_size:
            line =  uncert + f'\'balance=True\' '
            line += f'\'select_batch_size={size}\' '
            line += f'\'pre_batch_size={size * r}\' '
            print(line)

if __name__ == '__main__':
    # for s in strategies:
    #     line = 'python main.py '
    #     line += f'\'balance=False\' '
    #     line += f'\'strategy={s}\' '
    #     if s == 'uncertainty':
    #         for t in types:
    #             new_line = line + f'\'strategy.type={t}\' '
    #             print(new_line + '\n')
    #     else:
    #         print(line + '\n')

    # for s in strategies:
    #     for f in filters:
    #         line = 'python main.py '
    #         line += f'\'balance=True\' '
    #         line += f'\'strategy={s}\' '
    #         line += f'\'class_strategy={f}\' '
    #         if s == 'uncertainty':
    #             for t in types:
    #                 new_line = line + f'\'strategy.type={t}\' '
    #                 print(line + '\n')
    #         else:
    #             print(line + '\n')
    experiment_1()


