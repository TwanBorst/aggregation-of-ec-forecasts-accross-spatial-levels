import subprocess

def main():
    als = [10, 30, 100, 300, 2070]
    # als = [2070]

    # exp1 nonAda GNN
    epochs = 150
    for al in als:
        data_path = f'data/2070HH_al{al}'
        save_path = f'./garage/al{al}/exp4_nonAdaGNN/2070HH_al{al}'
        num_nodes = 2070 // al
        arguments = ['--adjtype=doubletransition', f'--data={data_path}', f'--save={save_path}',
                     f'--num_nodes={num_nodes}', '--in_dim=2', f'--adjdata=data/graph/adj_mat_al{al}.pkl',
                     f'--ag_level={al}', f'--epochs={epochs}']
        command = ['python', 'train.py'] + arguments
        subprocess.run(command)

    # # exp2 adaGNN
    # for al in als:
    #     data_path = f'data/2070HH_al{al}'
    #     save_path = f'./garage/al{al}/exp5_adaGNN/2070HH_al{al}'
    #     num_nodes = 2070 // al
    #     arguments = ['--gcn_bool', '--adjtype=doubletransition', f'--data={data_path}', f'--save={save_path}',
    #                  f'--num_nodes={num_nodes}', '--in_dim=2', '--randomadj', '--addaptadj',
    #                  f'--adjdata=data/graph/adj_mat_al{al}.pkl', f'--ag_level={al}', f'--epochs={epochs}']
    #     command = ['python', 'train.py'] + arguments
    #     subprocess.run(command)


if __name__ == "__main__":
    main()
