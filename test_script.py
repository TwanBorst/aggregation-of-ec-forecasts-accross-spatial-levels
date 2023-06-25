import subprocess
import os

def main():
    als = [10, 30, 100, 300, 2070]
    # als = [2070]

    # exp1 nonAda GNN
    for al in als:
        data_path = f'data/2070HH_al{al}'
        directory = f'./garage/al{al}/exp4_nonAdaGNN/'
        pattern = f'2070HH_al{al}_exp1_best_'
        path = None
        for filename in os.listdir(directory):
            if filename.startswith(pattern):
                path = filename
        checkpoint = f'./garage/al{al}/exp4_nonAdaGNN/{path}'
        save_path = f'./garage/al{al}/exp4_nonAdaGNN/wave_al{al}'
        num_nodes = 2070 // al
        arguments = ['--adjtype=doubletransition', f'--data={data_path}', f'--save={save_path}',
                     f'--num_nodes={num_nodes}', '--in_dim=2', '--plotheatmap=False', '--model=nonAda_GNN',
                     f'--adjdata=data/graph/adj_mat_al{al}.pkl', f'--ag_level={al}', f'--checkpoint={checkpoint}']
        command = ['python', 'test.py'] + arguments
        subprocess.run(command)

    # exp2 adaGNN
    # for al in als:
    #     data_path = f'data/2070HH_al{al}'
    #     best_id = 50
    #     best_loss = 0.0264
    #     path = f'2070HH_al{al}_epoch_{best_id}_{best_loss}.pth'
    #     save = f'./garage/al{al}/exp5_adaGNN/2070HH_al{al}'
    #     save_path = f'./garage/al{al}/exp5_adaGNN/wave_al{al}'
    #     num_nodes = 2070 // al
    #     arguments = ['--gcn_bool', '--adjtype=doubletransition', f'--data={data_path}', f'--save={save}',
    #                  f'--num_nodes={num_nodes}', '--in_dim=2', '--randomadj', '--addaptadj',
    #                  f'--adjdata=data/graph/adj_mat_al{al}.pkl', f'--best_id={best_id}', f'--ag_level={al}',
    #                  f'--best_loss={best_loss}', f'--save_path={save_path}', '--model=AdaGNN']
    #     command = ['python', 'test2.py'] + arguments
    #     subprocess.run(command)

if __name__ == "__main__":
    main()
