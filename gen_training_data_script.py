import subprocess


def main():
    als = [10, 30, 100, 300, 2070]
    for al in als:
        output_dir = f"data/2070HH_al{al}_Wh"
        traffic_df_filename = f"data/2070hh_Wh_al{al}.h5"

        arguments = [f'--output_dir={output_dir}',
                     f'--traffic_df_filename={traffic_df_filename}']

        command = ['python', 'generate_training_data.py'] + arguments
        subprocess.run(command)


if __name__ == "__main__":
    main()
