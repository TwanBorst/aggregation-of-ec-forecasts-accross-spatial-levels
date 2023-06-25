import dask
from dask.distributed import Client
from constants import *
from experiments import epochs_100_without_hyperparameters, epochs_25_with_hyperparameters, epochs_25_without_hyperparameters, get_std_ec, preprocess_data, save_expected_model_output, get_mean_ec


if __name__ == '__main__':
    dask.config.set(temporary_directory=DASK_TMP_DIR)

    client = Client(n_workers=4, threads_per_worker=2, memory_limit="10GB")
    print(client)
 
    preprocess_data()

    epochs_25_without_hyperparameters()
    
    epochs_100_without_hyperparameters()
    
    epochs_25_with_hyperparameters()
    
    save_expected_model_output()
    
    # Print mean energy consumption over time per spatial level
    mean_ec_appliance, mean_ec_household, mean_ec_community, mean_ec_city = get_mean_ec()
    print(f"Mean energy consumption appliance: {mean_ec_appliance}\n", f"Mean energy consumption household: {mean_ec_household}\n",
          f"Mean energy consumption community: {mean_ec_community}\n", f"Mean energy consumption city: {mean_ec_city}")
    
    # Print standard deviation of energy consumption over time per spatial level
    std_ec_appliance, std_ec_household, std_ec_community, std_ec_city = get_std_ec()
    print(f"STD of energy consumption appliance: {std_ec_appliance}\n", f"STD of energy consumption household: {std_ec_household}\n",
          f"STD of energy consumption community: {std_ec_community}\n", f"STD of energy consumption city: {std_ec_city}")
    
    

    