"""
libtbx.python $MODULES/torchBragg/create_fp_fdp_dat_file.py

Terminal output:
New .dat file 'Mn_fp_0_fdp_-.dat' created.
New .dat file 'Mn_fp_1_fdp_-.dat' created.
New .dat file 'Mn_fp_-_fdp_0.dat' created.
New .dat file 'Mn_fp_-_fdp_1.dat' created.
New .dat file 'Mn_fp_0_fdp_0.dat' created.
"""

import os
from LS49 import ls49_big_data

def full_path(filename):
  return os.path.join(ls49_big_data, filename)

def create_new_dat(source_filename, output_filename, change_col, new_value):
    # Open the source file for reading
    # source_filename = 'source.dat'
    with open(source_filename, 'r') as source_file:
        # Read data from the source file
        lines = source_file.readlines()

    # Extract the first column from the source data
    col_0 = [line.split()[0] for line in lines]
    col_1 = [line.split()[1] for line in lines]
    col_2 = [line.split()[2] for line in lines]

    # Create a new .dat file with the first column and two columns of zeros
    # output_filename = 'output.dat'
    with open(output_filename, 'w') as output_file:
        # Write the data to the new file
        for value_0, value_1, value_2 in zip(col_0, col_1, col_2):
            if change_col == 1:
                output_file.write(f"{value_0} {new_value} {value_2}\n")
            elif change_col == 2:
                output_file.write(f"{value_0} {value_1} {new_value}\n")
            else:
                raise ValueError("change_col must be 1 or 2")

    print(f"New .dat file '{output_filename}' created.")

if __name__ =="__main__":
    Mn_oxidized_model = full_path("data_sherrell/MnO2_spliced.dat")

    output_filename = "Mn_fp_0_fdp_-.dat"
    change_col = 1
    new_value = 0
    create_new_dat(Mn_oxidized_model, output_filename, change_col, new_value)

    output_filename = "Mn_fp_1_fdp_-.dat"
    change_col = 1
    new_value = 1
    create_new_dat(Mn_oxidized_model, output_filename, change_col, new_value)

    output_filename = "Mn_fp_-_fdp_0.dat"
    change_col = 2
    new_value = 0
    create_new_dat(Mn_oxidized_model, output_filename, change_col, new_value)

    output_filename = "Mn_fp_-_fdp_1.dat"
    change_col = 2
    new_value = 1
    create_new_dat(Mn_oxidized_model, output_filename, change_col, new_value)

    input_filename = "Mn_fp_0_fdp_-.dat"
    output_filename = "Mn_fp_0_fdp_0.dat"
    change_col = 2
    new_value = 0
    create_new_dat(input_filename, output_filename, change_col, new_value)