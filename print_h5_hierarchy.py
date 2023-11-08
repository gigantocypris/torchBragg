import h5py

def print_hdf5_hierarchy(name, obj, indent='', last=True):
    if isinstance(obj, h5py.File) or isinstance(obj, h5py.Group):
        print(indent + ('└─ ' if last else '├─ ') + name)
        indent += '   ' if last else '│  '
        for key, item in obj.items():
            print_hdf5_hierarchy(key, item, indent, key == list(obj.keys())[-1])
    elif isinstance(obj, h5py.Dataset):
        print(indent + ('└─ ' if last else '├─ ') + name)

# Open the HDF5 file
file = h5py.File('image_rank_00000.h5', 'r')

# Start traversing the file from the root
print_hdf5_hierarchy('/', file)

# Close the file when you're done
file.close()
