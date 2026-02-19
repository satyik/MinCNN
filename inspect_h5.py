import h5py

try:
    with h5py.File("camelyonpatch_level_2_split_train_x.h5", "r") as f:
        print("Keys:", list(f.keys()))
        if 'x' in f:
            print("Shape of x:", f['x'].shape)
        if 'y' in f:
            print("Shape of y:", f['y'].shape)
except Exception as e:
    print(e)
