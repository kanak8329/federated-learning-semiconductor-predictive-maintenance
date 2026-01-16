import glob

files = glob.glob("data/windows/clients/*.npy")
print("Found files:", len(files))
for f in files:
    print(f)
