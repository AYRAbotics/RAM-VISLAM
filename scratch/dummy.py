import time

class SurfelMap:
    def __init__(self):
        self.active_n = 42
    def export_ply(self, path):
        with open(path, "w") as f:
            f.write(f"surfels: {self.active_n}\n")
        print(f"Exported to {path}")

surfel_map = SurfelMap()

print("Dummy running...")
while True:
    time.sleep(1)
