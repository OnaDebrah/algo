import time

print("Starting import test...")
start = time.time()

try:
    print(f"Successfully imported app.main.app in {time.time() - start:.2f}s")
except Exception as e:
    print(f"Failed to import: {e}")
