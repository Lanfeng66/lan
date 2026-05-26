import subprocess

ports = {8000: "Backend", 3000: "Frontend"}

print("Stopping DocMind...")
for port, name in ports.items():
    try:
        result = subprocess.run(
            f'netstat -ano | findstr ":{port} "',
            shell=True, capture_output=True, text=True
        )
        pids = set()
        for line in result.stdout.strip().split('\n'):
            parts = line.strip().split()
            if parts:
                try:
                    pid = int(parts[-1])
                    if pid > 0:
                        pids.add(pid)
                except ValueError:
                    pass
        for pid in pids:
            subprocess.run(['taskkill', '/pid', str(pid), '/f'],
                           capture_output=True)
            print(f"  {name} (:{port}) PID {pid} stopped")
        if not pids:
            print(f"  {name} (:{port}) not found")
    except Exception as e:
        print(f"  {name} (:{port}) error: {e}")

print("Done.")
input("Press Enter to exit")
