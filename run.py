import subprocess
import random
import time

num_clients = 4
server_port = 50080
server_ip = "127.0.0.1"
local_epochs = 2
active_mode = "adptpolu"
dataset = "PeMSD7"
mode = "FED"

rand_ports = {server_port: 1}

print(f"server ip: {server_ip}")
print(f"server port: {server_port}")

server_command = [
    "python", "server.py",
    "-n", str(num_clients),
    "-p", str(server_port),
    "-i", server_ip
]
subprocess.Popen(server_command)

time.sleep(0.01)

for i in range(1, num_clients + 1):
    client_port = random.randint(20000, 60000)

    while client_port in rand_ports:
        client_port = random.randint(20000, 60000)
    rand_ports[client_port] = 1

    print(f"client {i} port: {client_port}")

    client_command = [
        "python", "client.py", dataset, mode,
        "--cid", str(i),
        "-sip", server_ip,
        "-sp", str(server_port),
        "-cp", str(client_port),
        "--device", "cuda:0",
        "--num_clients", str(num_clients),
        "--divide", "metis",
        "--fedavg",
        "--active_mode", active_mode,
        "--act_k", "2",
        "--local_epochs", str(local_epochs)
    ]

    subprocess.Popen(client_command)
