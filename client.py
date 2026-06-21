import json
import socket


class Client:
    def __init__(self, host: str = "localhost", port: int = 5001):
        self.host = host
        self.port = port
        self.sock = None
        self.stream = None

    def connect(self):
        if self.sock is not None:
            return

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.stream = self.sock.makefile("r", encoding="utf-8")

    def send_command(self, command: str):
        self.connect()
        self.sock.sendall((command.strip() + "\n").encode("utf-8"))

    def reset(self, seed: int | None = None):
        if seed is None:
            self.send_command("RESET")
        else:
            self.send_command(f"RESET {int(seed)}")

    def send_action(self, action_id: int):
        self.send_command(str(int(action_id)))

    def receive_message(self) -> dict:
        self.connect()
        while True:
            line = self.stream.readline()
            if not line:
                raise ConnectionError("Server closed connection")
            line = line.strip()
            if line:
                return json.loads(line)

    def close(self):
        try:
            if self.sock:
                self.send_command("DONE")
        except Exception:
            pass
        finally:
            try:
                if self.stream:
                    self.stream.close()
                if self.sock:
                    self.sock.close()
            finally:
                self.stream = None
                self.sock = None
