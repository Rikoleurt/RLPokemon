import socket

def main():
    host = 'localhost'
    port = 5000

    try:
        # Create a TCP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Connect to the Java server
            s.connect((host, port))
            print("Connected to server")

            # Passive loop waiting for messages from the server
            while True:
                try:
                    # Receive data from the server
                    data = s.recv(4096)
                    if not data:
                        print("Disconnected from server")
                        break
                    print("Data received :", data.decode('utf-8'))

                except ConnectionResetError:
                    # Connection was forcibly closed by the Java server
                    print("Connection reset by server")
                    break

    except ConnectionRefusedError:
        # Server is not running or unavailable
        print("Impossible to connect to server : Server unavailable")
    except KeyboardInterrupt:
        # Graceful shutdown when Ctrl+C is pressed
        print("Closing python client")


if __name__ == "__main__":
    main()
