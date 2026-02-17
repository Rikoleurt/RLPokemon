import json, socket
# TCP connection between Python & Java
def main():
    host = 'localhost'
    port = 5001
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            print("Connected to server")
            f = s.makefile('r', encoding='utf-8')
            while True:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    jsonObj = json.loads(line)
                    n_nb_self = jsonObj['npc']['pokemonNb']
                    p_nb_self = jsonObj['player']['pokemonNb']
                    print(n_nb_self, p_nb_self)
                    print("Data received :", jsonObj)
                    if n_nb_self == 0 or p_nb_self == 0:
                        print("npc or player defeated")
                        break
    except ConnectionRefusedError:
        print("Impossible to connect to server : Server unavailable")
    except KeyboardInterrupt:
        print("Closing python client")

if __name__ == "__main__":
    main()
