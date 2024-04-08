import socket
import argparse

def find_free_port(starting_port, interval=1):
    for port in range(starting_port, 65536, interval):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError as e:
                if e.errno == 98:  # errno 98 means the port is already in use
                    continue
                raise


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--starting_port', type=int, default=29500, required=False)
    argparser.add_argument('--interval', type=int, default=1, required=False)
    args = argparser.parse_args()
    print(find_free_port(args.starting_port, args.interval))