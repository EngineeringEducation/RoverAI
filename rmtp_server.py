import socket
import struct

class SimpleRTMPServer:
    def __init__(self, host='0.0.0.0', port=1935, output_file='stream.flv'):
        self.host = host
        self.port = port
        self.output_file = output_file
        self.server_socket = None

    def start(self):
        # Create a TCP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)  # Allow only one client at a time
        print(f"RTMP server listening on {self.host}:{self.port}")

        while True:
            client_socket, client_address = self.server_socket.accept()
            print(f"Connection from {client_address}")
            self.handle_client(client_socket)

    def handle_client(self, client_socket):
        try:
            # Perform RTMP handshake
            self.handshake(client_socket)
            
            # Start streaming data from the client
            self.stream_data(client_socket)
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()

    def handshake(self, client_socket):
        # Receive C0 and C1 from the client
        c0_c1 = client_socket.recv(1537)

        # Send S0 and S1 back to the client
        s0_s1 = b'\x03' + c0_c1[1:]  # S0 is 0x03, S1 is same as C1
        client_socket.send(s0_s1)

        # Send S2 back to the client (same as C1)
        client_socket.send(c0_c1[1:])

        # Receive C2 from the client
        client_socket.recv(1536)

        print("RTMP handshake completed")

    def stream_data(self, client_socket):
        with open(self.output_file, 'wb') as f:
            while True:
                # RTMP messages have a basic header followed by a payload
                header = client_socket.recv(12)
                if not header:
                    break
                
                # Extract the message type and payload length
                msg_type = header[0]
                msg_length = struct.unpack('>I', b'\x00' + header[1:4])[0]

                # Read the payload
                payload = client_socket.recv(msg_length)
                if not payload:
                    break

                print(f"Received RTMP message type {msg_type} with length {msg_length}")
                ## extract only the video data
                if msg_type == 9:
                    print(f"Writing {len(payload)} bytes to {self.output_file}")
                    
                    # Write the payload to the output file
                    f.write(header + payload)

                    # Overwrite the file from the start for the next chunk of data
                    f.seek(0)
                else:
                    print(f"Skipping non-video message")
                

if __name__ == '__main__':
    server = SimpleRTMPServer()
    server.start()
