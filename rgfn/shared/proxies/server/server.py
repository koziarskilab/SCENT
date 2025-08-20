import json
import logging
import socket
import threading
import time

from rgfn.shared.proxies.server_proxy import ServerProxy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ProxyServer")


class ProxyServer:
    def __init__(
        self,
        proxy: ServerProxy,
        host: str = "0.0.0.0",
        port: int = 5555,
    ):
        """Initialize the proxy server.

        Args:
            host: IP address to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.proxy = proxy
        self.server_socket = None
        self.running = False
        self.clients = []
        self.stats = {"queries_processed": 0, "total_molecules": 0, "start_time": None}

    def start(self):
        """Start the server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allow reuse of address
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)  # Allow up to 10 pending connections

        self.running = True
        self.stats["start_time"] = time.time()

        logger.info(f"Server started on {self.host}:{self.port}")
        logger.info(f"Proxy: {self.proxy}")

        # Start a thread to log stats periodically
        stats_thread = threading.Thread(target=self._log_stats)
        stats_thread.daemon = True
        stats_thread.start()

        try:
            while self.running:
                client_socket, addr = self.server_socket.accept()
                logger.info(f"Connection from {addr}")

                client_thread = threading.Thread(
                    target=self._handle_client, args=(client_socket, addr)
                )
                client_thread.daemon = True
                client_thread.start()
                self.clients.append((client_socket, addr, client_thread))

        except KeyboardInterrupt:
            self.stop()

    def _log_stats(self):
        """Log server statistics periodically."""
        while self.running:
            if self.stats["start_time"]:
                uptime = time.time() - self.stats["start_time"]
                logger.info(
                    f"Stats: Uptime={uptime:.2f}s, "
                    f"Queries={self.stats['queries_processed']}, "
                    f"Molecules={self.stats['total_molecules']}"
                )
            time.sleep(60)  # Log every minute

    def _handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle client connection.

        Args:
            client_socket: Socket for communicating with the client
            address: Client address
        """
        try:
            while self.running:
                # Receive data size first (4 bytes for message length)
                size_data = client_socket.recv(4)
                if not size_data:
                    break

                msg_size = int.from_bytes(size_data, byteorder="big")

                # Receive the actual message
                chunks = []
                bytes_received = 0
                while bytes_received < msg_size:
                    chunk = client_socket.recv(min(4096, msg_size - bytes_received))
                    if not chunk:
                        raise RuntimeError("Socket connection broken")
                    chunks.append(chunk)
                    bytes_received += len(chunk)

                data = b"".join(chunks)

                try:
                    # Decode the message
                    message = json.loads(data.decode("utf-8"))
                    smiles_list = message.get("smiles", [])
                    job_id = message.get("job_id", "unknown")
                    batch_id = message.get("batch_id", 0)

                    logger.info(
                        f"Processing job {job_id}, batch {batch_id} with {len(smiles_list)} molecules"
                    )

                    # Process the request with the proxy
                    start_time = time.time()
                    results = self.proxy(smiles_list)
                    elapsed = time.time() - start_time

                    # Update stats
                    self.stats["queries_processed"] += 1
                    self.stats["total_molecules"] += len(smiles_list)

                    # Send back the results
                    response = {
                        "job_id": job_id,
                        "batch_id": batch_id,
                        "results": results,
                        "processing_time": elapsed,
                        "status": "success",
                    }

                    response_bytes = json.dumps(response).encode("utf-8")
                    # Send the size of the message first
                    size_bytes = len(response_bytes).to_bytes(4, byteorder="big")
                    client_socket.sendall(size_bytes + response_bytes)

                    logger.info(
                        f"Completed job {job_id}, batch {batch_id} in {elapsed:.4f}s "
                        f"({len(smiles_list)/elapsed:.2f} molecules/s)"
                    )

                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON message")
                    response = {"status": "error", "message": "Invalid JSON format"}
                    response_bytes = json.dumps(response).encode("utf-8")
                    size_bytes = len(response_bytes).to_bytes(4, byteorder="big")
                    client_socket.sendall(size_bytes + response_bytes)

                except Exception as e:
                    logger.error(f"Error processing request: {str(e)}")
                    response = {"status": "error", "message": str(e)}
                    response_bytes = json.dumps(response).encode("utf-8")
                    size_bytes = len(response_bytes).to_bytes(4, byteorder="big")
                    client_socket.sendall(size_bytes + response_bytes)

        except ConnectionResetError:
            logger.info(f"Client {address} disconnected")

        except Exception as e:
            logger.error(f"Error handling client {address}: {str(e)}")

        finally:
            client_socket.close()
            # Remove client from list
            self.clients = [(s, a, t) for s, a, t in self.clients if s != client_socket]
            logger.info(f"Connection closed for {address}")

    def stop(self):
        """Stop the server."""
        logger.info("Stopping server...")
        self.running = False

        # Close all client connections
        for client_socket, _, _ in self.clients:
            try:
                client_socket.close()
            except:
                pass

        # Close server socket
        if self.server_socket:
            self.server_socket.close()

        logger.info("Server stopped")


if __name__ == "__main__":
    import argparse

    import gin

    parser = argparse.ArgumentParser(description="Start a Proxy server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5555, help="Port to listen on")
    parser.add_argument(
        "--proxy-config",
        type=str,
        help="Path to gin config file",
        default="configs/proxies/target/qed.gin",
    )

    args = parser.parse_args()
    cfg = args.proxy_config

    # The gin config should define an 'train_proxy' object
    gin.parse_config_files_and_bindings([cfg], bindings=[])
    proxy = gin.get_configurable("proxy/gin.singleton")()

    if proxy is None:
        raise ValueError("No proxy configured in gin file. Please check your configuration.")

    # Check that the proxy implements the ServerProxy interface
    if not isinstance(proxy, ServerProxy):
        raise TypeError("Proxy must be an instance of ServerProxy")

    logger.info(f"Using proxy: {proxy}")
    server = ProxyServer(host=args.host, port=args.port, proxy=proxy)
    server.start()
