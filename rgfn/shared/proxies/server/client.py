import argparse
import concurrent
import concurrent.futures
import json
import logging
import random
import socket
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ProxyClient")


class ProxyServerConnection:
    """Represents a connection to a remote Proxy server."""

    def __init__(self, server_id: int, hostname: str, port: int, status: str = "unknown"):
        """Initialize server connection info.

        Args:
            server_id: Unique ID for this server
            hostname: Server hostname or IP
            port: Server port
            status: Initial server status
        """
        self.id = server_id
        # TODO add a parameter of `hostname_mapping`. Make it configurable by `gin`.
        self.hostname = f"{hostname}.ccm.sickkids.ca"  # hostname
        self.port = port
        self.status = status
        self.socket = None
        self.connected = False
        self.last_used = 0
        self.active_queries = 0
        self.stats = {
            "queries_sent": 0,
            "molecules_processed": 0,
            "errors": 0,
            "total_processing_time": 0,
            "response_times": [],  # List of response times for calculating average
        }

    def connect(self) -> bool:
        """Connect to the server.

        Returns:
            bool: True if connection successful
        """
        if self.connected:
            return True

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)  # 10 second timeout for connection
            self.socket.connect((self.hostname, self.port))
            self.connected = True
            self.status = "connected"
            logger.info(f"Connected to server {self.id} at {self.hostname}:{self.port}")
            return True
        except Exception as e:
            self.status = "error"
            logger.error(
                f"Failed to connect to server {self.id} at {self.hostname}:{self.port}: {str(e)}"
            )
            if self.socket:
                self.socket.close()
                self.socket = None
            return False

    def disconnect(self):
        """Disconnect from the server."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        self.connected = False
        self.status = "disconnected"

    def is_available(self) -> bool:
        """Check if server is available for queries.

        Returns:
            bool: True if server is available
        """
        return self.connected and self.status == "connected"

    def get_score(self) -> float:
        """Calculate server score for load balancing.

        A lower score means the server is more preferable.

        Returns:
            float: Score value
        """
        # Include active queries in the scoring
        time_factor = max(0, 5 - (time.time() - self.last_used))
        query_factor = (self.stats["queries_sent"] + self.active_queries) * 0.1
        active_query_penalty = self.active_queries * 2.0  # Penalize servers with active queries

        # Add randomness to distribute initial load
        random_factor = random.random() * 0.5

        return time_factor + query_factor + active_query_penalty + random_factor

    def query(self, smiles_list: List[str], job_id: str, batch_id: int) -> Dict[str, Any]:
        """Send a query to the server.

        Args:
            smiles_list: List of SMILES strings to process
            job_id: Unique job ID
            batch_id: Batch identifier within the job

        Returns:
            Dict: Server response
        """
        # Create a new socket connection for each query to prevent concurrent access issues
        sock = None
        try:
            # Start timing the request
            start_time = time.time()

            # Create a fresh socket for this specific query
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)  # 10 second timeout for connection
            sock.connect((self.hostname, self.port))

            # Prepare the request
            request = {"job_id": job_id, "batch_id": batch_id, "smiles": smiles_list}

            # Encode the request
            request_bytes = json.dumps(request).encode("utf-8")
            size_bytes = len(request_bytes).to_bytes(4, byteorder="big")

            # Send the request
            sock.sendall(size_bytes)
            sock.sendall(request_bytes)

            # Receive the response size
            sock.settimeout(10800)  # 3 hour timeout for response
            size_data = sock.recv(4)
            if not size_data or len(size_data) != 4:
                raise ConnectionError(
                    f"Connection closed by server (received {len(size_data)} bytes)"
                )

            msg_size = int.from_bytes(size_data, byteorder="big")
            if msg_size <= 0 or msg_size > 100 * 1024 * 1024:  # Sanity check: max 100MB
                raise ValueError(f"Invalid message size: {msg_size} bytes")

            # Receive the response data
            chunks = []
            bytes_received = 0
            while bytes_received < msg_size:
                chunk = sock.recv(min(4096, msg_size - bytes_received))
                if not chunk:
                    raise ConnectionError(
                        f"Connection broken during response (received {bytes_received}/{msg_size} bytes)"
                    )
                chunks.append(chunk)
                bytes_received += len(chunk)

            response_data = b"".join(chunks)

            # Try to decode the response
            try:
                response = json.loads(response_data.decode("utf-8"))
            except json.JSONDecodeError as e:
                # Log a preview of the raw data for debugging
                preview = response_data[:200].decode("utf-8", errors="replace")
                raise ValueError(f"Invalid JSON response: {str(e)}. Preview: {preview}")

            # Update server stats (thread-safe via self.stats)
            response_time = time.time() - start_time
            self.last_used = time.time()
            self.stats["queries_sent"] += 1
            self.stats["molecules_processed"] += len(smiles_list)
            self.stats["total_processing_time"] += response.get("processing_time", 0)
            self.stats["response_times"].append(response_time)

            # Keep only last 100 response times
            if len(self.stats["response_times"]) > 100:
                self.stats["response_times"] = self.stats["response_times"][-100:]

            return response

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error querying server {self.id}: {str(e)}")

            return {
                "status": "error",
                "message": str(e),
                "job_id": job_id,
                "batch_id": batch_id,
                "results": [float("nan")] * len(smiles_list),
            }
        finally:
            # Always close the socket when done, even if there's an exception
            if sock:
                try:
                    sock.close()
                except:
                    pass

    def __str__(self):
        avg_response_time = 0
        if self.stats["response_times"]:
            avg_response_time = sum(self.stats["response_times"]) / len(
                self.stats["response_times"]
            )

        return (
            f"Server {self.id} ({self.hostname}:{self.port}) - "
            f"Status: {self.status}, "
            f"Queries: {self.stats['queries_sent']}, "
            f"Molecules: {self.stats['molecules_processed']}, "
            f"Avg Response: {avg_response_time:.4f}s"
        )

    def reserve(self) -> bool:
        """Reserve this server for a query. Thread-safe."""
        self.active_queries += 1
        return True

    def release(self):
        """Release this server after a query. Thread-safe."""
        if self.active_queries > 0:
            self.active_queries -= 1


class ProxyClient:
    """Client for distributing Proxy queries across multiple servers."""

    def __init__(
        self,
        registry_file: str,
        batch_size: int = 100,
        max_retries: int = 3,
    ):
        """Initialize Proxy client.

        Args:
            registry_file: Path to the server registry JSON file
            batch_size: Default batch size for splitting queries
            max_retries: Maximum number of retries for failed queries
        """
        self.registry_file = registry_file
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.servers = []
        self.server_lock = threading.Lock()
        self.job_results = {}
        self.job_lock = threading.Lock()

        if not self._are_servers_running():
            logger.warning("No active servers found in registry, not using client")
            return

        # Load servers from registry
        self._load_servers()

        # Start server monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_servers)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _are_servers_running(self) -> bool:
        try:
            with open(self.registry_file, "r") as f:
                registry = json.load(f)

            active_servers = 0
            for server in registry:
                job_id = server.get("job_id")
                status = server.get("status")
                if not job_id or status == "terminated":
                    continue
                try:
                    result = subprocess.run(
                        ["squeue", "-j", job_id, "-o", "%T", "-h"],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        status = result.stdout.strip()
                        if status and status in ["RUNNING", "PENDING"]:
                            active_servers += 1
                except Exception as e:
                    logger.error(f"Error checking status for job {job_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load server registry: {str(e)}")
            return False
        return active_servers > 0

    def _load_servers(self):
        """Load server information from registry file."""
        try:
            with open(self.registry_file, "r") as f:
                registry = json.load(f)

            with self.server_lock:
                # Create server objects
                self.servers: List[ProxyServerConnection] = []
                for server_info in registry:
                    # Use IP address if available, otherwise hostname
                    hostname = server_info.get("ip_address", server_info.get("hostname"))
                    if not hostname or hostname == "(None)":
                        continue

                    server = ProxyServerConnection(
                        server_id=server_info["id"],
                        hostname=hostname,
                        port=server_info["port"],
                        status=server_info.get("status", "unknown"),
                    )
                    self.servers.append(server)

                logger.info(f"Loaded {len(self.servers)} servers from registry")

        except Exception as e:
            logger.error(f"Failed to load server registry: {str(e)}")

    def _monitor_servers(self):
        """Monitor server status periodically."""
        while True:
            try:
                # Reload registry occasionally to pick up new servers
                if random.random() < 0.2:  # 20% chance to reload registry
                    self._load_servers()

                with self.server_lock:
                    # Test connection to each server
                    for server in self.servers:
                        if not server.connected:
                            server.connect()

                # Log server stats
                logger.info("Server status:")
                with self.server_lock:
                    for server in self.servers:
                        logger.info(f"  {server}")

            except Exception as e:
                logger.error(f"Error in server monitor: {str(e)}")

            # Sleep for a while
            time.sleep(60)

    def _select_server(self) -> Optional[ProxyServerConnection]:
        """Select the best server for a query based on load balancing.

        Returns:
            ProxyServer or None: Selected server or None if no servers available
        """
        with self.server_lock:
            # Filter available servers
            available_servers = [s for s in self.servers if s.is_available()]
            if not available_servers:
                logger.warning("No available servers!")
                return None

            # Strategy selection (can be made configurable)
            strategy = "weighted_random"  # Options: "best_score", "round_robin", "weighted_random"

            if strategy == "best_score":
                # Sort by score (lower is better)
                available_servers.sort(key=lambda s: s.get_score())
                return available_servers[0]

            elif strategy == "round_robin":
                # Simple round-robin
                self._last_server_index = getattr(self, "_last_server_index", -1) + 1
                if self._last_server_index >= len(available_servers):
                    self._last_server_index = 0
                return available_servers[self._last_server_index]

            elif strategy == "weighted_random":
                # Calculate weights based on current load including active queries
                weights = []
                for server in available_servers:
                    if server.active_queries == 0:
                        server.reserve()
                        # once identified a inactive server, submit to it.
                        return server
                    # Higher active queries = lower weight
                    active_penalty = 1.0 / (1.0 + server.active_queries)

                    # Historical query load
                    total_queries = sum(s.stats["queries_sent"] for s in available_servers) + len(
                        available_servers
                    )
                    query_weight = 1.0 - (
                        server.stats["queries_sent"] / total_queries if total_queries > 0 else 0
                    )

                    # Response time factor
                    response_weight = 1.0
                    if server.stats["response_times"]:
                        avg_time = sum(server.stats["response_times"]) / len(
                            server.stats["response_times"]
                        )
                        response_weight = 1.0 / (1.0 + avg_time)

                    # Combine all factors
                    weight = 0.4 * active_penalty + 0.3 * query_weight + 0.3 * response_weight
                    weights.append(weight)

                # Normalize weights
                total_weight = sum(weights) if weights else 1.0
                weights = [w / total_weight for w in weights]

                # Weighted random selection

                selected_server = random.choices(available_servers, weights=weights, k=1)[0]

            else:
                # Default to best score
                available_servers.sort(key=lambda s: s.get_score())
                selected_server = available_servers[0]

            # Reserve the selected server
            if selected_server:
                selected_server.reserve()

            return selected_server

    def _process_batch(self, job_id: str, batch_id: int, smiles_batch: List[str]) -> Dict[str, Any]:
        """Process a batch of SMILES strings."""
        # logger.info(f"")

        attempts = 0
        servers_tried = set()

        while attempts <= self.max_retries:
            # Select server
            server = self._select_server()
            if not server:
                logger.error(
                    f"batch {batch_id} with {len(smiles_batch)} molecules is failed due to no available server"
                )
                return {
                    "status": "error",
                    "message": "No available servers",
                    "results": [float("nan")] * len(smiles_batch),
                }
            logger.info(
                f"batch {batch_id} with {len(smiles_batch)} molecules is submitted to server {server.id}"
            )
            # Track which servers we've tried
            servers_tried.add(server.id)

            try:
                # Send request to server
                start_time = time.time()
                response = server.query(smiles_batch, job_id, batch_id)
                elapsed = time.time() - start_time
                # Check if response indicates an error
                if response.get("status") == "error":
                    server.release()
                    attempts += 1
                    logger.warning(
                        f"Batch {batch_id} failed on server {server.id}, attempt {attempts}/{self.max_retries}: {response.get('message')}"
                    )
                    if attempts <= self.max_retries:
                        continue  # Try again with potentially a different server
                else:
                    # Success!
                    server.release()
                    logger.info(
                        f"Completed batch {batch_id} in {elapsed:.4f}s from server {server.id}"
                    )
                    logger.info(
                        f"Server {server.id} returned: status={response.get('status')}, results_count={len(response.get('results', []))}"
                    )
                    return response

            except Exception as e:
                server.release()
                attempts += 1
                logger.error(f"Error processing batch {batch_id} on server {server.id}: {str(e)}")
                if attempts <= self.max_retries:
                    continue  # Try again with potentially a different serve

        # All retries failed
        logger.error(
            f"Batch {batch_id} failed after {attempts} attempts on {len(servers_tried)} different servers"
        )
        return {
            "status": "error",
            "message": f"Failed after {attempts} attempts",
            "results": [float("nan")] * len(smiles_batch),
        }

    def query(self, smiles_list: List[str], _batch_size: Optional[int] = None) -> List[float]:
        """Query Proxy with a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings to process
            _batch_size: Optional custom batch size

        Returns:
            List[float]: Results for each SMILES string
        """
        if not smiles_list:
            return []

        batch_size: int
        batch_size = _batch_size or self.batch_size
        job_id = str(uuid.uuid4())

        # Split into batches
        batches = []
        for i in range(0, len(smiles_list), batch_size):
            batches.append(smiles_list[i : i + batch_size])

        # Create placeholders for results
        results = [None] * len(smiles_list)

        logger.info(
            f"Job {job_id}: Processing {len(smiles_list)} molecules in {len(batches)} batches"
        )

        # Process batches with explicit timeout
        with ThreadPoolExecutor(max_workers=min(len(batches), 10)) as executor:
            futures = {}
            for batch_id, batch in enumerate(batches):
                future = executor.submit(self._process_batch, job_id, batch_id, batch)
                futures[future] = (batch_id, len(batch))

            # Collect results with timeout
            for future in concurrent.futures.as_completed(futures):
                batch_id, batch_size = futures[future]
                try:
                    # Use a shorter timeout for testing
                    response = future.result(timeout=None)

                    # Process the response
                    batch_results = response.get("results", [float("nan")] * batch_size)

                    # Place the results in the correct position
                    start_idx = batch_id * self.batch_size
                    for i, result in enumerate(batch_results):
                        if start_idx + i < len(results):
                            results[start_idx + i] = result

                    if response.get("status") != "success":
                        logger.warning(
                            f"Batch {batch_id} failed: {response.get('message', 'Unknown error')}"
                        )
                    else:
                        logger.info(f"Successfully processed batch {batch_id}")

                except concurrent.futures.TimeoutError:
                    logger.error(f"Batch {batch_id} timed out")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_id}: {str(e)}")

        logger.info(f"Job {job_id} completed with {results.count(None)} missing results")
        return [0.0 if r is None else r for r in results]

    def query_async(
        self, smiles_list: List[str], callback=None, batch_size: Optional[int] = None
    ) -> str:
        """Asynchronously query Proxy with a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings to process
            callback: Optional callback function to call with results when complete
            batch_size: Optional custom batch size

        Returns:
            str: Job ID
        """
        if not smiles_list:
            if callback:
                callback([])
            return ""

        batch_size = batch_size or self.batch_size
        job_id = str(uuid.uuid4())

        # Initialize job results
        with self.job_lock:
            self.job_results[job_id] = {
                "status": "running",
                "total_molecules": len(smiles_list),
                "completed_molecules": 0,
                "results": [None] * len(smiles_list),
                "start_time": time.time(),
                "callback": callback,
            }

        # Start a thread to process the job
        thread = threading.Thread(target=self._process_job, args=(job_id, smiles_list, batch_size))
        thread.daemon = True
        thread.start()

        return job_id

    def _process_job(self, job_id: str, smiles_list: List[str], batch_size: int):
        """Process a job asynchronously.

        Args:
            job_id: Unique job ID
            smiles_list: List of SMILES strings
            batch_size: Batch size
        """
        try:
            # Split into batches
            batches = []
            for i in range(0, len(smiles_list), batch_size):
                batches.append(smiles_list[i : i + batch_size])

            logger.info(
                f"Job {job_id}: Processing {len(smiles_list)} molecules in {len(batches)} batches"
            )

            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=min(len(batches), 10)) as executor:
                futures = []
                for batch_id, batch in enumerate(batches):
                    future = executor.submit(self._process_batch, job_id, batch_id, batch)
                    futures.append((future, batch_id, len(batch)))

                # Collect results
                for future, batch_id, batch_size in futures:
                    try:
                        response = future.result()
                        batch_results = response.get("results", [float("nan")] * batch_size)

                        # Update job results
                        with self.job_lock:
                            if job_id in self.job_results:
                                job_info = self.job_results[job_id]

                                # Place the results in the correct position
                                start_idx = batch_id * self.batch_size
                                for i, result in enumerate(batch_results):
                                    if start_idx + i < len(job_info["results"]):
                                        if job_info["results"][start_idx + i] is None:
                                            job_info["completed_molecules"] += 1
                                        job_info["results"][start_idx + i] = result

                            else:
                                # Job was cancelled
                                logger.warning(f"Job {job_id} was cancelled during processing")
                                return

                    except Exception as e:
                        logger.error(f"Error processing batch {batch_id}: {str(e)}")

            # Job completed
            with self.job_lock:
                if job_id in self.job_results:
                    job_info = self.job_results[job_id]
                    job_info["status"] = "completed"
                    job_info["end_time"] = time.time()

                    # Replace any None values with NaN
                    results = [0.0 if r is None else r for r in job_info["results"]]

                    # Call callback if provided
                    if job_info["callback"]:
                        job_info["callback"](results)

                    logger.info(
                        f"Job {job_id} completed in {job_info['end_time'] - job_info['start_time']:.2f}s "
                        f"with {results.count(0.0)} missing results"
                    )

        except Exception as e:
            # Job failed
            with self.job_lock:
                if job_id in self.job_results:
                    job_info = self.job_results[job_id]
                    job_info["status"] = "failed"
                    job_info["error"] = str(e)
                    job_info["end_time"] = time.time()

                    # Call callback with empty results if provided
                    if job_info["callback"]:
                        job_info["callback"]([float("nan")] * len(smiles_list))

                    logger.error(f"Job {job_id} failed: {str(e)}")

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of an asynchronous job.

        Args:
            job_id: Job ID to check

        Returns:
            Dict: Job status information
        """
        with self.job_lock:
            if job_id not in self.job_results:
                return {"status": "not_found"}

            job_info = self.job_results[job_id]
            return {
                "status": job_info["status"],
                "total_molecules": job_info["total_molecules"],
                "completed_molecules": job_info["completed_molecules"],
                "progress": job_info["completed_molecules"] / job_info["total_molecules"]
                if job_info["total_molecules"] > 0
                else 0,
                "start_time": job_info["start_time"],
                "elapsed_time": time.time() - job_info["start_time"],
                "error": job_info.get("error"),
            }

    def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> List[float]:
        """Wait for an asynchronous job to complete.

        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            List[float]: Results or empty list if job failed or timed out
        """
        start_time = time.time()
        while True:
            with self.job_lock:
                if job_id not in self.job_results:
                    return []

                job_info = self.job_results[job_id]
                if job_info["status"] == "completed":
                    # Return results and clean up
                    results = [0.0 if r is None else r for r in job_info["results"]]
                    return results

                if job_info["status"] == "failed":
                    return []

            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for job {job_id}")
                return []

            # Wait a bit before checking again
            time.sleep(0.5)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel an asynchronous job.

        Args:
            job_id: Job ID to cancel

        Returns:
            bool: True if job was cancelled, False if job not found
        """
        with self.job_lock:
            if job_id not in self.job_results:
                return False

            job_info = self.job_results[job_id]
            job_info["status"] = "cancelled"

            # Call callback with empty results if provided
            if job_info["callback"]:
                job_info["callback"]([float("nan")] * job_info["total_molecules"])

            logger.info(f"Job {job_id} cancelled")
            return True

    def clean_old_jobs(self, max_age: float = 3600):
        """Clean up old completed jobs.

        Args:
            max_age: Maximum age in seconds for jobs to keep
        """
        now = time.time()
        with self.job_lock:
            to_remove = []
            for job_id, job_info in self.job_results.items():
                if job_info["status"] in ["completed", "failed", "cancelled"]:
                    if "end_time" in job_info and now - job_info["end_time"] > max_age:
                        to_remove.append(job_id)

            for job_id in to_remove:
                del self.job_results[job_id]

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old jobs")

    def shutdown(self):
        """Shutdown the client and disconnect from all servers."""
        with self.server_lock:
            for server in self.servers:
                server.disconnect()

        logger.info("Client shutdown complete")


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proxy Client CLI")
    parser.add_argument("--registry", type=str, required=True, help="Path to server registry file")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for queries")
    parser.add_argument("--test", action="store_true", help="Run a simple test query")
    parser.add_argument("--test-timeout", action="store_true", help="Run a test query with timeout")

    args = parser.parse_args()

    # Create client
    client = ProxyClient(registry_file=args.registry, batch_size=args.batch_size)

    try:
        # Run a test query if requested
        if args.test:
            test_smiles = ["CC", "CCC", "CCCC", "CCCCC", "c1ccccc1"]
            logger.info(f"Running test query with {len(test_smiles)} molecules...")

            results = client.query(test_smiles)

            logger.info("Results:")
            for smiles, result in zip(test_smiles, results):
                logger.info(f"  {smiles}: {result}")

        # Run a test query with timeout if requested
        if args.test_timeout:
            from concurrent.futures import ThreadPoolExecutor

            test_smiles = ["CC", "CCC", "CCCC", "CCCCC", "c1ccccc1"]
            logger.info(f"Running test query with timeout for {len(test_smiles)} molecules...")

            with ThreadPoolExecutor() as executor:
                future = executor.submit(client.query, test_smiles)
                try:
                    results = future.result(timeout=10)  # Short 10 second timeout
                    logger.info(f"Got results: {results}")
                except concurrent.futures.TimeoutError:
                    logger.error("Query timed out!")
    finally:
        # Shutdown client
        client.shutdown()
