import argparse
import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ProxyClusterManager")


class ProxyClusterManager:
    """Utility for managing the Proxy server cluster."""

    def __init__(self, script_dir: Optional[str] = None):
        """Initialize the cluster manager.

        Args:
            script_dir: Directory containing the scripts (default: current directory)
        """
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self.registry_file = os.path.join(self.script_dir, "server_registry.json")
        self.logs_dir = os.path.join(self.script_dir, "server_logs")

        # Ensure logs directory exists
        os.makedirs(self.logs_dir, exist_ok=True)

    def start_cluster(
        self,
        num_servers: int,
        proxy_config: str,
        slurm_config: str = "cpu_template.sh",
        base_port: int = 5555,
    ):
        """Start a new Proxy server cluster.

        Args:
            num_servers: Number of servers to start
            base_port: Base port number
        """
        # Check if registry already exists
        if os.path.exists(self.registry_file):
            registry = self._load_registry()
            if registry and any(server.get("status") != "terminated" for server in registry):
                logger.warning(
                    "Registry file exists with active servers. Use stop_cluster first or use --force."
                )
                return False

        # Launch the servers
        launch_script = os.path.join(self.script_dir, "launch.sh")
        if not os.path.exists(launch_script):
            logger.error(f"Launch script not found in {self.script_dir}")
            return False

        # Make sure the script is executable
        os.chmod(launch_script, 0o755)

        try:
            cmd = [
                launch_script,
                "--num-servers",
                str(num_servers),
                "--base-port",
                str(base_port),
                "--proxy-config",
                str(proxy_config),
                "--slurm-config",
                str(slurm_config),
            ]
            logger.info(f"Launching cluster with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            logger.info("Cluster started successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"All attempts to start cluster failed: {str(e)}")
            return False

    def stop_cluster(self, force: bool = False):
        """Stop the Proxy server cluster.

        Args:
            force: Force stop even if some servers are unreachable
        """
        registry = self._load_registry()
        if not registry:
            logger.warning("No registry file found. Nothing to stop.")
            return True

        success = True
        for server in registry:
            job_id = server.get("job_id")
            if not job_id:
                continue

            status = server.get("status")
            if status == "terminated":
                continue

            try:
                # Cancel the SLURM job
                logger.info(f"Stopping server {server.get('id')} (Job ID: {job_id})...")
                result = subprocess.run(["scancel", job_id], check=True, capture_output=True)

                # Update server status
                server["status"] = "terminated"

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to stop server {server.get('id')}: {e.stderr.decode()}")
                if not force:
                    success = False

        # Save updated registry
        self._save_registry(registry)

        if success:
            logger.info("All servers stopped successfully")
        else:
            logger.warning("Some servers could not be stopped")

        return success

    def status(self, verbose: bool = False):
        """Print cluster status.

        Args:
            verbose: Show detailed server information
        """
        registry = self._load_registry()
        if not registry:
            logger.info("No registry file found. Cluster is not running.")
            return

        active_servers = 0

        # Update server status from SLURM
        for server in registry:
            job_id = server.get("job_id")
            if not job_id:
                continue

            try:
                # Check job status
                result = subprocess.run(
                    ["squeue", "-j", job_id, "-o", "%T", "-h"],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0 and result.stdout.strip():
                    status = result.stdout.strip()
                    if status:
                        server["slurm_status"] = status
                        if status in ["RUNNING", "PENDING"]:
                            active_servers += 1
                    else:
                        server["slurm_status"] = "COMPLETED"
                else:
                    server["slurm_status"] = "UNKNOWN"
            except Exception as e:
                logger.error(f"Error checking status for job {job_id}: {str(e)}")
                server["slurm_status"] = "ERROR"

        # Print status
        logger.info(f"Cluster Status: {active_servers}/{len(registry)} servers active")

        if verbose:
            logger.info("\nServer Details:")
            for server in registry:
                server_id = server.get("id", "unknown")
                job_id = server.get("job_id", "unknown")
                hostname = server.get("hostname", "unknown")
                ip = server.get("ip_address", hostname)
                port = server.get("port", "unknown")
                status = server.get("slurm_status", "unknown")

                logger.info(
                    f"  Server {server_id}: Job {job_id} on {hostname} ({ip}:{port}) - Status: {status}"
                )

                # Check if log file exists
                log_file = os.path.join(self.logs_dir, f"proxy_server_{server_id}.log")
                if os.path.exists(log_file):
                    # Get last few lines of log
                    try:
                        with open(log_file, "r") as f:
                            # Seek to the end of file
                            f.seek(0, os.SEEK_END)
                            # Get the current position
                            pos = f.tell()
                            # Go back 1000 characters or to the beginning
                            pos = max(pos - 1000, 0)
                            f.seek(pos)
                            # If we're not at the beginning, discard the first partial line
                            if pos > 0:
                                f.readline()
                            # Get the last few lines
                            last_lines = f.read().strip().split("\n")[-3:]

                            if last_lines:
                                logger.info("    Last log entries:")
                                for line in last_lines:
                                    logger.info(f"      {line}")
                    except Exception as e:
                        logger.error(f"Error reading log file: {str(e)}")

        return active_servers

    def monitor(self, interval: int = 60):
        """Monitor the cluster continuously.

        Args:
            interval: Update interval in seconds
        """
        try:
            while True:
                active_servers = self.status(verbose=True)
                if active_servers == 0:
                    logger.warning("No active servers found. Exiting monitor.")
                    break

                logger.info(f"\nNext update in {interval} seconds. Press Ctrl+C to exit.\n")
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("\nMonitor stopped by user.")

    def test_cluster(self, batch_size=10, total_queries=50):
        """Test the cluster by running a simple query."""
        # Run the client with test flag
        from client import ProxyClient

        logger.info("Testing cluster connectivity...")

        client = ProxyClient(
            registry_file=self.registry_file,
            batch_size=batch_size,
        )

        try:
            # Generate some test SMILES
            test_smiles = ["c1ccccc1"]
            test_smiles = test_smiles * (max(1, total_queries // len(test_smiles)))

            logger.info(f"Querying cluster with {len(test_smiles)} test molecules...")
            logger.info(f"Available servers: {len(client.servers)}")

            # Add more detailed logging about servers
            with client.server_lock:
                for i, server in enumerate(client.servers):
                    logger.info(f"Server {i}: {server}, connected: {server.connected}")

            results = client.query(test_smiles)

            if len(results) == len(test_smiles):
                logger.info(f"Test successful: received {len(results)} results")

                # Print a few sample results
                for i in range(min(5, len(test_smiles))):
                    logger.info(f"  {test_smiles[i]}: {results[i]}")
                return True
            else:
                logger.error(
                    f"Test failed: expected {len(test_smiles)} results, got {len(results)}"
                )
                return False

        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}")
            return False
        finally:
            client.shutdown()

    def _load_registry(self) -> List[Dict[str, Any]]:
        """Load the registry file.

        Returns:
            List[Dict]: Registry data or empty list if file not found
        """
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, "r") as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading registry: {str(e)}")
            return []

    def _save_registry(self, registry: List[Dict[str, Any]]):
        """Save the registry file.

        Args:
            registry: Registry data to save
        """
        try:
            with open(self.registry_file, "w") as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")


# Main function for CLI
def main():
    parser = argparse.ArgumentParser(description="Proxy Cluster Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the Proxy cluster")
    start_parser.add_argument(
        "--num-servers", type=int, default=2, help="Number of servers to start"
    )
    start_parser.add_argument("--base-port", type=int, default=5555, help="Base port number")
    start_parser.add_argument(
        "--proxy-cfg",
        type=str,
        default="configs/proxies/target/qed.gin",
        help="Path to the gin configuration file for the Proxy server",
    )
    start_parser.add_argument(
        "--slurm-cfg",
        type=str,
        default="cpu_template.sh",
        help="Name of slurm script to launch remote nodes (co-located with this file)",
    )
    start_parser.add_argument(
        "--force", action="store_true", help="Force start even if registry exists"
    )

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the Proxy cluster")
    stop_parser.add_argument(
        "--force",
        action="store_true",
        help="Force stop even if some servers are unreachable",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show cluster status")
    status_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed information"
    )

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor the cluster continuously")
    monitor_parser.add_argument(
        "--interval", type=int, default=60, help="Update interval in seconds"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test the cluster by running a simple query")
    test_parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for testing queries"
    )
    test_parser.add_argument(
        "--total-queries", type=int, default=50, help="Total number of queries to test"
    )

    args = parser.parse_args()

    # Create manager
    manager = ProxyClusterManager()

    # Execute command
    if args.command == "start":
        manager.start_cluster(
            num_servers=args.num_servers,
            proxy_config=args.proxy_cfg,
            slurm_config=args.slurm_cfg,
            base_port=args.base_port,
        )
    elif args.command == "stop":
        manager.stop_cluster(force=args.force)
    elif args.command == "status":
        manager.status(verbose=args.verbose)
    elif args.command == "monitor":
        manager.monitor(interval=args.interval)
    elif args.command == "test":
        manager.test_cluster(batch_size=args.batch_size, total_queries=args.total_queries)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
