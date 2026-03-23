#!/usr/bin/env python3
"""Generate a self-signed TLS certificate for the Aize HTTP mesh.

Usage (standalone):
    python3 -m tls.gen_self_signed_cert [--cert CERT_PATH] [--key KEY_PATH] [--days DAYS] [--hosts HOSTNAME ...]

Defaults:
    cert:  $AIZE_RUNTIME_ROOT/tls/server.crt  (fallback: ./.agent-mesh-runtime/tls/server.crt)
    key:   $AIZE_RUNTIME_ROOT/tls/server.key
    days:  3650
    hosts: localhost 127.0.0.1 (always included; extra hosts added via --hosts)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _default_runtime_root() -> Path:
    root = os.environ.get("AIZE_ROOT")
    if root:
        base = Path(root)
    else:
        base = Path(__file__).resolve().parents[2]
    runtime = os.environ.get("AIZE_RUNTIME_ROOT")
    if runtime:
        return Path(runtime)
    return base / ".agent-mesh-runtime"


def _build_san(extra_hosts: list[str] | None = None) -> str:
    """Build a subjectAltName string covering localhost, 127.0.0.1, and any extra hosts."""
    import ipaddress

    dns_names = ["localhost"]
    ip_addrs = ["127.0.0.1"]
    for h in (extra_hosts or []):
        h = h.strip()
        if not h:
            continue
        try:
            ipaddress.ip_address(h)
            if h not in ip_addrs:
                ip_addrs.append(h)
        except ValueError:
            if h not in dns_names:
                dns_names.append(h)
    parts = [f"DNS:{n}" for n in dns_names] + [f"IP:{a}" for a in ip_addrs]
    return "subjectAltName=" + ",".join(parts)


def generate_self_signed_cert(
    cert_path: Path,
    key_path: Path,
    *,
    days: int = 3650,
    cn: str = "localhost",
    extra_hosts: list[str] | None = None,
) -> None:
    """Generate a self-signed cert+key pair using openssl.

    Creates parent directories as needed.  Overwrites existing files.
    Always adds SAN for DNS:localhost and IP:127.0.0.1.
    Pass extra_hosts to include additional DNS names or IP addresses in the SAN.
    """
    cert_path = Path(cert_path)
    key_path = Path(key_path)
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    san = _build_san(extra_hosts)

    # Try modern openssl (>=1.1.1) first: supports -addext inline
    result = subprocess.run(
        [
            "openssl", "req", "-x509",
            "-newkey", "rsa:2048",
            "-keyout", str(key_path),
            "-out", str(cert_path),
            "-days", str(days),
            "-nodes",
            "-subj", f"/CN={cn}",
            "-addext", san,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Fallback: write an openssl config file with SAN extension
        cfg = (
            "[req]\n"
            "distinguished_name=dn\n"
            "x509_extensions=v3_req\n"
            "prompt=no\n"
            "[dn]\n"
            f"CN={cn}\n"
            "[v3_req]\n"
            f"{san}\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as f:
            f.write(cfg)
            cfg_path = f.name
        try:
            subprocess.run(
                [
                    "openssl", "req", "-x509",
                    "-newkey", "rsa:2048",
                    "-keyout", str(key_path),
                    "-out", str(cert_path),
                    "-days", str(days),
                    "-nodes",
                    "-config", cfg_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        finally:
            Path(cfg_path).unlink(missing_ok=True)


def main() -> int:
    tls_dir = _default_runtime_root() / "tls"
    parser = argparse.ArgumentParser(description="Generate a self-signed TLS certificate.")
    parser.add_argument("--cert", default=str(tls_dir / "server.crt"), help="Output path for the certificate (PEM)")
    parser.add_argument("--key", default=str(tls_dir / "server.key"), help="Output path for the private key (PEM)")
    parser.add_argument("--days", type=int, default=3650, help="Certificate validity in days (default: 3650)")
    parser.add_argument("--cn", default="localhost", help="Common name (default: localhost)")
    parser.add_argument("--hosts", nargs="*", default=[], metavar="HOST",
                        help="Additional DNS names or IP addresses to add to the SAN (space-separated)")
    args = parser.parse_args()

    cert_path = Path(args.cert)
    key_path = Path(args.key)
    generate_self_signed_cert(cert_path, key_path, days=args.days, cn=args.cn, extra_hosts=args.hosts)
    print(f"cert: {cert_path}")
    print(f"key:  {key_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
