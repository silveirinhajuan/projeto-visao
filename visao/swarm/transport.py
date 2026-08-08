"""Transporte local do enxame (Fase 2.1).

Por enquanto: descoberta de porta livre para o socket local entre processos
na mesma maquina. O transporte P2P real (libp2p/Iroh + handshake R3) entra em
sub-tarefa da 2.3.
"""
import socket


def free_port() -> int:
    """Porta TCP livre na interface loopback para uso local."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
