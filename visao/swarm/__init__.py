"""visao.swarm - coordenacao descentralizada do organismo liquido (Fase 2).

No local: dois processos-worker na mesma maquina trocam deltas por socket local,
sem expor rede. P2P real e handshake R3 entram em sub-tarefas posteriores.
"""
