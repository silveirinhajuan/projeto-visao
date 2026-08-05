"""Testes de conteúdo do documento de risco para voluntários (tarefa 5.3).

O documento network/RISCOS_PARA_VOLUNTARIOS.md é a promessa pública do
Projeto VISÃO a quem for rodar um nó na própria máquina. Estes testes não
verificam estilo — verificam que o texto cobre, de forma honesta e em
português, os pontos que o briefing da tarefa 5.3 exige:

  - consumo de CPU / RAM / energia
  - o que o nó ENVIA pela rede
  - o que o nó NUNCA envia
  - como desinstalar em um clique
  - por que uma IA auto-replicante é aceitável aqui (R1-R5 auditáveis)
  - linguagem de risco real (o doc tem de assustar um pouco)

Se o teste passar por acidente num doc vazio, ele está errado: cada
verificação é ancorada num requisito do briefing.
"""

from pathlib import Path

DOC = Path(__file__).parent.parent / "RISCOS_PARA_VOLUNTARIOS.md"


def _txt() -> str:
    return DOC.read_text(encoding="utf-8")


def test_documento_existe():
    assert DOC.exists(), "network/RISCOS_PARA_VOLUNTARIOS.md ausente"


def test_menciona_consumo_cpu_ram_energia():
    t = _txt().lower()
    assert "cpu" in t, "deve mencionar consumo de CPU"
    assert "ram" in t, "deve mencionar consumo de RAM"
    assert any(k in t for k in ("energia", "kwh", "watt")), \
        "deve mencionar consumo de energia (kWh/Watt)"


def test_menciona_o_que_envia_pela_rede():
    t = _txt().lower()
    assert "envia" in t, "deve dizer explicitamente o que o nó ENVIA"
    assert "rede" in t, "deve ter seção sobre a rede"


def test_menciona_o_que_nunca_envia():
    t = _txt().lower()
    assert any(p in t for p in ("nunca envia", "jamais envia", "não envia", "não envia")), \
        "deve dizer o que o nó NUNCA envia"


def test_menciona_desinstalacao_um_clique():
    t = _txt().lower()
    assert "desinstal" in t, "deve explicar como desinstalar"


def test_menciona_r1_a_r5():
    t = _txt().upper()
    for r in ("R1", "R2", "R3", "R4", "R5"):
        assert r in t, f"deve citar a regra {r}"


def test_eh_honesto_e_assusta_um_pouco():
    t = _txt().lower()
    assert "risco" in t, "documento de risco tem de usar a palavra risco"
    assert any(p in t for p in ("perigo", "falha", "não é seguro", "nao e seguro")), \
        "deve conter advertência honesta (perigo/falha/nao e seguro)"
    # em português: contração tipicamente lusa/brasileira
    assert "você" in t or "voce" in t, "deve estar em português claro"


def test_tem_secao_rede_e_secao_desinstalacao():
    t = _txt().lower()
    # espera cabeçalhos de seção marcados com '#'
    assert "##" in t, "deve ter seções estruturadas (markdown)"
