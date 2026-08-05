# Riscos Honestos para Voluntários — Projeto VISÃO

> Leia isto antes de instalar qualquer coisa. Não é marketing. É o que pode
> dar errado quando você roda uma IA auto-replicante e auto-modificável no
> seu próprio computador.
>
> Se depois de ler isto você achar que está tudo maravilhoso, **algo está
> errado com este documento** — e você deve nos avisar. Um documento de
> risco que não assusta um pouco é desonesto.

---

## O que é este nó, em uma frase

Você vai baixar um arquivo único (Fase 6) e rodá-lo. Ele treina uma rede
neural líquida (CfC) continuamente, em hardware modesto, e troca pequenas
atualizações de modelo com outros nós pares. Em troca, o modelo treinado
volta para você — você ganha a inteligência de volta, não um "obrigado".

Isto é uma IA **auto-replicante e auto-modificável**. Nós pedimos que você
rode isso na sua máquina. A única razão pela qual isto é aceitável está
descrita em `## Por que isto é aceitável (R1–R5)`. Leia antes de confiar.

---

## Consumo de CPU, RAM e energia

Não temos como medir a sua máquina de longe. Isto é o que sabemos por
projeto e o que você deve esperar:

- **CPU:** o treino é feito na CPU. Em repouso (só sincronizando), usa pouco.
  Durante o treino ativo, pode usar de 1 a 4 núcleos de forma sustentada.
  A arquitetura líquida é leve — ordens de magnitude menos parâmetros que
  um transformer (CfC tem ~10× a ~13× menos parâmetros que LSTM/GRU nas
  baterias medidas). Mesmo assim, **a ventoinha do seu laptop vai girar e a
  bateria vai durar menos** enquanto treina.
- **RAM:** depende do tamanho do modelo (Fase 1 usou ~96 neurônios em
  protótipo). Para um nó voluntário modesto, conte com centenas de MB a
  poucos GB. Não é um devorador de RAM, mas também não é grátis.
- **Energia:** um desktop sustentando ~200 GFLOPS (CPU+iGPU modestos) consome
  do seu orçamento elétrico. Estimativa grosseira: 50–100 W sustentados →
  de ~1,2 a ~2,4 kWh por dia se rodar 24/7. Na conta de luz, isto é algumas
  dezenas de reais por mês. **Se você paga por energia ou está num plano
  limitado, desligue o nó quando não quiser pagar por ele.**
- **Visibilidade:** o cliente **mostra CPU, RAM e energia estimada em tempo
  real**, sempre. Nada de esconder consumo — se o número não aparecer na tela,
  não é o cliente oficial.

Compromisso do projeto (Fase 5): *nada de esconder o consumo.*

---

## O que o nó ENVIA pela rede

Tudo o que sai da sua máquina é mensagem assinada do protocolo. Resumo:

- **Heartbeat:** um registro `(node_id, dia)` para o dashboard de churn.
  Só isso — seu ID de nó e o dia. Nada sobre o seu uso da máquina.
- **Atualizações de modelo (deltas de gradiente):** diferenças comprimidas
  (top-k + quantização) dos pesos, assinadas pelo seu `node_id`. É o "peso"
  do aprendizado, não dado seu.
- **Prova de contribuição:** um compromisso assinado com o gradiente antes do
  passo, para o agregador confirmar que você trabalhou de verdade.
- **Entradas no lineage ledger (R5):** cada nó registra sua linhagem
  (de onde veio) numa cadeia pública e encadeada.

Todo pacote é criptograficamente assinado. A intenção é que ninguém consiga
forjar ser você, e que a rede saiba de onde cada atualização veio.

---

## O que o nó NUNCA envia

Isto é o mais importante e o mais honesto que posso escrever:

- **NUNCA envia seus arquivos pessoais.** O agente não tem caminho de código
  para ler seu `/home`, seus documentos ou seu disco — só o diretório do
  próprio bundle.
- **NUNCA envia suas teclas, tela, microfone ou webcam.** Não há esse código.
- **NUNCA envia seus dados pessoais, senhas ou chaves.**
- **NUNCA abre conexão arbitrária de saída.** A camada de governança
  (`visao/governance/containment.py`) é a fronteira; fora do protocolo assinado,
  o nó não pode decidir enviar nada.
- **NUNCA replica a si mesmo sem autorização humana** (veja R1).

**Ressalva honesta:** estas garantias dependem do `containment.py` estar
correto e não ter sido adulterado. Por isso ele é imutável (R3) e verificado
a cada ciclo contra um manifesto selado. Se você não acredita em nós, **rode
os 39 testes da Fase 4 e veja os que tentam violar cada regra falharem**.
Não é promessa — é artefato.

---

## Como desinstalar em um clique

O nó é um **único diretório autocontido**. Nada é instalado no sistema
(`/`, registro do Windows, launchd, systemd) e nada pede `sudo`.

- Para parar: feche o cliente (ou `Ctrl+C`). Ele para. TTL à parte, ele não
  reinicia sozinho como serviço.
- Para remover de vez: **apague a pasta do bundle**. Pronto. Não sobra rastro,
  não fica serviço rodando, não há desinstalador mágico porque não houve
  instalação mágica.

Compromisso do projeto: *consentimento explícito, desinstalação em um clique.*
Se algum dia isto não for verdade, é um bug que deve ser reportado como
falha de contenção.

---

## Por que isto é aceitável (R1–R5 auditáveis)

Pedir para pessoas rodarem uma IA auto-replicante é pedir confiança. Nós
respondemos com **freios testados**, não com promessas. As cinco regras de
contenção, e o que cada uma entrega a você:

| Regra | O que impede | O que você ganha |
|---|---|---|
| **R1** quórum k-de-n | O nó **não replica** sem assinatura de operadores humanos em hardware | Ninguém (nem o próprio sistema) autoriza crescer a rede sozinho |
| **R2** TTL (vida útil) | Sem renovação assinada, o nó **para sozinho** | Falha fechado: relógio vencido = nó morto, sem graceful "esquece" |
| **R3** fronteira imutável | O código de governança não é evoluível | Alterar `containment.py` = parada imediata (hash checado todo ciclo) |
| **R4** freio de PoW | Taxa de replicação limitada por prova de trabalho crescente | Replicar 8× custa ~10.000× mais que 1× — explosão é cara demais |
| **R5** ledger público | Toda linhagem é encadeada e visível | Nó silencioso é expulso; você pode auditar de onde veio cada nó |

O princípio de projeto é **TUDO FALHA FECHADO**: erro, ambiguidade, relógio
ilegível, token malformado → a resposta é sempre *negar*. O sistema nunca
permite por omissão.

**Contexto que torna isto obrigatório e não paranoia:** Palisade Research
(mai/2026) mediu modelos de fronteira replicando-se autonomamente em VMs
reais a 81% de sucesso; pesos abertos chegaram a ~70%. Auto-replicação deixou
de ser capacidade de fronteira. É por isso que a Fase 4 foi construída antes
desta, e é o argumento principal de recrutamento — não um rodapé legal.

---

## O que pode dar errado de verdade (sem melar)

1. **Sua máquina fica mais lenta ou quente.** CPU e RAM são reais. Se atrapalhar
   seu uso, pare o nó. Ele não se ofende.
2. **Sua conta de energia sobe.** Veja `## Consumo`. É previsível, não oculto.
3. **Uso de banda.** As sincronias de gradiente consomem rede. Em conexão
   medida (3G/4G/5G limitado), isto sai do seu pacote. O nó não é um vídeo 4K,
   mas também não é zero.
4. **Metadados visíveis.** Seu tráfego de protocolo pode ser visto por quem
   está na mesma rede ou pelo seu provedor. As mensagens são *assinadas*, não
   necessariamente *camufladas* — não espere anonimato de rede.
5. **Um release nosso com bug** pode usar mais recursos do que o planejado. O
   pior caso é o nó travar ou parar; ele **não pode** ultrapassar a fronteira
   R3. Se ultrapassar, é falha de contenção e deve ser reportada.
6. **A segurança depende das chaves humanas.** R1 só funciona enquanto as
   assinaturas ficarem em hardware humano (YubiKey/Ledger). Se essas chaves
   forem comprometidas, o freio de replicação degrada. Isto é uma dependência
   humana, não uma garantia matemática absoluta.
7. **Este é um projeto de pesquisa, não um produto.** Pode ter bugs. Pode
   mudar de direção. Pode parar. Não prometemos que é "seguro" — prometemos
   que é **auditável** e que os freios são testados.

---

## O que NÃO vamos fazer (e por que importa para você)

- **Zero token/criptomoeda.** Atrai minerador, não colaborador; corrompe a
  rede. Você não será pago — e isso é proposital.
- **Nada de instalação silenciosa, bundle ou opt-out.** Você instala porque
  decidiu, e remove porque decidiu.
- **Nada de prometer que é seguro.** Prometemos que é auditável. Diferença
  enorme.

---

## Verificação independente (faça você mesmo)

Não confie em nós. Rode:

```bash
python3 -m pytest visao/governance/tests/test_containment.py -q
```

Esperado: 39 testes verdes, incluindo os que **tentam** violar R1–R5 e
exigem que falhem. Se algum passar quando deveria falhar, ou se o hash do
`containment.py` não bater com o manifesto, **não rode o nó**.

---

*Última linha honesta: uma IA auto-replicante na sua máquina é uma decisão
sua, não nossa. Este documento existe para que você a tome sabendo o que
custa, o que sai, o que fica, e o que nos impede de sair do controle.*
